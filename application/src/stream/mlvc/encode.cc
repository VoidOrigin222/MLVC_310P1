#include <mlvc/application/input/frame_input_queue.h>
#include <mlvc/application/pipeline/codec_frame_pipeline.h>
#include <mlvc/application/runtime/mlvc_codec_runtime.h>
#include <mlvc/application/stream/encode/encode_frame.h>
#include <mlvc/application/stream/encode/encode_output.h>
#include <mlvc/application/stream/encode/encode_schedule.h>
#include <mlvc/application/stream/encode/encode_setup.h>
#include <mlvc/application/stream/encode/encode_state.h>
#include <mlvc/application/stream/mlvc_internal.h>
#include <mlvc/application/stream/mlvc_stream.h>
#include <mlvc/codec/detail/stage/stage_runtime_state.h>
#include <mlvc/codec/mlvc_rate_control.h>
#include <mlvc/entropy/mlvc_official_entropy.h>
#include <mlvc/framework/codec_graph_executor.h>
#include <mlvc/framework/entropy_worker.h>
#include <mlvc/framework/profiler.h>
#include <mlvc/runtime/stage_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>

#include "mlvc/core/status.h"

namespace mlvc::codec {

int RunEncodeStream(const EncodeStreamOptions& options, EncodePipelineServices* services) {
  ValidateEncodeInput(options);
  try {
    ScopedRuntimeState runtime_state_guard;
    std::unique_ptr<mlvc::app::MlvcCodecRuntime> owned_codec_runtime;
    mlvc::app::MlvcCodecRuntime* codec_runtime =
        services == nullptr ? nullptr : services->codec_runtime;
    if (codec_runtime == nullptr) {
      owned_codec_runtime = std::make_unique<mlvc::app::MlvcCodecRuntime>(
          options.manifest_path, options.device, mlvc::codec::StageOutputBindingMode::kCpu);
      codec_runtime = owned_codec_runtime.get();
    }

    mlvc::StageRuntime& runtime = codec_runtime->runtime();
    mlvc::StageModelSet& models = codec_runtime->models();
    const mlvc::RuntimeSidecar& sidecar = codec_runtime->sidecar();
    Check(HasMlvcModels(models.manifest()),
          "manifest does not contain MLVCEncoder / MLVCDecoder");
    const mlvc::ModelRecord& encoder_record = models.manifest().GetModel("MLVCEncoder");
    ConfigureRuntimeState(&codec_runtime->stage_output_workspace(), runtime,
                          options.enable_stage_fusion);

    const mlvc::TensorSpec& frame_spec = encoder_record.inputs.at(0);
    double fps = options.fps;
    const SourceFrameGeometry source_geometry =
        ResolveSourceGeometry(options.input_video_path, options.input_frame_dir, frame_spec, &fps);
    const EncodeDimensions dimensions{
        static_cast<int>(encoder_record.outputs.at(2).shape.at(1) * 2),
        static_cast<int>(encoder_record.outputs.at(2).shape.at(2)),
        static_cast<int>(encoder_record.outputs.at(2).shape.at(3)),
        static_cast<int>(encoder_record.outputs.at(1).shape.at(2)),
        static_cast<int>(encoder_record.outputs.at(1).shape.at(3))};

    std::optional<mlvc::Profiler> owned_profiler;
    std::optional<mlvc::CodecGraphExecutor> owned_graph_executor;
    std::optional<mlvc::EntropyWorker> owned_entropy_worker;
    EncodePipelineServices owned_services;
    if (services == nullptr) {
      owned_profiler.emplace();
      owned_graph_executor.emplace(options.pipeline.graph_packet_capacity);
      owned_entropy_worker.emplace(options.pipeline.entropy_workers);
      owned_services = EncodePipelineServices{&*owned_profiler, &*owned_graph_executor,
                                              &*owned_entropy_worker, codec_runtime};
      services = &owned_services;
    }
    Check(services->profiler != nullptr && services->graph_executor != nullptr &&
              services->entropy_worker != nullptr,
          "encode pipeline services must provide profiler, graph executor, and entropy worker");
    mlvc::Profiler& profiler = *services->profiler;
    mlvc::CodecGraphExecutor& graph_executor = *services->graph_executor;
    mlvc::EntropyWorker& entropy_worker = *services->entropy_worker;
    profiler.ReserveEvents(256);
    g_codec_graph_executor = &graph_executor;
    g_codec_graph_executor->RecordTemplate(&profiler);

    const mlvc::io::MlvcBitstreamHeader header =
        BuildEncodeHeader(options, source_geometry, fps);
    MlvcRateControlOptions rate_options;
    rate_options.width = source_geometry.width;
    rate_options.height = source_geometry.height;
    rate_options.fps = fps;
    rate_options.target_bitrate_bps = options.target_bitrate_bps;
    rate_options.default_q_index = options.qp;
    rate_options.min_q_index = options.min_qp;
    rate_options.max_q_index = options.max_qp;
    EncodeOutput output(options, header, rate_options, &profiler);
    MlvcOfficialEntropyEncoder entropy_encoder(models.manifest().directory());
    EncodeState state(encoder_record.outputs.at(0).shape);
    EncodeFrameProcessor frame_processor(
        options, &models, &sidecar, &profiler, &entropy_worker, &state,
        &output.rate_controller(), &entropy_encoder, dimensions, fps);

    const int frames_to_attempt =
        options.frame_num > 0 ? options.frame_num : std::numeric_limits<int>::max();
    mlvc::app::FrameInputArena frame_prepare_arena(frame_spec, options.pipeline.frame_buffer_slots);
    mlvc::app::AsyncFrameInputQueue prepare_queue(
        frames_to_attempt, options.frame_num, options.input_video_path, options.input_frame_dir,
        frame_spec, source_geometry.width, source_geometry.height, &frame_prepare_arena, &profiler,
        &graph_executor);
    std::atomic<bool> frame_pipeline_running{true};
    mlvc::StreamingPipeline frame_pipeline(options.pipeline.stream_workers,
                                           options.pipeline.queue_capacity);
    frame_pipeline.SetProcessor([](const std::shared_ptr<mlvc::DataObject>& input) {
      mlvc::Check(input != nullptr, "frame pipeline received an empty input");
      return input;
    });
    frame_pipeline.Start();
    mlvc::app::PreparedFrameProducer frame_producer(frame_pipeline, frame_pipeline_running,
                                                     prepare_queue);
    frame_producer.Start();

    const auto encode_start = std::chrono::steady_clock::now();
    std::deque<PendingEncodedFrame> pending_entropy;
    constexpr std::size_t kMaxPendingEntropyJobs = 2;
    int encoded_frames = 0;
    int i_frames = 0;
    int p_frames = 0;
    int ltr_frames = 0;
    uint64_t q_index_sum = 0;
    int min_q_index_used = std::numeric_limits<int>::max();
    int max_q_index_used = std::numeric_limits<int>::min();
    auto consume_frame = [&](const std::shared_ptr<mlvc::DataObject>& data) {
      PendingEncodedFrame pending = frame_processor.Process(data, encoded_frames);
      if (pending.frame_type == MlvcFrameType::kIFrame) {
        ++i_frames;
      } else if (pending.frame_type == MlvcFrameType::kLtrRecovery) {
        ++ltr_frames;
      } else {
        ++p_frames;
      }
      q_index_sum += static_cast<uint64_t>(pending.q_index);
      min_q_index_used = std::min(min_q_index_used, pending.q_index);
      max_q_index_used = std::max(max_q_index_used, pending.q_index);
      ++encoded_frames;
      pending_entropy.push_back(std::move(pending));
      if (ShouldRetirePendingEntropy(pending_entropy.size(), kMaxPendingEntropyJobs)) {
        output.Flush(std::move(pending_entropy.front()));
        pending_entropy.pop_front();
      }
    };
    mlvc::app::CallbackDataConsumer frame_consumer(frame_pipeline, frame_pipeline_running,
                                                    consume_frame);
    frame_consumer.Start();
    frame_producer.Join();
    frame_producer.RethrowIfFailed();
    frame_consumer.Join();
    frame_consumer.RethrowIfFailed();
    frame_pipeline.Stop();
    while (!pending_entropy.empty()) {
      output.Flush(std::move(pending_entropy.front()));
      pending_entropy.pop_front();
    }
    output.SendEnd();
    output.Close();

    const auto encode_end = std::chrono::steady_clock::now();
    if (!options.profile_output_path.empty()) profiler.WriteChromeTrace(options.profile_output_path);
    const double seconds = std::chrono::duration<double>(encode_end - encode_start).count();
    std::cout << "mode=mlvc_encode\n";
    std::cout << "input_frame_dir=" << options.input_frame_dir << "\n";
    std::cout << "input_video=" << options.input_video_path << "\n";
    std::cout << "qp=" << options.qp << "\n";
    std::cout << "gop=" << options.gop << "\n";
    std::cout << "reset_interval=" << options.reset_interval << "\n";
    std::cout << "ltr_start_idx=" << options.ltr_start_idx << "\n";
    std::cout << "ltr_period=" << options.ltr_period << "\n";
    std::cout << "ltr_qp_shift=" << options.ltr_qp_shift << "\n";
    std::cout << "target_bitrate_bps=" << options.target_bitrate_bps << "\n";
    std::cout << "min_qp=" << options.min_qp << "\n";
    std::cout << "max_qp=" << options.max_qp << "\n";
    std::cout << "forced_ltr_recovery_frame=" << options.forced_ltr_recovery_frame << "\n";
    std::cout << "forced_ltr_reference_frame=" << options.forced_ltr_reference_frame << "\n";
    std::cout << "frames=" << encoded_frames << "\n";
    std::cout << "frame_counts=i:" << i_frames << ",p:" << p_frames << ",ltr:" << ltr_frames
              << "\n";
    std::cout << "encode_fps="
              << (seconds > 0.0 ? static_cast<double>(encoded_frames) / seconds : 0.0) << "\n";
    std::cout << "bitstream_bytes=" << output.file_bytes() << "\n";
    std::cout << "payload_bytes=" << output.payload_bytes() << "\n";
    std::cout << "q_index_min=" << (encoded_frames > 0 ? min_q_index_used : 0) << "\n";
    std::cout << "q_index_max=" << (encoded_frames > 0 ? max_q_index_used : 0) << "\n";
    std::cout << "q_index_mean="
              << (encoded_frames > 0 ? static_cast<double>(q_index_sum) / encoded_frames : 0.0)
              << "\n";
    std::cout << "average_bpp="
              << (encoded_frames > 0
                      ? static_cast<double>(output.file_bytes()) * 8.0 /
                            (static_cast<double>(source_geometry.width) * source_geometry.height *
                             encoded_frames)
                      : 0.0)
              << "\n";
    std::cout << "encode=ok\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "MLVC encode failed: " << error.what() << "\n";
    return 1;
  }
}

}  // namespace mlvc::codec
