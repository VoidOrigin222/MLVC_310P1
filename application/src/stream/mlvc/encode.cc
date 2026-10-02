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
#include <mlvc/codec/translation_warp.h>
#include <mlvc/io/fp16_yuv444_to_nv12.h>

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
    const int sidecar_q_index_count = sidecar.q_index_count();
    mlvc::io::ValidateMlvcQIndexForSidecar(options.qp, sidecar_q_index_count);
    mlvc::io::ValidateMlvcQIndexForSidecar(options.min_qp, sidecar_q_index_count);
    mlvc::io::ValidateMlvcQIndexForSidecar(options.max_qp, sidecar_q_index_count);
    Check(HasMlvcModels(models.manifest()), "manifest does not contain MLVCEncoder / MLVCDecoder");
    if (options.translation_warp) RequireTranslationWarpModels(models.manifest());
    const mlvc::ModelRecord& encoder_record = models.manifest().GetModel("MLVCEncoder");
    ConfigureRuntimeState(&codec_runtime->stage_output_workspace(), runtime,
                          options.enable_stage_fusion);

    const mlvc::TensorSpec& frame_spec = encoder_record.inputs.at(0);
    double fps = options.fps;
    auto source_camera_options = options.camera_options;
    if (source_camera_options)
      source_camera_options->motion_nv12 = options.motion_camera_nv12;
    const SourceFrameGeometry source_geometry =
        ResolveSourceGeometry(options.input_video_path, options.input_frame_dir,
                              source_camera_options, runtime.context(), frame_spec, &fps);
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

    mlvc::io::MlvcBitstreamHeader header = BuildEncodeHeader(options, source_geometry, fps);
    header.codec_bundle_sha256 = mlvc::ComputeModelBundleSha256(models.manifest());
    const mlvc::ModelRecord& decoder_record = models.manifest().GetModel("MLVCDecoder");
    const auto decoder_output = std::find_if(
        decoder_record.outputs.begin(), decoder_record.outputs.end(),
        [](const mlvc::TensorSpec& spec) { return spec.name == "x_hat"; });
    Check(decoder_output != decoder_record.outputs.end() && decoder_output->shape.size() == 4 &&
              decoder_output->shape[0] == 1 && decoder_output->shape[1] == 3 &&
              decoder_output->shape[2] > 0 && decoder_output->shape[3] > 0,
          "MLVCDecoder x_hat output shape is missing or invalid");
    header.coded_height = static_cast<int>(decoder_output->shape[2]);
    header.coded_width = static_cast<int>(decoder_output->shape[3]);
    mlvc::io::ValidateMlvcBitstreamHeader(header);
    MlvcRateControlOptions rate_options;
    rate_options.width = source_geometry.width;
    rate_options.height = source_geometry.height;
    rate_options.fps = fps;
    rate_options.target_bitrate_bps = options.target_bitrate_bps;
    rate_options.default_q_index = options.qp;
    rate_options.min_q_index = options.min_qp;
    rate_options.max_q_index = options.max_qp;
    EncodeOutput output(options, header, rate_options, &profiler);
    MlvcOfficialEntropyEncoder entropy_encoder(models.manifest());
    EncodeState state(encoder_record.outputs.at(0).shape);
    EncodeFrameProcessor frame_processor(options, &models, &sidecar, &profiler, &entropy_worker,
                                         &state, &output.rate_controller(), &entropy_encoder,
                                         dimensions, fps, source_geometry, runtime.context());

    const int frames_to_attempt =
        options.frame_num > 0 ? options.frame_num : std::numeric_limits<int>::max();
    mlvc::app::FrameInputArena frame_prepare_arena(
        frame_spec, options.motion_prefetch_frames > 0 ? options.motion_prefetch_frames + 1
                                                      : options.pipeline.frame_buffer_slots,
        !options.camera_options.has_value());
    mlvc::app::AsyncFrameInputQueue::MotionInputPrepareFunction motion_input_prepare;
    if (options.motion_prefetch_frames > 0 && options.motion_shifts_file.empty() &&
        !options.motion_camera_nv12) {
      const mlvc::io::Nv12Layout layout{source_geometry.width, source_geometry.height,
                                       source_geometry.width, source_geometry.height};
      motion_input_prepare = [layout](
          const TensorData& input, std::vector<uint8_t>* nv12) {
        const auto location = input.View().location();
        Check((location == MemoryLocation::kCpu || location == MemoryLocation::kPinnedCpu) &&
                  input.dtype == DataType::kFloat16 &&
                  input.ByteSize() == input.shape.NumElements() * sizeof(uint16_t),
              "motion proxy FP16 input must be a valid host tensor");
        mlvc::io::ConvertFp16Yuv444ToNv12(input, layout, nv12);
      };
    }
    mlvc::app::AsyncFrameInputQueue prepare_queue(
        frames_to_attempt, options.frame_num, options.input_video_path, options.input_frame_dir,
        source_camera_options, runtime.context(), frame_spec, source_geometry.width,
        source_geometry.height,
        &frame_prepare_arena, &profiler, &graph_executor, std::move(motion_input_prepare));
    std::atomic<bool> frame_pipeline_running{true};
    // Stateful H264 motion must run on exactly one worker. Arena slots include
    // source preparation, motion queues, and the current codec frame.
    mlvc::StreamingPipeline frame_pipeline(options.motion_prefetch_frames > 0 ? 1
                                                                          : options.pipeline.stream_workers,
                                           options.pipeline.queue_capacity);
    frame_pipeline.SetProcessor([&](const std::shared_ptr<mlvc::DataObject>& input)
        -> std::shared_ptr<mlvc::DataObject> {
      try {
        prepare_queue.RethrowIfFailed();
        if (!frame_pipeline_running.load()) return nullptr;
        mlvc::Check(input != nullptr, "frame pipeline received an empty input");
        if (options.motion_prefetch_frames > 0) frame_processor.PrepareMotion(input);
        return input;
      } catch (...) {
        prepare_queue.Cancel(std::current_exception());
        frame_pipeline_running.store(false);
        frame_pipeline.CloseInput();  // Stop cannot be called from its own worker.
        throw;
      }
    });
    frame_pipeline.Start();
    mlvc::app::PreparedFrameProducer frame_producer(frame_pipeline, frame_pipeline_running,
                                                    prepare_queue);

    std::chrono::steady_clock::time_point encode_start = std::chrono::steady_clock::now();
    bool timing_started = options.profile_warmup_frames == 0;
    std::deque<PendingEncodedFrame> pending_entropy;
    constexpr std::size_t kMaxPendingEntropyJobs = 2;
    uint64_t entropy_ready_retirements = 0;
    const auto retire_ready_entropy = [&] {
      if (options.motion_prefetch_frames == 0 || output.rate_controller().enabled()) return;
      while (!pending_entropy.empty() &&
             ShouldRetireReadyEntropy(options.motion_prefetch_frames,
                 output.rate_controller().enabled(),
                 pending_entropy.front().payload.wait_for(std::chrono::milliseconds(0)))) {
        // Only inspect the front: writing a later ready frame would break
        // reference order. Flush may still wait for file/network output.
        output.Flush(std::move(pending_entropy.front()));
        pending_entropy.pop_front();
        ++entropy_ready_retirements;
      }
    };
    int encoded_frames = 0;
    int i_frames = 0;
    int p_frames = 0;
    int ltr_frames = 0;
    uint64_t q_index_sum = 0;
    int min_q_index_used = std::numeric_limits<int>::max();
    int max_q_index_used = std::numeric_limits<int>::min();
    std::deque<mlvc::transport::MlvcControlMessage> pending_controls;
    auto progress_start = std::chrono::steady_clock::now();
    int progress_frame = 0;
    auto consume_frame = [&](const std::shared_ptr<mlvc::DataObject>& data) {
      prepare_queue.RethrowIfFailed();
      frame_pipeline.RethrowIfFailed();
      retire_ready_entropy();
      mlvc::transport::MlvcControlMessage control;
      while (output.PopMlvcControl(&control)) pending_controls.push_back(std::move(control));
      for (auto it = pending_controls.begin(); it != pending_controls.end();) {
        const bool due = it->apply_after_frame_id == 0xffffffffu ||
                         encoded_frames >= static_cast<int>(it->apply_after_frame_id);
        if (!due) { ++it; continue; }
        const bool changes_gop = std::any_of(it->tlvs.begin(), it->tlvs.end(),
            [](const auto& tlv) { return tlv.type == 3 || tlv.type == 4; });
        const bool accepted = !(options.translation_warp && changes_gop) &&
                              state.ApplyControl(*it, encoded_frames);
        output.SendMlvcControlResponse(*it, accepted,
                                        accepted ? "applied" : "unsupported-or-invalid");
        it = pending_controls.erase(it);
      }
      const bool random_access_request = output.ConsumeRandomAccessRequest();
      state.SetForceRandomAccess(random_access_request && !options.translation_warp);
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
      retire_ready_entropy();
      if (ShouldRetirePendingEntropy(pending_entropy.size(), kMaxPendingEntropyJobs)) {
        output.Flush(std::move(pending_entropy.front()));
        pending_entropy.pop_front();
      }
      if (!timing_started && encoded_frames >= options.profile_warmup_frames) {
        encode_start = std::chrono::steady_clock::now();
        timing_started = true;
      }
      if (encoded_frames % 900 == 0) {
        const auto now = std::chrono::steady_clock::now();
        const double interval_seconds =
            std::chrono::duration<double>(now - progress_start).count();
        const auto input_stats = prepare_queue.stats();
        std::cerr << "encode_progress frames=" << encoded_frames
                  << " interval_fps="
                  << (interval_seconds > 0.0
                          ? static_cast<double>(encoded_frames - progress_frame) /
                                interval_seconds
                          : 0.0)
                  << " input_ready_max_depth=" << input_stats.ready_max_depth
                  << " input_full_waits=" << input_stats.producer_full_wait_count
                  << " input_empty_waits=" << input_stats.consumer_empty_wait_count
                  << std::endl;
        progress_start = now;
        progress_frame = encoded_frames;
      }
    };
    mlvc::app::CallbackDataConsumer frame_consumer(frame_pipeline, frame_pipeline_running,
                                                   consume_frame, [&] {
      prepare_queue.Cancel(std::current_exception());
      frame_pipeline_running.store(false);
      frame_pipeline.CloseInput();
    });
    frame_consumer.Start();
    frame_producer.Start();
    frame_producer.Join();
    frame_consumer.Join();
    frame_producer.RethrowIfFailed();
    frame_consumer.RethrowIfFailed();
    frame_pipeline.Stop();
    frame_pipeline.RethrowIfFailed();
    frame_processor.ValidateMotionFrameCount(encoded_frames);
    while (!pending_entropy.empty()) {
      output.Flush(std::move(pending_entropy.front()));
      pending_entropy.pop_front();
    }
    output.SendEnd();
    output.Close();

    const auto encode_end = std::chrono::steady_clock::now();
    if (!options.profile_output_path.empty())
      profiler.WriteChromeTrace(options.profile_output_path);
    const int measured_frames = std::max(0, encoded_frames - options.profile_warmup_frames);
    const double seconds =
        timing_started ? std::chrono::duration<double>(encode_end - encode_start).count() : 0.0;
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
    std::cout << "warmup_frames=" << options.profile_warmup_frames << "\n";
    std::cout << "measured_frames=" << measured_frames << "\n";
    std::cout << "frame_counts=i:" << i_frames << ",p:" << p_frames << ",ltr:" << ltr_frames
              << "\n";
    std::cout << "encode_fps="
              << (seconds > 0.0 ? static_cast<double>(measured_frames) / seconds : 0.0) << "\n";
    const auto input_stats = prepare_queue.stats();
    std::cout << "input_queue prepared_frames=" << input_stats.prepared_frames
              << " ready_max_depth=" << input_stats.ready_max_depth
              << " producer_full_waits=" << input_stats.producer_full_wait_count
              << " producer_full_wait_total_ms=" << input_stats.producer_full_wait_ms
              << " producer_full_wait_max_ms=" << input_stats.producer_full_wait_max_ms
              << " consumer_empty_waits=" << input_stats.consumer_empty_wait_count
              << " consumer_empty_wait_total_ms=" << input_stats.consumer_empty_wait_ms
              << " consumer_empty_wait_max_ms=" << input_stats.consumer_empty_wait_max_ms
              << "\n";
    std::cout << "bitstream_bytes=" << output.file_bytes() << "\n";
    std::cout << "payload_bytes=" << output.payload_bytes() << "\n";
    if (options.translation_warp) {
      const uint64_t geometry_bytes = 2u * static_cast<uint64_t>(p_frames);
      std::cout << "translation_warp=true\n";
      const auto print_latency = [](const char* name, const LatencyStats& stats) {
        std::cout << name << "_frames=" << stats.count << "\n"
                  << name << "_total_ms=" << stats.total_ms << "\n"
                  << name << "_avg_ms=" << stats.average_ms() << "\n"
                  << name << "_max_ms=" << stats.max_ms << "\n";
      };
      std::cout << "motion_camera_nv12=" << (options.motion_camera_nv12 ? "true" : "false") << "\n";
      std::cout << "motion_proxy_width=" << source_geometry.width << "\n";
      std::cout << "motion_proxy_height=" << source_geometry.height << "\n";
      std::cout << "entropy_ready_retirements=" << entropy_ready_retirements << "\n";
      std::cout << "motion_prefetch_frames=" << options.motion_prefetch_frames << "\n";
      std::cout << "input_max_acquired_frames=" << input_stats.max_acquired_frames << "\n";
      std::cout << "input_max_ahead_frames="
                << (input_stats.max_acquired_frames ? input_stats.max_acquired_frames - 1 : 0) << "\n";
      std::cout << "motion_input_prepare_total_ms=" << input_stats.motion_prepare_ms << "\n";
      std::cout << "motion_input_prepare_avg_ms="
                << (input_stats.prepared_frames ? input_stats.motion_prepare_ms / input_stats.prepared_frames : 0.0) << "\n";
      std::cout << "motion_input_prepare_max_ms=" << input_stats.motion_prepare_max_ms << "\n";
      std::cout << "motion_consumer_wait_total_ms=" << frame_consumer.wait_ms() << "\n";
      std::cout << "motion_consumer_wait_max_ms=" << frame_consumer.wait_max_ms() << "\n";
      print_latency("motion_work", frame_processor.motion_work());
      print_latency("ready_to_motion", frame_processor.ready_to_motion());
      print_latency("ready_to_encode", frame_processor.ready_to_encode());
      print_latency("ready_to_output", output.ready_to_output());
      std::cout << "latency_boundary=source_tensor_ready_to_codec_complete_or_file_write_rtp_enqueue\n";
      std::cout << "motion_source=" << (options.motion_shifts_file.empty() ? options.motion_backend : "csv-replay") << "\n";
      if (options.motion_backend == "libx264" && options.motion_shifts_file.empty()) {
        std::cout << "motion_x264_preset=" << options.motion_x264_preset << "\n";
        std::cout << "motion_x264_threads=" << options.motion_x264_threads << "\n";
      }
      if (options.motion_shifts_file.empty())
        std::cout << "motion_skip_loop_filter=" << (options.motion_skip_loop_filter ? "true" : "false") << "\n";
      std::cout << "motion_nonzero_frames=" << frame_processor.motion_nonzero_frames() << "\n";
      std::cout << "geometry_bytes=" << geometry_bytes << "\n";
      std::cout << "codec_total_bytes=" << output.payload_bytes() + geometry_bytes << "\n";
    }
    std::cout << "q_index_min=" << (encoded_frames > 0 ? min_q_index_used : 0) << "\n";
    std::cout << "q_index_max=" << (encoded_frames > 0 ? max_q_index_used : 0) << "\n";
    std::cout << "q_index_mean="
              << (encoded_frames > 0 ? static_cast<double>(q_index_sum) / encoded_frames : 0.0)
              << "\n";
    std::cout << "average_bpp="
              << (encoded_frames > 0 ? static_cast<double>(output.file_bytes()) * 8.0 /
                                           (static_cast<double>(source_geometry.width) *
                                            source_geometry.height * encoded_frames)
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
