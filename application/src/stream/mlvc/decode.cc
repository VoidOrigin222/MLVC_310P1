#include <mlvc/application/input/frame_input_queue.h>
#include <mlvc/application/pipeline/ordered_future_window.h>
#include <mlvc/application/pipeline/codec_frame_pipeline.h>
#include <mlvc/application/progress.h>
#include <mlvc/application/runtime/mlvc_codec_runtime.h>
#include <mlvc/application/stream/mlvc_entropy_decode.h>
#include <mlvc/codec/execution_profile.h>
#include <mlvc/application/stream/mlvc_stream.h>
#include <mlvc/codec/mlvc_entropy.h>
#include <mlvc/codec/mlvc_rate_control.h>
#include <mlvc/codec/tensor_utils.h>
#include <mlvc/entropy/entropy_codec.h>
#include <mlvc/entropy/mlvc_official_entropy.h>
#include <mlvc/entropy/sidecar.h>
#include <mlvc/framework/codec_graph_executor.h>
#include <mlvc/framework/entropy_worker.h>
#include <mlvc/framework/profile_range.h>
#include <mlvc/framework/profiler.h>
#include <mlvc/io/frame_source.h>
#include <mlvc/io/mlvc_bitstream.h>
#include <mlvc/io/udp_frame_transport.h>
#include <mlvc/io/video_io.h>
#include <mlvc/runtime/stage_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <mlvc/codec/detail/profile/codec_profile.h>
#include <mlvc/codec/detail/stage/constants.h>
#include <mlvc/codec/detail/frame/reference_state.h>
#include <mlvc/codec/detail/stage/stage_runner.h>
#include <mlvc/codec/detail/stage/stage_runtime_state.h>
#include <mlvc/codec/detail/stage/stage_types.h>
#include <mlvc/codec/detail/tensor/tensor_utils.h>
#include "mlvc/core/status.h"

#include "mlvc/application/stream/mlvc_internal.h"

namespace mlvc::codec {
int RunDecodeStream(const DecodeStreamOptions& options, DecodePipelineServices* services) {
  try {
    ScopedRuntimeState runtime_state_guard;
    const bool device_dvpp_mode =
        options.forward_port > 0 && options.forward_mode == "dvpp_jpeg_device_async";
    std::unique_ptr<mlvc::app::MlvcCodecRuntime> owned_codec_runtime;
    mlvc::app::MlvcCodecRuntime* codec_runtime =
        services == nullptr ? nullptr : services->codec_runtime;
    if (codec_runtime == nullptr) {
      owned_codec_runtime = std::make_unique<mlvc::app::MlvcCodecRuntime>(
          options.manifest_path, options.device,
          device_dvpp_mode ? mlvc::codec::StageOutputBindingMode::kAclMirror
                           : mlvc::codec::StageOutputBindingMode::kCpu);
      codec_runtime = owned_codec_runtime.get();
    }
    mlvc::StageRuntime& runtime = codec_runtime->runtime();
    mlvc::StageModelSet& models = codec_runtime->models();
    const mlvc::RuntimeSidecar& sidecar = codec_runtime->sidecar();
    const std::filesystem::path model_directory = models.manifest().directory();
    Check(HasMlvcModels(models.manifest()),
          "manifest does not contain MLVCEncoder / MLVCDecoder");
    const mlvc::ModelRecord& decoder_record = models.manifest().GetModel("MLVCDecoder");
    mlvc::codec::StageOutputWorkspace& stage_output_workspace =
        codec_runtime->stage_output_workspace();
    ConfigureRuntimeState(&stage_output_workspace, runtime, false);
    Check(!device_dvpp_mode ||
              stage_output_workspace.binding_mode() ==
                  mlvc::codec::StageOutputBindingMode::kAclMirror,
          "dvpp_jpeg_device_async requires an ACL-mirror codec runtime");

    std::optional<mlvc::Profiler> owned_profiler;
    std::optional<mlvc::CodecGraphExecutor> owned_graph_executor;
    std::optional<mlvc::EntropyWorker> owned_entropy_worker;
    DecodePipelineServices owned_services;
    if (services == nullptr) {
      owned_profiler.emplace();
      owned_graph_executor.emplace(options.pipeline.graph_packet_capacity);
      owned_entropy_worker.emplace(options.pipeline.entropy_workers);
      owned_services = DecodePipelineServices{&*owned_profiler, &*owned_graph_executor,
                                              &*owned_entropy_worker, codec_runtime};
      services = &owned_services;
    }
    Check(services->profiler != nullptr && services->graph_executor != nullptr &&
              services->entropy_worker != nullptr,
          "decode pipeline services must provide profiler, graph executor, and entropy worker");
    mlvc::Profiler& profiler = *services->profiler;
    mlvc::CodecGraphExecutor& graph_executor = *services->graph_executor;
    mlvc::EntropyWorker& entropy_worker = *services->entropy_worker;
    profiler.ReserveEvents(256);
    g_codec_graph_executor = &graph_executor;
    g_codec_graph_executor->RecordTemplate(&profiler);

    const bool udp_input = options.udp_port > 0;
    std::optional<io::MlvcBitstreamReader> bitstream_reader;
    std::optional<io::UdpMlvcReceiver> udp_receiver;
    io::MlvcBitstreamHeader header;
    if (udp_input) {
      udp_receiver.emplace(static_cast<uint16_t>(options.udp_port));
      header = udp_receiver->ReceiveHeader();
    } else {
      Check(!options.input_bitstream_path.empty(), "decode requires input bitstream or udp_port");
      bitstream_reader.emplace(options.input_bitstream_path);
      header = bitstream_reader->header();
    }
    const bool write_output = options.output_format != "none";
    const std::string writer_format =
        options.output_format.empty()
            ? io::GuessDecodeOutputFormat(options.output_video_path, "")
            : options.output_format;
    const double writer_fps = header.fps > 0.0 ? header.fps : options.fps;
    std::optional<io::DecodedVideoWriter> video_writer;
    if (write_output) {
      video_writer.emplace(options.output_video_path, writer_format, writer_fps, header.width,
                           header.height, options.crf, options.bitrate, options.preset);
    }
    const bool device_only_outputs = device_dvpp_mode && !write_output;
    ScopedAsyncDecodeDeviceOnlyMirrorSkip device_output_scope(device_only_outputs);
    Check(options.forward_port == 0 || !options.forward_host.empty(),
          "forward_host is required when forward_port is set");
    std::optional<io::VideoTransUdpSender> forwarder;
    std::optional<io::AsyncVideoTransUdpSender> async_forwarder;
    std::optional<io::AsyncDvppVideoTransUdpSender> dvpp_forwarder;
    std::optional<io::AsyncDeviceDvppVideoTransUdpSender> device_dvpp_forwarder;
    std::optional<io::RawYuvUdpSender> raw_forwarder;
    if (options.forward_port > 0) {
      if (options.forward_mode == "raw_fp16_yuv444") {
        raw_forwarder.emplace(options.forward_host, static_cast<uint16_t>(options.forward_port));
      } else if (options.forward_mode == "jpeg_async") {
        async_forwarder.emplace(options.forward_host, static_cast<uint16_t>(options.forward_port), header.width,
                                header.height);
      } else if (options.forward_mode == "dvpp_jpeg_async") {
        dvpp_forwarder.emplace(
            options.forward_host, static_cast<uint16_t>(options.forward_port), header.width,
            header.height, static_cast<std::size_t>(options.forward_queue_capacity),
            static_cast<uint32_t>(options.forward_jpeg_quality), runtime.context());
      } else if (options.forward_mode == "dvpp_jpeg_device_async") {
        device_dvpp_forwarder.emplace(
            options.forward_host, static_cast<uint16_t>(options.forward_port), header.width,
            header.height, static_cast<std::size_t>(options.forward_queue_capacity),
            static_cast<uint32_t>(options.forward_jpeg_quality), runtime.context());
      } else {
        forwarder.emplace(options.forward_host, static_cast<uint16_t>(options.forward_port));
      }
    }
    ReferenceState state;
    TensorData zero_feature = MakeFp16Tensor(decoder_record.inputs.at(3).shape, 0.0f);
    TensorData z_raw_tensor;
    TensorData y_raw_0_tensor;
    TensorData y_raw_1_tensor;
    TensorData ltr_feature = MakeFp16Tensor(decoder_record.inputs.at(3).shape, 0.0f);
    bool has_ltr_feature = false;
    std::map<int, TensorData> ltr_features;
    std::vector<std::unique_ptr<MlvcOfficialEntropyDecoder>> entropy_decoders;
    entropy_decoders.reserve(options.pipeline.entropy_workers);
    for (std::size_t i = 0; i < options.pipeline.entropy_workers; ++i) {
      entropy_decoders.emplace_back(
          std::make_unique<MlvcOfficialEntropyDecoder>(model_directory));
    }
    mlvc::app::OrderedFutureWindow<DecodedEntropyFrame> entropy_window(
        options.pipeline.entropy_workers);
    bool device_conversion_pending = false;
    const auto decode_start = std::chrono::steady_clock::now();
    int decoded_frames = 0;
    int input_frames_seen = 0;
    int expected_udp_frame_index = 0;
    uint64_t bitstream_bytes = 0;
    auto submit_entropy = [&](std::size_t slot, int frame_index, MlvcFrameType frame_type,
                              int q_index,
                              std::vector<uint8_t> payload) -> std::future<DecodedEntropyFrame> {
      return entropy_worker.SubmitValue<DecodedEntropyFrame>(
          [&, slot, frame_index, frame_type, q_index, payload = std::move(payload)]() mutable {
            return DecodeMlvcEntropyFrame(entropy_decoders.at(slot).get(), decoder_record,
                                          frame_index, frame_type, q_index, std::move(payload),
                                          &profiler);
          });
    };

    auto read_next_packet = [&]() -> std::shared_ptr<mlvc::app::BitstreamPacket> {
      while (options.frame_num <= 0 || input_frames_seen < options.frame_num) {
        int next_frame_index = 0;
        MlvcFrameType next_frame_type = MlvcFrameType::kPFrame;
        int next_q_index = 0;
        std::vector<uint8_t> next_payload;
        bool received = false;
        if (udp_input) {
          mlvc::ScopedCpuTimer timer(&profiler, "udp.receive_queue");
          received = udp_receiver->ReceiveFrame(&next_frame_index, &next_frame_type, &next_q_index,
                                                &next_payload);
        } else {
          received = bitstream_reader->ReadFrame(&next_frame_index, &next_frame_type, &next_q_index,
                                                 &next_payload);
        }
        if (!received) {
          return nullptr;
        }
        ++input_frames_seen;
        if (udp_input) {
          Check(next_frame_index == expected_udp_frame_index,
                "UDP MLVC frame loss or reordering: expected frame " +
                    std::to_string(expected_udp_frame_index) + ", got " +
                    std::to_string(next_frame_index));
          ++expected_udp_frame_index;
        }
        if (next_frame_index == options.drop_frame_index) {
          continue;
        }
        bitstream_bytes += next_payload.size();
        auto packet = std::make_shared<mlvc::app::BitstreamPacket>();
        packet->frame_index = next_frame_index;
        packet->frame_type = next_frame_type;
        packet->q_index = next_q_index;
        packet->payload = std::move(next_payload);
        return packet;
      }
      return nullptr;
    };

    std::atomic<bool> bitstream_pipeline_running{true};
    mlvc::StreamingPipeline bitstream_pipeline(options.pipeline.stream_workers,
                                               options.pipeline.queue_capacity);
    bitstream_pipeline.SetProcessor(
        [](const std::shared_ptr<mlvc::DataObject>& input) {
          mlvc::Check(input != nullptr, "bitstream pipeline received an empty input");
          return input;
        });
    bitstream_pipeline.Start();
    mlvc::app::BitstreamProducer bitstream_producer(
        bitstream_pipeline, bitstream_pipeline_running, read_next_packet);
    bitstream_producer.Start();

    auto run_decoded_frame = [&](const DecodedEntropyFrame& decoded) {
      runtime.MakeCurrent();
      const int frame_index = decoded.frame_index;
      const MlvcFrameType frame_type = decoded.frame_type;
      const int q_index = decoded.q_index;
      const bool is_i_frame = frame_type == MlvcFrameType::kIFrame;
      const bool use_ltr_recovery = frame_type == MlvcFrameType::kLtrRecovery;
      const int gop_cycle_index = header.gop > 0 ? frame_index % header.gop : frame_index;
      if (ShouldResetReferenceFeature(frame_index, header.gop, header.reset_interval)) {
        state.ResetFeature();
      }
      if (is_i_frame) {
        has_ltr_feature = false;
        FillFp16Tensor(0.0f, &ltr_feature);
        ltr_features.clear();
      }
      if (use_ltr_recovery) {
        Check(has_ltr_feature, "LTR recovery frame has no cached LTR feature");
        const int reference_frame =
            frame_index == options.forced_ltr_recovery_frame &&
                    options.forced_ltr_reference_frame >= 0
                ? options.forced_ltr_reference_frame
                : ltr_features.rbegin()->first;
        const auto ltr_it = ltr_features.find(reference_frame);
        Check(ltr_it != ltr_features.end(),
              "requested LTR reference frame is not cached: " + std::to_string(reference_frame));
        state.feature = CloneTensor(ltr_it->second);
        state.feature_handle.reset();
      }

      const int frame_adaptation_index = kIndexMap[(frame_index + 1) % 8];
      const int q_index_shifted = sidecar.ShiftedQp(q_index, frame_adaptation_index);
      TensorData q_index_shifted_tensor = MakeInt32ScalarTensor(q_index_shifted);
      const StageInput ref_feature_input = BuildReferenceFeatureInput(state, zero_feature);
      if (device_conversion_pending) {
        CheckAcl(aclrtSynchronizeStream(runtime.stream()),
                 "synchronize device NV12 conversion before decoder output reuse");
        device_conversion_pending = false;
      }
      RunOutput decoder_output =
          RunStage(&models, "MLVCDecoder",
                   {TensorInput("z_raw", decoded.z_raw), TensorInput("y_raw_0", decoded.y_raw_0),
                    TensorInput("y_raw_1", decoded.y_raw_1), ref_feature_input,
                    TensorInput("q_index_shifted", q_index_shifted_tensor)},
                   &profiler);
      if (raw_forwarder.has_value()) {
        mlvc::ScopedCpuTimer timer(&profiler, "udp.raw_forward_enqueue");
        raw_forwarder->SendFrame(decoder_output.At("x_hat"), frame_index);
      } else if (device_dvpp_forwarder.has_value()) {
        mlvc::ScopedCpuTimer timer(&profiler, "udp.dvpp_jpeg_device_async_enqueue");
        const mlvc::TensorHandle* x_hat = decoder_output.Handle("x_hat");
        Check(x_hat != nullptr && x_hat->has_acl_buffer() && x_hat->acl_valid(),
              "device DVPP forward requires an ACL-resident x_hat");
        device_dvpp_forwarder->SendFrame(
            x_hat->AclView().data(), x_hat->shape(), runtime.stream(), frame_index,
            &profiler);
        device_conversion_pending = true;
      } else if (dvpp_forwarder.has_value()) {
        mlvc::ScopedCpuTimer timer(&profiler, "udp.dvpp_jpeg_async_enqueue");
        dvpp_forwarder->SendFrame(decoder_output.At("x_hat"), frame_index);
      } else if (async_forwarder.has_value()) {
        mlvc::ScopedCpuTimer timer(&profiler, "udp.jpeg_async_enqueue");
        async_forwarder->SendFrame(decoder_output.At("x_hat"), frame_index);
      } else if (forwarder.has_value()) {
        cv::Mat frame;
        {
          mlvc::ScopedCpuTimer timer(&profiler, "video.forward_convert");
          frame = io::ConvertTensorToBgr(decoder_output.At("x_hat"), header.width, header.height);
        }
        {
          mlvc::ScopedCpuTimer timer(&profiler, "udp.forward_enqueue");
          forwarder->SendFrame(frame);
        }
      }
      if (video_writer.has_value()) {
        video_writer->WriteTensorFrame(decoder_output.At("x_hat"), decoded_frames);
      }
      UpdateReferenceFeature(decoder_output, &state, &profiler);
      const bool mark_as_ltr =
          header.ltr_period > 0 &&
          (gop_cycle_index == header.ltr_start_idx ||
           (gop_cycle_index > header.ltr_start_idx && gop_cycle_index % header.ltr_period == 0));
      if (mark_as_ltr) {
        if (device_only_outputs) {
          Check(state.feature_handle.has_value(),
                "device LTR capture requires an ACL feature handle");
          state.feature_handle->MaterializeToCpu("ltr_capture", runtime.stream());
          const mlvc::TensorView cpu_feature = state.feature_handle->CpuView();
          ltr_feature.shape = state.feature_handle->shape();
          ltr_feature.dtype = state.feature_handle->dtype();
          ltr_feature.bytes.resize(state.feature_handle->bytes());
          std::memcpy(ltr_feature.bytes.data(), cpu_feature.data(), ltr_feature.bytes.size());
        } else {
          CloneTensorInto(decoder_output.At("feature"), &ltr_feature);
        }
        ltr_features[frame_index] = CloneTensor(ltr_feature);
        has_ltr_feature = true;
      }
      ++decoded_frames;
    };

    auto consume_packet = [&](const std::shared_ptr<mlvc::DataObject>& data) {
      auto packet = std::dynamic_pointer_cast<mlvc::app::BitstreamPacket>(data);
      Check(packet != nullptr, "bitstream pipeline returned an unexpected data object");
      std::optional<DecodedEntropyFrame> ready;
      if (entropy_window.full()) {
        ready = entropy_window.PopFront();
      }
      entropy_window.SubmitNext([&](std::size_t slot) {
        return submit_entropy(slot, packet->frame_index, packet->frame_type, packet->q_index,
                              std::move(packet->payload));
      });
      if (ready.has_value()) {
        run_decoded_frame(*ready);
      }
    };
    mlvc::app::CallbackDataConsumer bitstream_consumer(bitstream_pipeline,
                                                        bitstream_pipeline_running,
                                                        consume_packet);
    bitstream_consumer.Start();
    bitstream_producer.Join();
    bitstream_producer.RethrowIfFailed();
    bitstream_consumer.Join();
    bitstream_consumer.RethrowIfFailed();
    while (!entropy_window.empty()) {
      run_decoded_frame(entropy_window.PopFront());
    }
    bitstream_pipeline.Stop();
    if (dvpp_forwarder.has_value()) {
      dvpp_forwarder->Close();
    }
    if (device_dvpp_forwarder.has_value()) {
      device_dvpp_forwarder->Close();
    }
    if (video_writer.has_value()) {
      video_writer->Close();
    }
    const auto decode_end = std::chrono::steady_clock::now();
    if (!options.profile_output_path.empty()) {
      profiler.WriteChromeTrace(options.profile_output_path);
    }
    const double seconds = std::chrono::duration<double>(decode_end - decode_start).count();
    std::cout << "mode=mlvc_decode\n";
    std::cout << "frames=" << decoded_frames << "\n";
    std::cout << "decode_fps="
              << (seconds > 0.0 ? static_cast<double>(decoded_frames) / seconds : 0.0) << "\n";
    std::cout << "entropy_workers=" << options.pipeline.entropy_workers << "\n";
    std::cout << "entropy_max_in_flight=" << entropy_window.max_observed_size() << "\n";
    std::cout << "bitstream_bytes=" << bitstream_bytes << "\n";
    std::cout << "drop_frame_index=" << options.drop_frame_index << "\n";
    std::cout << "forced_ltr_reference_frame=" << options.forced_ltr_reference_frame << "\n";
    std::cout << "forced_ltr_recovery_frame=" << options.forced_ltr_recovery_frame << "\n";
    std::cout << "output_video="
              << (write_output ? options.output_video_path.string() : "none") << "\n";
    std::cout << "output_transport_target="
              << ((forwarder.has_value() || async_forwarder.has_value() ||
                   dvpp_forwarder.has_value() || device_dvpp_forwarder.has_value() ||
                   raw_forwarder.has_value())
                      ? options.forward_host + ":" + std::to_string(options.forward_port)
                      : "none")
              << "\n";
    std::cout << "output_transport_mode="
              << ((raw_forwarder.has_value())
                      ? "raw_fp16_yuv444"
                      : (device_dvpp_forwarder.has_value()
                             ? "dvpp_jpeg_device_async"
                             : (dvpp_forwarder.has_value()
                             ? "dvpp_jpeg_async"
                             : (async_forwarder.has_value()
                                    ? "jpeg_async"
                                    : (forwarder.has_value() ? "jpeg" : "none")))))
              << "\n";
    std::cout << "forward_dropped_frames="
              << (raw_forwarder.has_value()
                      ? raw_forwarder->dropped_frames()
                      : (device_dvpp_forwarder.has_value()
                             ? device_dvpp_forwarder->dropped_frames()
                             : (dvpp_forwarder.has_value()
                             ? dvpp_forwarder->dropped_frames()
                             : (async_forwarder.has_value() ? async_forwarder->dropped_frames()
                                                            : 0))))
              << "\n";
    if (dvpp_forwarder.has_value() || device_dvpp_forwarder.has_value()) {
      const io::DvppForwardStats forward_stats =
          device_dvpp_forwarder.has_value() ? device_dvpp_forwarder->stats()
                                            : dvpp_forwarder->stats();
      const double encoded = static_cast<double>(forward_stats.encoded);
      const double sent = static_cast<double>(forward_stats.sent);
      std::cout << "forward_enqueued_frames=" << forward_stats.enqueued << "\n";
      std::cout << "forward_encoded_frames=" << forward_stats.encoded << "\n";
      std::cout << "forward_sent_frames=" << forward_stats.sent << "\n";
      std::cout << "forward_failed_frames=" << forward_stats.failed << "\n";
      std::cout << "forward_convert_avg_ms="
                << (encoded > 0.0 ? forward_stats.conversion_ms / encoded : 0.0) << "\n";
      std::cout << "forward_h2d_avg_ms="
                << (encoded > 0.0 ? forward_stats.h2d_ms / encoded : 0.0) << "\n";
      std::cout << "forward_dvpp_encode_avg_ms="
                << (encoded > 0.0 ? forward_stats.encode_ms / encoded : 0.0) << "\n";
      std::cout << "forward_d2h_avg_ms="
                << (encoded > 0.0 ? forward_stats.d2h_ms / encoded : 0.0) << "\n";
      std::cout << "forward_udp_avg_ms="
                << (sent > 0.0 ? forward_stats.udp_ms / sent : 0.0) << "\n";
    }
    std::cout << "output_video_frames="
              << (video_writer.has_value() ? video_writer->frame_count() : decoded_frames) << "\n";
    std::cout << "decode=ok\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "MLVC decode failed: " << error.what() << "\n";
    return 1;
  }
}


}  // namespace mlvc::codec
