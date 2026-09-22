#include <acl/acl.h>
#include <mlvc/application/input/frame_input_queue.h>
#include <mlvc/application/pipeline/codec_frame_pipeline.h>
#include <mlvc/application/pipeline/ordered_future_window.h>
#include <mlvc/application/progress.h>
#include <mlvc/application/runtime/mlvc_codec_runtime.h>
#include <mlvc/application/stream/mlvc_entropy_decode.h>
#include <mlvc/application/stream/mlvc_stream.h>
#include <mlvc/codec/detail/frame/reference_state.h>
#include <mlvc/codec/detail/profile/codec_profile.h>
#include <mlvc/codec/detail/stage/constants.h>
#include <mlvc/codec/detail/stage/stage_runner.h>
#include <mlvc/codec/detail/stage/stage_runtime_state.h>
#include <mlvc/codec/detail/stage/stage_types.h>
#include <mlvc/codec/detail/tensor/tensor_utils.h>
#include <mlvc/codec/execution_profile.h>
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
#include <mlvc/runtime/fp16_yuv444_to_nv12_acl.h>
#include <mlvc/runtime/stage_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "mlvc/application/stream/mlvc_internal.h"
#include "mlvc/core/status.h"

namespace mlvc::codec {
namespace {

void CheckAclCall(aclError status, const char* operation) {
  if (status != ACL_ERROR_NONE) {
    throw mlvc::Error(std::string(operation) + " failed: ret=" + std::to_string(status));
  }
}

class AsyncRtspNv12Pipeline {
 public:
  AsyncRtspNv12Pipeline(const io::Nv12Layout& layout, aclrtStream decode_stream,
                        mlvc::io::RtspVideoPublisher* publisher, mlvc::Profiler* profiler)
      : layout_(layout), decode_stream_(decode_stream), publisher_(publisher), profiler_(profiler) {
    constexpr uint32_t kTimedSyncEvent = ACL_EVENT_SYNC | ACL_EVENT_TIME_LINE;
    CheckAclCall(aclrtCreateStream(&convert_stream_), "aclrtCreateStream RTSP NV12 convert");
    CheckAclCall(aclrtCreateStream(&copy_stream_), "aclrtCreateStream RTSP NV12 copy");
    const std::size_t bytes = io::Nv12BufferSize(layout_);
    for (Slot& slot : slots_) {
      slot.device.Allocate(bytes);
      slot.host.Allocate(bytes);
      CheckAclCall(aclrtCreateEventWithFlag(&slot.input_ready, kTimedSyncEvent),
                   "aclrtCreateEvent RTSP NV12 input ready");
      CheckAclCall(aclrtCreateEventWithFlag(&slot.conversion_start, kTimedSyncEvent),
                   "aclrtCreateEvent RTSP NV12 conversion start");
      CheckAclCall(aclrtCreateEventWithFlag(&slot.converted, kTimedSyncEvent),
                   "aclrtCreateEvent RTSP NV12 converted");
      CheckAclCall(aclrtCreateEventWithFlag(&slot.copy_start, kTimedSyncEvent),
                   "aclrtCreateEvent RTSP NV12 copy start");
      CheckAclCall(aclrtCreateEventWithFlag(&slot.copied, kTimedSyncEvent),
                   "aclrtCreateEvent RTSP NV12 copied");
    }
  }

  AsyncRtspNv12Pipeline(const AsyncRtspNv12Pipeline&) = delete;
  AsyncRtspNv12Pipeline& operator=(const AsyncRtspNv12Pipeline&) = delete;

  ~AsyncRtspNv12Pipeline() {
    try {
      Drain();
    } catch (...) {
    }
    if (convert_stream_ != nullptr) {
      (void)aclrtSynchronizeStream(convert_stream_);
    }
    if (copy_stream_ != nullptr) {
      (void)aclrtSynchronizeStream(copy_stream_);
    }
    for (Slot& slot : slots_) {
      if (slot.input_ready != nullptr) {
        (void)aclrtDestroyEvent(slot.input_ready);
      }
      if (slot.conversion_start != nullptr) {
        (void)aclrtDestroyEvent(slot.conversion_start);
      }
      if (slot.converted != nullptr) {
        (void)aclrtDestroyEvent(slot.converted);
      }
      if (slot.copy_start != nullptr) {
        (void)aclrtDestroyEvent(slot.copy_start);
      }
      if (slot.copied != nullptr) {
        (void)aclrtDestroyEvent(slot.copied);
      }
    }
    if (convert_stream_ != nullptr) {
      (void)aclrtDestroyStream(convert_stream_);
    }
    if (copy_stream_ != nullptr) {
      (void)aclrtDestroyStream(copy_stream_);
    }
  }

  void Enqueue(mlvc::TensorHandle* x_hat) {
    Check(x_hat != nullptr && x_hat->has_acl_buffer() && x_hat->acl_valid(),
          "RTSP ACL NV12 conversion requires a device-resident x_hat");
    if (pending_slots_.size() == slots_.size()) {
      FlushOne();
    }

    Slot& slot = slots_.at(next_slot_);
    slot.conversion_enqueue = std::chrono::steady_clock::now();
    x_hat->WaitReady(decode_stream_);
    CheckAclCall(aclrtRecordEvent(slot.input_ready, decode_stream_),
                 "aclrtRecordEvent RTSP NV12 input ready");
    CheckAclCall(aclrtStreamWaitEvent(convert_stream_, slot.input_ready),
                 "aclrtStreamWaitEvent RTSP NV12 input ready");
    Fp16Yuv444ToNv12Acl(x_hat->AclView().data(), x_hat->shape(), layout_, slot.device.data(),
                        convert_stream_, slot.conversion_start, slot.converted, profiler_);
    CheckAclCall(aclrtStreamWaitEvent(copy_stream_, slot.converted),
                 "aclrtStreamWaitEvent RTSP NV12 converted");
    slot.copy_enqueue = std::chrono::steady_clock::now();
    CheckAclCall(aclrtRecordEvent(slot.copy_start, copy_stream_),
                 "aclrtRecordEvent RTSP NV12 copy start");
    CheckAclCall(aclrtMemcpyAsync(slot.host.data(), slot.host.bytes(), slot.device.data(),
                                  slot.device.bytes(), ACL_MEMCPY_DEVICE_TO_HOST, copy_stream_),
                 "aclrtMemcpyAsync RTSP NV12 D2H");
    CheckAclCall(aclrtRecordEvent(slot.copied, copy_stream_), "aclrtRecordEvent RTSP NV12 copied");
    pending_slots_.push_back(next_slot_);
    next_slot_ = (next_slot_ + 1) % slots_.size();
  }

  void Drain() {
    while (!pending_slots_.empty()) {
      FlushOne();
    }
  }

 private:
  struct Slot {
    mlvc::AclBuffer device;
    mlvc::PinnedHostBuffer host;
    aclrtEvent input_ready = nullptr;
    aclrtEvent conversion_start = nullptr;
    aclrtEvent converted = nullptr;
    aclrtEvent copy_start = nullptr;
    aclrtEvent copied = nullptr;
    std::chrono::steady_clock::time_point conversion_enqueue;
    std::chrono::steady_clock::time_point copy_enqueue;
  };

  void FlushOne() {
    const std::size_t index = pending_slots_.front();
    Slot& slot = slots_.at(index);
    CheckAclCall(aclrtSynchronizeEvent(slot.copied), "aclrtSynchronizeEvent RTSP NV12 copied");
    CheckAclCall(aclrtSynchronizeEvent(slot.converted),
                 "aclrtSynchronizeEvent RTSP NV12 converted");
    float conversion_ms = 0.0f;
    float copy_ms = 0.0f;
    CheckAclCall(aclrtEventElapsedTime(&conversion_ms, slot.conversion_start, slot.converted),
                 "aclrtEventElapsedTime RTSP NV12 conversion");
    CheckAclCall(aclrtEventElapsedTime(&copy_ms, slot.copy_start, slot.copied),
                 "aclrtEventElapsedTime RTSP NV12 D2H");
    if (profiler_ != nullptr) {
      profiler_->AddEvent("acl_video.fp16_yuv444_to_nv12.device",
                          profiler_->StartMs(slot.conversion_enqueue), conversion_ms);
      profiler_->AddEvent("copy.rtsp.nv12_d2h.device", profiler_->StartMs(slot.copy_enqueue),
                          copy_ms);
    }
    const auto* begin = static_cast<const uint8_t*>(slot.host.data());
    publisher_->WriteNv12Frame(std::vector<uint8_t>(begin, begin + slot.host.bytes()));
    pending_slots_.pop_front();
  }

  io::Nv12Layout layout_;
  aclrtStream decode_stream_ = nullptr;
  aclrtStream convert_stream_ = nullptr;
  aclrtStream copy_stream_ = nullptr;
  mlvc::io::RtspVideoPublisher* publisher_ = nullptr;
  mlvc::Profiler* profiler_ = nullptr;
  std::array<Slot, 2> slots_;
  std::deque<std::size_t> pending_slots_;
  std::size_t next_slot_ = 0;
};

}  // namespace

int RunDecodeStream(const DecodeStreamOptions& options, DecodePipelineServices* services) {
  try {
    ScopedRuntimeState runtime_state_guard;
    // A decode run with no file or transport output does not need a host copy of
    // the reconstructed frame.  Keep the MLVC decoder outputs device-resident
    // so reference features can flow directly into the next frame.  Entropy
    // outputs remain materialized only where the CPU entropy worker consumes
    // them.  This avoids a full-resolution D2H copy and host synchronization on
    // every 1080P frame.
    const bool use_device_resident_outputs =
        options.output_format == "none" && options.output_transport_port == 0 &&
        (options.output_transport_mode == "none" || options.output_transport_mode == "rtsp");
    std::unique_ptr<mlvc::app::MlvcCodecRuntime> owned_codec_runtime;
    mlvc::app::MlvcCodecRuntime* codec_runtime =
        services == nullptr ? nullptr : services->codec_runtime;
    if (codec_runtime == nullptr) {
      owned_codec_runtime = std::make_unique<mlvc::app::MlvcCodecRuntime>(
          options.manifest_path, options.device,
          use_device_resident_outputs
              ? mlvc::codec::StageOutputBindingMode::kAclDeviceWithCpuMirror
              : mlvc::codec::StageOutputBindingMode::kCpu);
      codec_runtime = owned_codec_runtime.get();
    }
    mlvc::StageRuntime& runtime = codec_runtime->runtime();
    mlvc::StageModelSet& models = codec_runtime->models();
    const mlvc::RuntimeSidecar& sidecar = codec_runtime->sidecar();
    const std::filesystem::path model_directory = models.manifest().directory();
    Check(HasMlvcModels(models.manifest()), "manifest does not contain MLVCEncoder / MLVCDecoder");
    const mlvc::ModelRecord& decoder_record = models.manifest().GetModel("MLVCDecoder");
    mlvc::codec::StageOutputWorkspace& stage_output_workspace =
        codec_runtime->stage_output_workspace();
    ConfigureRuntimeState(&stage_output_workspace, runtime, false);

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

    const bool udp_input = options.input_transport_port > 0;
    const bool rtp_input = udp_input && options.input_transport_mode == "rtp";
    std::optional<io::MlvcBitstreamReader> bitstream_reader;
    std::optional<io::UdpMlvcReceiver> udp_receiver;
    std::optional<io::RtpMlvcReceiver> rtp_receiver;
    io::MlvcBitstreamHeader header;
    if (udp_input) {
      if (rtp_input) {
        rtp_receiver.emplace(static_cast<uint16_t>(options.input_transport_port));
        header = rtp_receiver->ReceiveHeader();
      } else {
        udp_receiver.emplace(static_cast<uint16_t>(options.input_transport_port));
        header = udp_receiver->ReceiveHeader();
      }
    } else {
      Check(!options.input_bitstream_path.empty(), "decode requires input bitstream or udp_port");
      bitstream_reader.emplace(options.input_bitstream_path);
      header = bitstream_reader->header();
    }
    const io::ForcedLtrFrames forced_ltr = io::ResolveForcedLtrFrames(
        header, options.forced_ltr_recovery_frame, options.forced_ltr_reference_frame);
    const int forced_ltr_recovery_frame = forced_ltr.recovery_frame;
    const int forced_ltr_reference_frame = forced_ltr.reference_frame;
    const bool write_output = options.output_format != "none";
    const std::string writer_format =
        options.output_format.empty() ? io::GuessDecodeOutputFormat(options.output_video_path, "")
                                      : options.output_format;
    const double writer_fps = header.fps > 0.0 ? header.fps : options.fps;
    std::optional<io::DecodedVideoWriter> video_writer;
    if (write_output) {
      video_writer.emplace(options.output_video_path, writer_format, writer_fps, header.width,
                           header.height, options.crf, options.bitrate, options.preset);
    }
    const bool device_only_outputs = use_device_resident_outputs && !write_output;
    ScopedAsyncDecodeDeviceOnlyMirrorSkip device_output_scope(device_only_outputs);
    ScopedDecodeVideoOutputMirror video_output_scope(options.output_transport_mode == "rtsp");
    Check(options.output_transport_port == 0 || !options.output_transport_host.empty(),
          "output_transport_host is required when output_transport_port is set");
    std::optional<io::RawYuvUdpSender> raw_forwarder;
    std::optional<io::RtspVideoPublisher> rtsp_publisher;
    if (options.output_transport_port > 0) {
      Check(options.output_transport_mode == "raw_fp16_yuv444",
            "only raw_fp16_yuv444 output transport is supported; JPEG forwarding was removed");
      raw_forwarder.emplace(options.output_transport_host,
                            static_cast<uint16_t>(options.output_transport_port));
    }
    if (options.output_transport_mode == "rtsp") {
      rtsp_publisher.emplace(options.output_transport_rtsp_url, writer_fps, header.width,
                             header.height, options.output_transport_rtsp_preset,
                             options.output_transport_rtsp_crf,
                             static_cast<std::size_t>(options.output_transport_queue_capacity),
                             options.output_transport_rtsp_transport);
    }
    const io::Nv12Layout rtsp_nv12_layout{header.width, header.height, header.width, header.height};
    const bool rtsp_acl_nv12 = rtsp_publisher.has_value() && Fp16Yuv444ToNv12AclAvailable();
    std::optional<AsyncRtspNv12Pipeline> rtsp_nv12_pipeline;
    if (rtsp_acl_nv12) {
      rtsp_nv12_pipeline.emplace(rtsp_nv12_layout, runtime.stream(), &*rtsp_publisher, &profiler);
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
      entropy_decoders.emplace_back(std::make_unique<MlvcOfficialEntropyDecoder>(model_directory));
    }
    mlvc::app::OrderedFutureWindow<DecodedEntropyFrame> entropy_window(
        options.pipeline.entropy_workers);
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
        if (rtp_input) {
          mlvc::ScopedCpuTimer timer(&profiler, "rtp.receive_queue");
          received = rtp_receiver->ReceiveFrame(&next_frame_index, &next_frame_type, &next_q_index,
                                                &next_payload);
        } else if (udp_input) {
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
        if (udp_input && !rtp_input) {
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
    // The decoder does not allocate encoder input tensors, so its frame buffer
    // budget applies to the packet pipeline itself.  Bound the queue by both
    // knobs so pipeline.frame_buffer_slots is effective instead of silently
    // being ignored.
    const std::size_t bitstream_queue_capacity =
        std::min(options.pipeline.queue_capacity, options.pipeline.frame_buffer_slots);
    mlvc::StreamingPipeline bitstream_pipeline(options.pipeline.stream_workers,
                                               bitstream_queue_capacity);
    bitstream_pipeline.SetProcessor([](const std::shared_ptr<mlvc::DataObject>& input) {
      mlvc::Check(input != nullptr, "bitstream pipeline received an empty input");
      return input;
    });
    bitstream_pipeline.Start();
    mlvc::app::BitstreamProducer bitstream_producer(bitstream_pipeline, bitstream_pipeline_running,
                                                    read_next_packet);
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
            frame_index == forced_ltr_recovery_frame && forced_ltr_reference_frame >= 0
                ? forced_ltr_reference_frame
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
      RunOutput decoder_output =
          RunStage(&models, "MLVCDecoder",
                   {TensorInput("z_raw", decoded.z_raw), TensorInput("y_raw_0", decoded.y_raw_0),
                    TensorInput("y_raw_1", decoded.y_raw_1), ref_feature_input,
                    TensorInput("q_index_shifted", q_index_shifted_tensor)},
                   &profiler);
      if (raw_forwarder.has_value()) {
        mlvc::ScopedCpuTimer timer(&profiler, "udp.raw_forward_enqueue");
        raw_forwarder->SendFrame(decoder_output.At("x_hat"), frame_index);
      }
      if (rtsp_publisher.has_value()) {
        mlvc::ScopedCpuTimer timer(&profiler, "rtsp.output_enqueue");
        if (rtsp_acl_nv12) {
          rtsp_nv12_pipeline->Enqueue(decoder_output.Handle("x_hat"));
        } else {
          rtsp_publisher->WriteTensorFrame(decoder_output.At("x_hat"));
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
                                                       bitstream_pipeline_running, consume_packet);
    bitstream_consumer.Start();
    bitstream_producer.Join();
    bitstream_producer.RethrowIfFailed();
    bitstream_consumer.Join();
    bitstream_consumer.RethrowIfFailed();
    while (!entropy_window.empty()) {
      run_decoded_frame(entropy_window.PopFront());
    }
    if (rtsp_nv12_pipeline.has_value()) {
      rtsp_nv12_pipeline->Drain();
    }
    bitstream_pipeline.Stop();
    bitstream_pipeline.RethrowIfFailed();
    if (raw_forwarder.has_value()) {
      raw_forwarder->Close();
    }
    if (rtsp_publisher.has_value()) {
      rtsp_publisher->Close();
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
    std::cout << "stage_output_binding="
              << (use_device_resident_outputs ? "acl-device-with-cpu-mirror" : "cpu") << "\n";
    std::cout << "device_resident_outputs="
              << (use_device_resident_outputs ? "true" : "false") << "\n";
    std::cout << "bitstream_bytes=" << bitstream_bytes << "\n";
    std::cout << "drop_frame_index=" << options.drop_frame_index << "\n";
    std::cout << "forced_ltr_reference_frame=" << forced_ltr_reference_frame << "\n";
    std::cout << "forced_ltr_recovery_frame=" << forced_ltr_recovery_frame << "\n";
    std::cout << "output_video=" << (write_output ? options.output_video_path.string() : "none")
              << "\n";
    std::cout << "output_transport_target="
              << (raw_forwarder.has_value()
                      ? options.output_transport_host + ":" +
                            std::to_string(options.output_transport_port)
                      : (rtsp_publisher.has_value() ? options.output_transport_rtsp_url : "none"))
              << "\n";
    std::cout << "output_transport_mode="
              << (raw_forwarder.has_value() ? "raw_fp16_yuv444"
                                            : (rtsp_publisher.has_value() ? "rtsp" : "none"))
              << "\n";
    std::cout << "forward_dropped_frames="
              << (raw_forwarder.has_value()
                      ? raw_forwarder->dropped_frames()
                      : (rtsp_publisher.has_value() ? rtsp_publisher->dropped_frames() : 0))
              << "\n";
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
