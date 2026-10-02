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
#include <mlvc/codec/translation_warp.h>
#include <mlvc/entropy/entropy_codec.h>
#include <mlvc/entropy/mlvc_official_entropy.h>
#include <mlvc/entropy/sidecar.h>
#include <mlvc/framework/codec_graph_executor.h>
#include <mlvc/framework/entropy_worker.h>
#include <mlvc/framework/profile_range.h>
#include <mlvc/framework/profiler.h>
#include <mlvc/io/frame_source.h>
#include <mlvc/io/dvpp_h264_encoder.h>
#include <mlvc/io/mlvc_bitstream.h>
#include <mlvc/io/udp_frame_transport.h>
#include <mlvc/io/video_io.h>
#include <mlvc/runtime/fp16_yuv444_to_nv12_acl.h>
#include <mlvc/runtime/stage_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
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
                        mlvc::io::RtspVideoPublisher* publisher, aclrtContext context,
                        std::optional<io::DvppH264EncoderConfig> h264_encoder_config,
                        mlvc::Profiler* profiler)
      : layout_(layout),
        decode_stream_(decode_stream),
        publisher_(publisher),
        context_(context),
        h264_encoder_config_(std::move(h264_encoder_config)),
        profiler_(profiler) {
    constexpr uint32_t kTimedSyncEvent = ACL_EVENT_SYNC | ACL_EVENT_TIME_LINE;
    CheckAclCall(aclrtCreateStream(&convert_stream_), "aclrtCreateStream RTSP NV12 convert");
    if (!h264_encoder_config_.has_value()) {
      CheckAclCall(aclrtCreateStream(&copy_stream_), "aclrtCreateStream RTSP NV12 copy");
    }
    const std::size_t bytes = io::Nv12BufferSize(layout_);
    for (Slot& slot : slots_) {
      slot.device.Allocate(bytes);
      if (!h264_encoder_config_.has_value()) {
        slot.host.Allocate(bytes);
      }
      CheckAclCall(aclrtCreateEventWithFlag(&slot.input_ready, kTimedSyncEvent),
                   "aclrtCreateEvent RTSP NV12 input ready");
      CheckAclCall(aclrtCreateEventWithFlag(&slot.conversion_start, kTimedSyncEvent),
                   "aclrtCreateEvent RTSP NV12 conversion start");
      CheckAclCall(aclrtCreateEventWithFlag(&slot.converted, kTimedSyncEvent),
                   "aclrtCreateEvent RTSP NV12 converted");
      if (!h264_encoder_config_.has_value()) {
        CheckAclCall(aclrtCreateEventWithFlag(&slot.copy_start, kTimedSyncEvent),
                     "aclrtCreateEvent RTSP NV12 copy start");
        CheckAclCall(aclrtCreateEventWithFlag(&slot.copied, kTimedSyncEvent),
                     "aclrtCreateEvent RTSP NV12 copied");
      }
    }
    if (h264_encoder_config_.has_value()) {
      encoder_worker_ = std::thread(&AsyncRtspNv12Pipeline::EncoderWorker, this);
    }
  }

  AsyncRtspNv12Pipeline(const AsyncRtspNv12Pipeline&) = delete;
  AsyncRtspNv12Pipeline& operator=(const AsyncRtspNv12Pipeline&) = delete;

  ~AsyncRtspNv12Pipeline() {
    try {
      Drain();
    } catch (...) {
    }
    if (encoder_worker_.joinable()) {
      {
        std::lock_guard<std::mutex> lock(worker_mutex_);
        worker_stopping_ = true;
      }
      worker_condition_.notify_all();
      encoder_worker_.join();
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
    const std::size_t index = next_slot_;
    const bool async_encode = h264_encoder_config_.has_value();
    if (async_encode) {
      ReserveSlot(index);
    } else if (pending_slots_.size() == slots_.size()) {
      const std::size_t completed_index = pending_slots_.front();
      FlushOne(completed_index);
      pending_slots_.pop_front();
    }

    Slot& slot = slots_.at(index);
    try {
      slot.conversion_enqueue = std::chrono::steady_clock::now();
      x_hat->WaitReady(decode_stream_);
      const void* conversion_input = x_hat->AclView().data();
      mlvc::TensorShape conversion_shape = x_hat->shape();
      if (async_encode) {
        // MLVCDecoder reuses one ACL output buffer for x_hat.  Move that
        // buffer into the reserved RTSP slot and immediately give the
        // decoder a fresh output buffer.  This avoids a full-frame D2D copy
        // while keeping the converted input alive until VENC is finished.
        const mlvc::DataType conversion_dtype = x_hat->dtype();
        if (!slot.fp16.has_acl_buffer()) {
          slot.fp16 = std::move(*x_hat);
        } else {
          // The slot is no longer in flight after ReserveSlot().  Reuse its
          // old decoder buffer as the workspace buffer for the next frame.
          std::swap(*x_hat, slot.fp16);
        }
        Check(slot.fp16.has_acl_buffer() && slot.fp16.acl_valid(),
              "RTSP FP16 slot lost the decoder ACL output");
        if (!x_hat->has_acl_buffer()) {
          x_hat->Reset(std::move(conversion_shape), conversion_dtype);
          x_hat->AllocateAclBuffer(false);
        }
        conversion_input = slot.fp16.AclView().data();
        conversion_shape = slot.fp16.shape();
      }
      CheckAclCall(aclrtRecordEvent(slot.input_ready, decode_stream_),
                   "aclrtRecordEvent RTSP NV12 input ready");
      CheckAclCall(aclrtStreamWaitEvent(convert_stream_, slot.input_ready),
                   "aclrtStreamWaitEvent RTSP NV12 input ready");
      Fp16Yuv444ToNv12Acl(conversion_input, conversion_shape, layout_, slot.device.data(),
                          convert_stream_, slot.conversion_start, slot.converted, profiler_);
      if (!async_encode) {
        CheckAclCall(aclrtStreamWaitEvent(copy_stream_, slot.converted),
                     "aclrtStreamWaitEvent RTSP NV12 converted");
        slot.copy_enqueue = std::chrono::steady_clock::now();
        CheckAclCall(aclrtRecordEvent(slot.copy_start, copy_stream_),
                     "aclrtRecordEvent RTSP NV12 copy start");
        CheckAclCall(aclrtMemcpyAsync(slot.host.data(), slot.host.bytes(), slot.device.data(),
                                      slot.device.bytes(), ACL_MEMCPY_DEVICE_TO_HOST, copy_stream_),
                     "aclrtMemcpyAsync RTSP NV12 D2H");
        CheckAclCall(aclrtRecordEvent(slot.copied, copy_stream_),
                     "aclrtRecordEvent RTSP NV12 copied");
      }

      if (async_encode) {
        std::exception_ptr worker_error;
        {
          std::lock_guard<std::mutex> lock(worker_mutex_);
          worker_error = worker_error_;
          if (worker_error == nullptr) {
            pending_slots_.push_back(index);
          } else {
            slot.in_flight = false;
          }
        }
        worker_condition_.notify_all();
        if (worker_error != nullptr) std::rethrow_exception(worker_error);
      } else {
        pending_slots_.push_back(index);
      }
    } catch (...) {
      if (async_encode) {
        {
          std::lock_guard<std::mutex> lock(worker_mutex_);
          slot.in_flight = false;
        }
        worker_condition_.notify_all();
      }
      throw;
    }
    next_slot_ = (next_slot_ + 1) % slots_.size();
    if (async_encode) worker_condition_.notify_one();
  }

  void Drain() {
    if (h264_encoder_config_.has_value()) {
      std::unique_lock<std::mutex> lock(worker_mutex_);
      worker_condition_.wait(lock, [this] {
        if (worker_error_ != nullptr) return true;
        if (!pending_slots_.empty()) return false;
        return std::none_of(slots_.begin(), slots_.end(),
                            [](const Slot& slot) { return slot.in_flight; });
      });
      const std::exception_ptr worker_error = worker_error_;
      lock.unlock();
      if (worker_error != nullptr) std::rethrow_exception(worker_error);
      return;
    }
    while (!pending_slots_.empty()) {
      const std::size_t index = pending_slots_.front();
      FlushOne(index);
      pending_slots_.pop_front();
    }
  }

 private:
  struct Slot {
    mlvc::TensorHandle fp16;
    mlvc::AclBuffer device;
    mlvc::PinnedHostBuffer host;
    aclrtEvent input_ready = nullptr;
    aclrtEvent conversion_start = nullptr;
    aclrtEvent converted = nullptr;
    aclrtEvent copy_start = nullptr;
    aclrtEvent copied = nullptr;
    bool in_flight = false;
    std::chrono::steady_clock::time_point conversion_enqueue;
    std::chrono::steady_clock::time_point copy_enqueue;
  };

  void ReserveSlot(std::size_t index) {
    mlvc::ScopedCpuTimer timer(profiler_, "rtsp.venc_slot_wait");
    std::unique_lock<std::mutex> lock(worker_mutex_);
    worker_condition_.wait(lock, [this, index] {
      return worker_error_ != nullptr || !slots_.at(index).in_flight;
    });
    const std::exception_ptr worker_error = worker_error_;
    if (worker_error != nullptr) {
      lock.unlock();
      std::rethrow_exception(worker_error);
    }
    slots_.at(index).in_flight = true;
  }

  void FlushOne(std::size_t index) {
    Slot& slot = slots_.at(index);
    CheckAclCall(aclrtSynchronizeEvent(slot.converted),
                 "aclrtSynchronizeEvent RTSP NV12 converted");
    float conversion_ms = 0.0f;
    CheckAclCall(aclrtEventElapsedTime(&conversion_ms, slot.conversion_start, slot.converted),
                 "aclrtEventElapsedTime RTSP NV12 conversion");
    if (profiler_ != nullptr) {
      profiler_->AddEvent("acl_video.fp16_yuv444_to_nv12.device",
                          profiler_->StartMs(slot.conversion_enqueue), conversion_ms);
    }
    if (h264_encoder_config_.has_value()) {
      if (!h264_encoder_) {
        h264_encoder_ = std::make_unique<io::DvppH264Encoder>(context_, *h264_encoder_config_);
      }
      const auto encode_start = std::chrono::steady_clock::now();
      std::vector<uint8_t> h264 =
          h264_encoder_->EncodeDevice(slot.device.data(), slot.device.bytes(), nullptr,
                                      false);
      if (profiler_ != nullptr) {
        const float encode_ms = static_cast<float>(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                      encode_start)
                .count());
        profiler_->AddEvent("dvpp_venc.h264_mpi_encode_d2h", profiler_->StartMs(encode_start),
                            encode_ms);
      }
      publisher_->WriteH264Frame(std::move(h264));
      ++encoded_frames_;
    } else {
      CheckAclCall(aclrtSynchronizeEvent(slot.copied),
                   "aclrtSynchronizeEvent RTSP NV12 copied");
      float copy_ms = 0.0f;
      CheckAclCall(aclrtEventElapsedTime(&copy_ms, slot.copy_start, slot.copied),
                   "aclrtEventElapsedTime RTSP NV12 D2H");
      if (profiler_ != nullptr) {
        profiler_->AddEvent("copy.rtsp.nv12_d2h.device", profiler_->StartMs(slot.copy_enqueue),
                            copy_ms);
      }
      const auto* begin = static_cast<const uint8_t*>(slot.host.data());
      // FFmpeg's raw NV12 input is visible-size packed.  The ACL/DVPP
      // conversion buffer may carry 16-pixel padded strides, so strip the
      // padding before handing the demo stream to libx264.
      const std::size_t visible_y_bytes =
          static_cast<std::size_t>(layout_.width) * layout_.height;
      const std::size_t packed_bytes = visible_y_bytes + visible_y_bytes / 2;
      std::vector<uint8_t> packed(packed_bytes);
      for (int row = 0; row < layout_.height; ++row) {
        std::memcpy(packed.data() + static_cast<std::size_t>(row) * layout_.width,
                    begin + static_cast<std::size_t>(row) * layout_.width_stride,
                    static_cast<std::size_t>(layout_.width));
      }
      const std::size_t source_uv = static_cast<std::size_t>(layout_.width_stride) *
                                    layout_.height_stride;
      const std::size_t packed_uv = visible_y_bytes;
      for (int row = 0; row < layout_.height / 2; ++row) {
        std::memcpy(packed.data() + packed_uv + static_cast<std::size_t>(row) * layout_.width,
                    begin + source_uv + static_cast<std::size_t>(row) * layout_.width_stride,
                    static_cast<std::size_t>(layout_.width));
      }
      publisher_->WriteNv12Frame(std::move(packed));
    }
  }

  void SetWorkerError(std::exception_ptr error, std::optional<std::size_t> active_slot) {
    {
      std::lock_guard<std::mutex> lock(worker_mutex_);
      if (worker_error_ == nullptr) worker_error_ = std::move(error);
      worker_stopping_ = true;
      if (active_slot.has_value()) slots_.at(*active_slot).in_flight = false;
      for (const std::size_t index : pending_slots_) slots_.at(index).in_flight = false;
      pending_slots_.clear();
    }
    worker_condition_.notify_all();
  }

  void EncoderWorker() {
    try {
      CheckAclCall(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext RTSP VENC worker");
    } catch (...) {
      SetWorkerError(std::current_exception(), std::nullopt);
      return;
    }
    for (;;) {
      std::size_t index = 0;
      {
        std::unique_lock<std::mutex> lock(worker_mutex_);
        worker_condition_.wait(lock,
                               [this] { return worker_stopping_ || !pending_slots_.empty(); });
        if (pending_slots_.empty()) {
          if (worker_stopping_) return;
          continue;
        }
        index = pending_slots_.front();
        pending_slots_.pop_front();
      }
      try {
        FlushOne(index);
      } catch (...) {
        SetWorkerError(std::current_exception(), index);
        return;
      }
      {
        std::lock_guard<std::mutex> lock(worker_mutex_);
        slots_.at(index).in_flight = false;
      }
      worker_condition_.notify_all();
    }
  }

  io::Nv12Layout layout_;
  aclrtStream decode_stream_ = nullptr;
  aclrtStream convert_stream_ = nullptr;
  aclrtStream copy_stream_ = nullptr;
  mlvc::io::RtspVideoPublisher* publisher_ = nullptr;
  aclrtContext context_ = nullptr;
  std::optional<io::DvppH264EncoderConfig> h264_encoder_config_;
  std::unique_ptr<io::DvppH264Encoder> h264_encoder_;
  mlvc::Profiler* profiler_ = nullptr;
  // Keep enough converted frames in flight to hide the NV12 conversion and
  // DVPP VENC latency behind the MLVC decoder stage.
  std::array<Slot, 4> slots_;
  std::deque<std::size_t> pending_slots_;
  std::mutex worker_mutex_;
  std::condition_variable worker_condition_;
  std::thread encoder_worker_;
  std::exception_ptr worker_error_;
  bool worker_stopping_ = false;
  std::size_t next_slot_ = 0;
  std::size_t encoded_frames_ = 0;
};

}  // namespace

int RunDecodeStream(const DecodeStreamOptions& options, DecodePipelineServices* services) {
  try {
    ScopedRuntimeState runtime_state_guard;
    Check(mlvc::codec::IsCodecExecutionProfile(options.execution_profile),
          "unsupported codec execution profile: " + options.execution_profile);
    // A decode run with no file or transport output does not need a host copy of
    // the reconstructed frame.  Keep the MLVC decoder outputs device-resident
    // so reference features can flow directly into the next frame.  Entropy
    // outputs remain materialized only where the CPU entropy worker consumes
    // them.  This avoids a full-resolution D2H copy and host synchronization on
    // every 1080P frame.
    const bool use_device_resident_outputs = options.output_format == "none";
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
    const int sidecar_q_index_count = sidecar.q_index_count();
    const std::filesystem::path model_directory = models.manifest().directory();
    Check(HasMlvcModels(models.manifest()), "manifest does not contain MLVCEncoder / MLVCDecoder");
    const mlvc::ModelRecord& decoder_record = models.manifest().GetModel("MLVCDecoder");
    const mlvc::TensorSpec& x_hat_spec = mlvc::codec::OutputSpec(decoder_record, "x_hat");
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
    const auto expected_bundle_hash = mlvc::ComputeModelBundleSha256(models.manifest());
    if (udp_input) {
      if (rtp_input) {
        rtp_receiver.emplace(static_cast<uint16_t>(options.input_transport_port),
                              expected_bundle_hash, options.input_transport_payload_type);
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
    if (header.codec_bundle_sha256 != std::array<uint8_t, 32>{} &&
        header.codec_bundle_sha256 != expected_bundle_hash) {
      Check(false, "MLVC stream model bundle hash does not match the local manifest");
    }
    mlvc::io::ValidateMlvcDecoderOutputShape(header, x_hat_spec.shape);
    if (header.translation_warp) {
      RequireTranslationWarpModels(models.manifest());
      Check(header.gop == 96 && header.reset_interval == 32 && header.ltr_period == 0 &&
                options.forced_ltr_recovery_frame < 0 && options.forced_ltr_reference_frame < 0,
            "translation warp requires GOP 96, reset 32 and LTR disabled");
    }
    mlvc::io::ValidateMlvcQIndexForSidecar(header.q_index, sidecar_q_index_count);
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
    const bool rtsp_use_dvpp_h264 = options.output_transport_mode == "rtsp" &&
                                    options.output_transport_rtsp_encoder == "dvpp";
    ScopedDecodeVideoOutputMirror video_output_scope(
        write_output || (options.output_transport_mode == "rtsp" && !rtsp_use_dvpp_h264));
    std::optional<io::RtspVideoPublisher> rtsp_publisher;
    if (options.output_transport_mode == "rtsp") {
      rtsp_publisher.emplace(options.output_transport_rtsp_url, writer_fps, header.width,
                             header.height, options.output_transport_rtsp_preset,
                             options.output_transport_rtsp_crf,
                             static_cast<std::size_t>(options.output_transport_queue_capacity),
                             options.output_transport_rtsp_transport, rtsp_use_dvpp_h264);
    }
    constexpr int kDvppStrideAlignment = 16;
    const int rtsp_width_stride =
        (header.width + kDvppStrideAlignment - 1) & ~(kDvppStrideAlignment - 1);
    const int rtsp_height_stride =
        (header.height + kDvppStrideAlignment - 1) & ~(kDvppStrideAlignment - 1);
    const io::Nv12Layout rtsp_nv12_layout{header.width, header.height, rtsp_width_stride,
                                          rtsp_height_stride};
    std::optional<io::DvppH264EncoderConfig> rtsp_h264_encoder_config;
    if (rtsp_use_dvpp_h264) {
      Check(Fp16Yuv444ToNv12AclAvailable(),
            "DVPP H.264 RTSP output requires ACL FP16 YUV444 to NV12 conversion");
      io::DvppH264EncoderConfig encoder_config;
      encoder_config.layout = rtsp_nv12_layout;
      encoder_config.fps = static_cast<uint32_t>(std::max(1.0, std::round(writer_fps)));
      encoder_config.gop = encoder_config.fps * 2;
      encoder_config.bitrate = 8'000'000;
      rtsp_h264_encoder_config = encoder_config;
    }
    const bool rtsp_acl_nv12 = rtsp_publisher.has_value() && Fp16Yuv444ToNv12AclAvailable();
    std::optional<AsyncRtspNv12Pipeline> rtsp_nv12_pipeline;
    if (rtsp_acl_nv12) {
      rtsp_nv12_pipeline.emplace(rtsp_nv12_layout, runtime.stream(), &*rtsp_publisher,
                                 runtime.context(), std::move(rtsp_h264_encoder_config), &profiler);
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
          std::make_unique<MlvcOfficialEntropyDecoder>(models.manifest()));
    }
    mlvc::app::OrderedFutureWindow<DecodedEntropyFrame> entropy_window(
        options.pipeline.entropy_workers);
    const auto decode_start = std::chrono::steady_clock::now();
    int decoded_frames = 0;
    int input_frames_seen = 0;
    int expected_udp_frame_index = 0;
    int active_short_reference_frame = -1;
    uint64_t bitstream_bytes = 0;
    uint64_t geometry_bytes = 0;
    uint64_t motion_nonzero_frames = 0;
    auto submit_entropy = [&](std::size_t slot, int frame_index, MlvcFrameType frame_type,
                              int q_index, mlvc::io::MlvcFrameMetadata metadata,
                              std::vector<uint8_t> payload) -> std::future<DecodedEntropyFrame> {
      return entropy_worker.SubmitValue<DecodedEntropyFrame>(
          [&, slot, frame_index, frame_type, q_index, metadata,
           payload = std::move(payload)]() mutable {
            auto decoded = DecodeMlvcEntropyFrame(entropy_decoders.at(slot).get(), decoder_record,
                                                  frame_index, frame_type, q_index,
                                                  std::move(payload), &profiler);
            decoded.metadata = metadata;
            return decoded;
          });
    };

    auto read_next_packet = [&]() -> std::shared_ptr<mlvc::app::BitstreamPacket> {
      while (options.frame_num <= 0 || input_frames_seen < options.frame_num) {
        int next_frame_index = 0;
        MlvcFrameType next_frame_type = MlvcFrameType::kPFrame;
        int next_q_index = 0;
        std::vector<uint8_t> next_payload;
        mlvc::io::MlvcFrameMetadata next_metadata;
        bool received = false;
        if (rtp_input) {
          mlvc::ScopedCpuTimer timer(&profiler, "rtp.receive_queue");
          received = rtp_receiver->ReceiveFrame(&next_frame_index, &next_frame_type, &next_q_index,
                                                &next_payload);
          if (received) next_metadata = rtp_receiver->last_frame_metadata();
        } else if (udp_input) {
          mlvc::ScopedCpuTimer timer(&profiler, "udp.receive_queue");
          received = udp_receiver->ReceiveFrame(&next_frame_index, &next_frame_type, &next_q_index,
                                                &next_payload);
          if (received) next_metadata = udp_receiver->last_frame_metadata();
        } else {
          received = bitstream_reader->ReadFrame(&next_frame_index, &next_frame_type, &next_q_index,
                                                 &next_payload);
          if (received) next_metadata = bitstream_reader->last_frame_metadata();
        }
        if (!received) {
          return nullptr;
        }
        mlvc::io::ValidateMlvcQIndexForSidecar(next_q_index, sidecar_q_index_count);
        if (next_metadata.explicit_metadata && next_metadata.model_q_index >= 0) {
          mlvc::io::ValidateMlvcQIndexForSidecar(next_metadata.model_q_index,
                                                 sidecar_q_index_count);
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
        if (header.translation_warp) {
          Check(next_metadata.translation_warp, "warp stream frame is missing translation metadata");
          if (next_frame_type != MlvcFrameType::kIFrame) geometry_bytes += 2;
          if (next_metadata.kx != 0 || next_metadata.ky != 0) ++motion_nonzero_frames;
        }
        auto packet = std::make_shared<mlvc::app::BitstreamPacket>();
        packet->frame_index = next_frame_index;
        packet->frame_type = next_frame_type;
        packet->q_index = next_q_index;
        packet->metadata = next_metadata;
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
      const bool explicit_metadata = decoded.metadata.explicit_metadata;
      const bool is_i_frame = frame_type == MlvcFrameType::kIFrame;
      const bool use_ltr_recovery = frame_type == MlvcFrameType::kLtrRecovery;
      const int gop_cycle_index = header.gop > 0 ? frame_index % header.gop : frame_index;
      const bool wire_reset_reference = explicit_metadata
                                       ? (decoded.metadata.unit_flags &
                                          mlvc::transport::kEfuResetReference) != 0
                                       : ShouldResetReferenceFeature(frame_index, header.gop,
                                                                      header.reset_interval);
      const bool reset_reference = header.translation_warp
          ? is_i_frame || frame_index % header.reset_interval == 0 : wire_reset_reference;
      if (reset_reference) {
        state.ResetFeature();
        if (wire_reset_reference) active_short_reference_frame = -1;
      }
      if (explicit_metadata && frame_type == MlvcFrameType::kPFrame && !wire_reset_reference) {
        Check(active_short_reference_frame >= 0 &&
                  decoded.metadata.short_ref_frame_id ==
                      static_cast<uint32_t>(active_short_reference_frame),
              "MLVC short reference does not match the currently available decoded frame");
      }
      if (is_i_frame) {
        has_ltr_feature = false;
        FillFp16Tensor(0.0f, &ltr_feature);
        ltr_features.clear();
      }
      if (use_ltr_recovery) {
        Check(has_ltr_feature, "LTR recovery frame has no cached LTR feature");
        const int reference_frame = explicit_metadata
                                         ? static_cast<int>(decoded.metadata.long_ref_frame_id)
                                         : (frame_index == forced_ltr_recovery_frame &&
                                                    forced_ltr_reference_frame >= 0
                                                ? forced_ltr_reference_frame
                                                : ltr_features.rbegin()->first);
        const auto ltr_it = ltr_features.find(reference_frame);
        Check(ltr_it != ltr_features.end(),
              "requested LTR reference frame is not cached: " + std::to_string(reference_frame));
        state.feature = CloneTensor(ltr_it->second);
        state.feature_handle.reset();
      }
      if (header.translation_warp && reset_reference) {
        PrepareWarpResetReference(&models, &state, x_hat_spec.shape, is_i_frame, &profiler);
      }

      const int frame_adaptation_index = kIndexMap[(frame_index + 1) % 8];
      const int q_index_shifted = explicit_metadata && decoded.metadata.model_q_index >= 0
                                      ? decoded.metadata.model_q_index
                                      : sidecar.ShiftedQp(q_index, frame_adaptation_index);
      TensorData q_index_shifted_tensor = MakeInt32ScalarTensor(q_index_shifted);
      TensorData warped_reference;
      const StageInput ref_feature_input = header.translation_warp
          ? BuildWarpedReferenceFeatureInput(&state, zero_feature, decoded.metadata.kx,
                                              decoded.metadata.ky, &warped_reference)
          : BuildReferenceFeatureInput(state, zero_feature);
      RunOutput decoder_output =
          RunStage(&models, "MLVCDecoder",
                   {TensorInput("z_raw", decoded.z_raw), TensorInput("y_raw_0", decoded.y_raw_0),
                    TensorInput("y_raw_1", decoded.y_raw_1), ref_feature_input,
                   TensorInput("q_index_shifted", q_index_shifted_tensor)},
                   &profiler);
      if (header.translation_warp && (frame_index + 1) % header.reset_interval == 0 &&
          (frame_index + 1) % header.gop != 0) {
        SaveWarpResetFrame(decoder_output, &state, &profiler);
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
      if (device_only_outputs) {
        // The decoder's feature output is also a single reusable ACL buffer.
        // Exchange it with the previous reference buffer instead of cloning
        // the full feature tensor every frame.  The old reference buffer is
        // returned to the decoder workspace for the next invocation.
        mlvc::TensorHandle* feature = decoder_output.Handle("feature");
        if (feature != nullptr && feature->has_acl_buffer() && feature->acl_valid()) {
          const mlvc::TensorShape feature_shape = feature->shape();
          const mlvc::DataType feature_dtype = feature->dtype();
          if (state.feature_handle.has_value()) {
            std::swap(*feature, *state.feature_handle);
          } else {
            state.feature.reset();
            state.feature_handle.emplace(std::move(*feature));
            feature->Reset(feature_shape, feature_dtype);
            feature->AllocateAclBuffer(false);
          }
        } else {
          UpdateReferenceFeature(decoder_output, &state, &profiler);
        }
      } else {
        UpdateReferenceFeature(decoder_output, &state, &profiler);
      }
      const bool mark_as_ltr = explicit_metadata
                                  ? (decoded.metadata.unit_flags & mlvc::transport::kEfuStoreAsLtr) != 0
                                  : (header.ltr_period > 0 &&
                                     (gop_cycle_index == header.ltr_start_idx ||
                                      (gop_cycle_index > header.ltr_start_idx &&
                                       gop_cycle_index % header.ltr_period == 0)));
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
      active_short_reference_frame = frame_index;
      if (header.translation_warp) TraceWarpReferenceState(frame_index, &state);
      ++decoded_frames;
    };

    auto consume_packet = [&](const std::shared_ptr<mlvc::DataObject>& data) {
      auto packet = std::dynamic_pointer_cast<mlvc::app::BitstreamPacket>(data);
      Check(packet != nullptr, "bitstream pipeline returned an unexpected data object");
      std::optional<DecodedEntropyFrame> ready;
      if (entropy_window.full()) {
        {
          mlvc::ScopedCpuTimer timer(&profiler, "decode.entropy_wait");
          ready = entropy_window.PopFront();
        }
      }
      entropy_window.SubmitNext([&](std::size_t slot) {
        return submit_entropy(slot, packet->frame_index, packet->frame_type, packet->q_index,
                              packet->metadata, std::move(packet->payload));
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
      DecodedEntropyFrame decoded;
      {
        mlvc::ScopedCpuTimer timer(&profiler, "decode.entropy_wait");
        decoded = entropy_window.PopFront();
      }
      run_decoded_frame(decoded);
    }
    if (rtsp_nv12_pipeline.has_value()) {
      rtsp_nv12_pipeline->Drain();
    }
    bitstream_pipeline.Stop();
    bitstream_pipeline.RethrowIfFailed();
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
    if (header.translation_warp) {
      std::cout << "translation_warp=true\n";
      std::cout << "payload_bytes=" << bitstream_bytes << "\n";
      std::cout << "geometry_bytes=" << geometry_bytes << "\n";
      std::cout << "codec_total_bytes=" << bitstream_bytes + geometry_bytes << "\n";
      std::cout << "motion_nonzero_frames=" << motion_nonzero_frames << "\n";
      std::cout << "codec_bitrate_bps=" << (decoded_frames > 0 ?
          static_cast<double>(bitstream_bytes + geometry_bytes) * 8.0 * writer_fps / decoded_frames : 0.0) << "\n";
      if (!udp_input) {
        std::cout << "file_total_bytes=" << std::filesystem::file_size(options.input_bitstream_path) << "\n";
      }
    }
    std::cout << "drop_frame_index=" << options.drop_frame_index << "\n";
    std::cout << "forced_ltr_reference_frame=" << forced_ltr_reference_frame << "\n";
    std::cout << "forced_ltr_recovery_frame=" << forced_ltr_recovery_frame << "\n";
    std::cout << "output_video=" << (write_output ? options.output_video_path.string() : "none")
              << "\n";
    std::cout << "output_transport_target="
              << (rtsp_publisher.has_value() ? options.output_transport_rtsp_url : "none")
              << "\n";
    std::cout << "output_transport_mode="
              << (rtsp_publisher.has_value() ? "rtsp" : "none")
              << "\n";
    std::cout << "output_transport_rtsp_encoder="
              << (rtsp_publisher.has_value() ? options.output_transport_rtsp_encoder : "none")
              << "\n";
    std::cout << "output_transport_rtsp_frames="
              << (rtsp_publisher.has_value() ? rtsp_publisher->frame_count() : 0) << "\n";
    std::cout << "forward_dropped_frames="
              << (rtsp_publisher.has_value() ? rtsp_publisher->dropped_frames() : 0)
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
