#include <mlvc/io/dvpp_h264_encoder.h>

#include <acl/ops/acl_dvpp.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <atomic>
#include <utility>
#include <vector>

#include "mlvc/core/status.h"
#include "mlvc/runtime/acl_runtime.h"

namespace mlvc::io {

class DvppH264Encoder::Impl {
 public:
  Impl(aclrtContext context, DvppH264EncoderConfig config)
      : context_(context), config_(std::move(config)) {
    Check(context_ != nullptr, "DVPP H.264 encoder requires an ACL context");
    Check(config_.layout.width > 0 && config_.layout.height > 0 &&
              config_.layout.width % 2 == 0 && config_.layout.height % 2 == 0,
          "DVPP H.264 dimensions must be positive and even");
    Check(config_.layout.width_stride >= config_.layout.width &&
              config_.layout.height_stride >= config_.layout.height &&
              config_.layout.width_stride % 2 == 0 && config_.layout.height_stride % 2 == 0,
          "DVPP H.264 strides are invalid");
    Check(config_.fps > 0 && config_.gop > 0 && config_.bitrate > 0,
          "DVPP H.264 rate configuration is invalid");
    input_bytes_ = Nv12BufferSize(config_.layout);
    try {
      CheckAcl(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext for DVPP H.264");
      channel_desc_ = aclvencCreateChannelDesc();
      Check(channel_desc_ != nullptr, "aclvencCreateChannelDesc failed");
      CheckAcl(aclvencSetChannelDescCallback(channel_desc_, &Impl::Callback),
               "aclvencSetChannelDescCallback");
      CheckAcl(aclvencSetChannelDescEnType(channel_desc_, H264_MAIN_LEVEL),
               "aclvencSetChannelDescEnType H.264");
      CheckAcl(aclvencSetChannelDescPicFormat(channel_desc_, PIXEL_FORMAT_YUV_SEMIPLANAR_420),
               "aclvencSetChannelDescPicFormat NV12");
      CheckAcl(aclvencSetChannelDescPicWidth(channel_desc_,
                                             static_cast<uint32_t>(config_.layout.width)),
               "aclvencSetChannelDescPicWidth");
      CheckAcl(aclvencSetChannelDescPicHeight(channel_desc_,
                                              static_cast<uint32_t>(config_.layout.height)),
               "aclvencSetChannelDescPicHeight");
      CheckAcl(aclvencSetChannelDescKeyFrameInterval(channel_desc_, config_.gop),
               "aclvencSetChannelDescKeyFrameInterval");
      CheckAcl(aclvencSetChannelDescRcMode(channel_desc_, 1),
               "aclvencSetChannelDescRcMode VBR");
      CheckAcl(aclvencSetChannelDescSrcRate(channel_desc_, config_.fps),
               "aclvencSetChannelDescSrcRate");
      CheckAcl(aclvencSetChannelDescMaxBitRate(channel_desc_, config_.bitrate),
               "aclvencSetChannelDescMaxBitRate");
      StartReportThread();
      CheckAcl(aclvencCreateChannel(channel_desc_), "aclvencCreateChannel");
      channel_created_ = true;
      picture_desc_ = acldvppCreatePicDesc();
      Check(picture_desc_ != nullptr, "acldvppCreatePicDesc for DVPP H.264 failed");
      CheckAcl(acldvppSetPicDescSize(picture_desc_, static_cast<uint32_t>(input_bytes_)),
               "acldvppSetPicDescSize DVPP H.264");
      CheckAcl(acldvppSetPicDescFormat(picture_desc_, PIXEL_FORMAT_YUV_SEMIPLANAR_420),
               "acldvppSetPicDescFormat DVPP H.264");
      CheckAcl(acldvppSetPicDescWidth(picture_desc_, static_cast<uint32_t>(config_.layout.width)),
               "acldvppSetPicDescWidth DVPP H.264");
      CheckAcl(acldvppSetPicDescHeight(picture_desc_, static_cast<uint32_t>(config_.layout.height)),
               "acldvppSetPicDescHeight DVPP H.264");
      CheckAcl(acldvppSetPicDescWidthStride(
                   picture_desc_, static_cast<uint32_t>(config_.layout.width_stride)),
               "acldvppSetPicDescWidthStride DVPP H.264");
      CheckAcl(acldvppSetPicDescHeightStride(
                   picture_desc_, static_cast<uint32_t>(config_.layout.height_stride)),
               "acldvppSetPicDescHeightStride DVPP H.264");
    } catch (...) {
      Cleanup();
      throw;
    }
  }

  ~Impl() { Cleanup(); }

  std::vector<uint8_t> EncodeDevice(const void* nv12_device, std::size_t bytes,
                                    aclrtEvent ready_event, bool force_keyframe) {
    Check(nv12_device != nullptr, "DVPP H.264 input must not be null");
    Check(bytes == input_bytes_, "DVPP H.264 input NV12 size mismatch");
    CheckAcl(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext DVPP H.264 encode");
    std::unique_lock<std::mutex> lock(mutex_);
    Check(!in_flight_, "DVPP H.264 encoder has an in-flight frame");
    error_ = nullptr;
    output_.clear();
    in_flight_ = true;
    lock.unlock();

    CheckAcl(acldvppSetPicDescData(picture_desc_, const_cast<void*>(nv12_device)),
             "acldvppSetPicDescData DVPP H.264");
    if (ready_event != nullptr) {
      CheckAcl(aclrtCreateStream(&wait_stream_), "aclrtCreateStream DVPP H.264 wait");
      CheckAcl(aclrtStreamWaitEvent(wait_stream_, ready_event),
               "aclrtStreamWaitEvent DVPP H.264 input");
      CheckAcl(aclrtSynchronizeStream(wait_stream_), "aclrtSynchronizeStream DVPP H.264 wait");
      CheckAcl(aclrtDestroyStream(wait_stream_), "aclrtDestroyStream DVPP H.264 wait");
      wait_stream_ = nullptr;
    }
    aclvencFrameConfig* frame_config = aclvencCreateFrameConfig();
    Check(frame_config != nullptr, "aclvencCreateFrameConfig failed");
    try {
      CheckAcl(aclvencSetFrameConfigForceIFrame(frame_config, force_keyframe ? 1 : 0),
               "aclvencSetFrameConfigForceIFrame");
      CheckAcl(aclvencSendFrame(channel_desc_, picture_desc_, nullptr, frame_config, this),
               "aclvencSendFrame DVPP H.264");
    } catch (...) {
      (void)aclvencDestroyFrameConfig(frame_config);
      std::lock_guard<std::mutex> error_lock(mutex_);
      in_flight_ = false;
      throw;
    }
    CheckAcl(aclvencDestroyFrameConfig(frame_config), "aclvencDestroyFrameConfig");

    lock.lock();
    condition_.wait(lock, [this] { return !in_flight_ || error_ != nullptr; });
    if (error_ != nullptr) std::rethrow_exception(error_);
    return std::move(output_);
  }

 private:
  void StartReportThread() {
    {
      std::lock_guard<std::mutex> lock(report_mutex_);
      report_running_.store(true, std::memory_order_release);
      report_ready_ = false;
      report_error_ = nullptr;
    }
    report_thread_ = std::thread([this] {
      try {
        CheckAcl(aclrtSetCurrentContext(context_),
                 "aclrtSetCurrentContext DVPP H.264 report thread");
        {
          std::lock_guard<std::mutex> lock(report_mutex_);
          report_thread_id_ = static_cast<uint64_t>(
              reinterpret_cast<uintptr_t>(report_thread_.native_handle()));
          report_ready_ = true;
        }
        report_condition_.notify_one();
        while (report_running_.load(std::memory_order_acquire)) {
          const aclError status = aclrtProcessReport(100);
          if (status != ACL_ERROR_NONE &&
              status != ACL_ERROR_RT_REPORT_TIMEOUT) {
            throw Error("aclrtProcessReport DVPP H.264 failed: ret=" +
                        std::to_string(status));
          }
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(report_mutex_);
        report_error_ = std::current_exception();
        report_ready_ = true;
        report_running_.store(false, std::memory_order_release);
        report_condition_.notify_one();
      }
    });
    std::unique_lock<std::mutex> lock(report_mutex_);
    report_condition_.wait(lock, [this] { return report_ready_; });
    if (report_error_ != nullptr) {
      std::rethrow_exception(report_error_);
    }
    CheckAcl(aclvencSetChannelDescThreadId(channel_desc_, report_thread_id_),
             "aclvencSetChannelDescThreadId");
  }

  static void Callback(acldvppPicDesc* /*input*/, acldvppStreamDesc* stream, void* userdata) {
    auto* self = static_cast<Impl*>(userdata);
    try {
      Check(stream != nullptr, "DVPP H.264 callback stream is null");
      const uint32_t size = acldvppGetStreamDescSize(stream);
      void* data = acldvppGetStreamDescData(stream);
      Check(data != nullptr && size > 0, "DVPP H.264 callback returned empty stream");
      std::vector<uint8_t> output(size);
      CheckAcl(aclrtSetCurrentContext(self->context_),
               "aclrtSetCurrentContext DVPP H.264 callback");
      CheckAcl(aclrtMemcpy(output.data(), output.size(), data, size, ACL_MEMCPY_DEVICE_TO_HOST),
               "aclrtMemcpy DVPP H.264 output");
      std::lock_guard<std::mutex> lock(self->mutex_);
      self->output_ = std::move(output);
      self->in_flight_ = false;
    } catch (...) {
      std::lock_guard<std::mutex> lock(self->mutex_);
      self->error_ = std::current_exception();
      self->in_flight_ = false;
    }
    self->condition_.notify_one();
  }

  void Cleanup() noexcept {
    report_running_.store(false, std::memory_order_release);
    if (report_thread_.joinable()) {
      report_thread_.join();
    }
    if (context_ != nullptr) (void)aclrtSetCurrentContext(context_);
    if (wait_stream_ != nullptr) {
      (void)aclrtDestroyStream(wait_stream_);
      wait_stream_ = nullptr;
    }
    if (channel_created_) {
      (void)aclvencDestroyChannel(channel_desc_);
      channel_created_ = false;
    }
    if (picture_desc_ != nullptr) {
      (void)acldvppDestroyPicDesc(picture_desc_);
      picture_desc_ = nullptr;
    }
    if (channel_desc_ != nullptr) {
      (void)aclvencDestroyChannelDesc(channel_desc_);
      channel_desc_ = nullptr;
    }
  }

  aclrtContext context_ = nullptr;
  DvppH264EncoderConfig config_;
  std::size_t input_bytes_ = 0;
  aclvencChannelDesc* channel_desc_ = nullptr;
  acldvppPicDesc* picture_desc_ = nullptr;
  aclrtStream wait_stream_ = nullptr;
  bool channel_created_ = false;
  std::thread report_thread_;
  std::atomic<bool> report_running_{false};
  std::mutex report_mutex_;
  std::condition_variable report_condition_;
  bool report_ready_ = false;
  uint64_t report_thread_id_ = 0;
  std::exception_ptr report_error_;
  std::mutex mutex_;
  std::condition_variable condition_;
  bool in_flight_ = false;
  std::vector<uint8_t> output_;
  std::exception_ptr error_;
};

DvppH264Encoder::DvppH264Encoder(aclrtContext context, DvppH264EncoderConfig config)
    : impl_(std::make_unique<Impl>(context, std::move(config))) {}

DvppH264Encoder::~DvppH264Encoder() = default;

std::vector<uint8_t> DvppH264Encoder::EncodeDevice(const void* nv12_device, std::size_t bytes,
                                                   aclrtEvent ready_event, bool force_keyframe) {
  return impl_->EncodeDevice(nv12_device, bytes, ready_event, force_keyframe);
}

}  // namespace mlvc::io
