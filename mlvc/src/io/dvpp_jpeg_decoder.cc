#include <mlvc/io/dvpp_jpeg_decoder.h>

#include <acl/ops/acl_dvpp.h>

#include <cstddef>
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <string>

#include "mlvc/core/status.h"

namespace mlvc::io {
namespace {

void CheckAcl(aclError status, const char* operation) {
  if (status != ACL_SUCCESS) {
    throw Error(std::string(operation) + " failed: ret=" + std::to_string(status));
  }
}

int Align16(int value) { return (value + 15) / 16 * 16; }

class DvppSurfacePool : public std::enable_shared_from_this<DvppSurfacePool> {
 public:
  DvppSurfacePool(aclrtContext context, std::size_t buffer_bytes, std::size_t count)
      : context_(context), buffer_bytes_(buffer_bytes) {
    Check(count > 0, "DVPP JPEG output surface pool must not be empty");
    CheckAcl(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext JPEGD pool");
    buffers_.reserve(count);
    try {
      for (std::size_t index = 0; index < count; ++index) {
        void* buffer = nullptr;
        CheckAcl(acldvppMalloc(&buffer, buffer_bytes_), "acldvppMalloc JPEGD surface pool");
        buffers_.push_back(buffer);
        free_.push_back(index);
      }
    } catch (...) {
      for (void* buffer : buffers_) (void)acldvppFree(buffer);
      buffers_.clear();
      throw;
    }
  }

  ~DvppSurfacePool() {
    (void)aclrtSetCurrentContext(context_);
    for (void* buffer : buffers_) {
      if (buffer != nullptr) (void)acldvppFree(buffer);
    }
  }

  std::pair<void*, std::shared_ptr<void>> Acquire() {
    std::unique_lock<std::mutex> lock(mutex_);
    available_.wait(lock, [this] { return !free_.empty(); });
    const std::size_t index = free_.front();
    free_.pop_front();
    void* buffer = buffers_[index];
    lock.unlock();
    auto lease = std::shared_ptr<void>(buffer, [pool = shared_from_this(), index](void*) {
      pool->Release(index);
    });
    return {buffer, std::move(lease)};
  }

  std::size_t buffer_bytes() const { return buffer_bytes_; }

 private:
  void Release(std::size_t index) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      free_.push_back(index);
    }
    available_.notify_one();
  }

  aclrtContext context_ = nullptr;
  std::size_t buffer_bytes_ = 0;
  std::vector<void*> buffers_;
  std::deque<std::size_t> free_;
  std::mutex mutex_;
  std::condition_variable available_;
};

}  // namespace

class DvppJpegDecoder::Impl {
 public:
  Impl(aclrtContext context, DvppJpegDecodeConfig config)
      : context_(context), config_(config) {
    Check(context_ != nullptr, "DVPP JPEG decoder requires an ACL context");
    Check(config_.width > 0 && config_.height > 0 && config_.width % 2 == 0 &&
              config_.height % 2 == 0,
          "DVPP JPEG decoder dimensions must be positive and even");
    CheckAcl(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext JPEGD");
    width_stride_ = Align16(config_.width);
    height_stride_ = Align16(config_.height);
    output_bytes_ = static_cast<std::size_t>(width_stride_) * height_stride_ * 3 / 2;
    const std::size_t allocation_bytes = output_bytes_ * 2;
    surface_pool_ = std::make_shared<DvppSurfacePool>(context_, allocation_bytes,
                                                      config_.output_surface_count);
    channel_desc_ = acldvppCreateChannelDesc();
    Check(channel_desc_ != nullptr, "acldvppCreateChannelDesc failed");
    CheckAcl(acldvppCreateChannel(channel_desc_), "acldvppCreateChannel");
    channel_created_ = true;
    CheckAcl(aclrtCreateStream(&stream_), "aclrtCreateStream JPEGD");
  }

  ~Impl() {
    if (stream_ != nullptr) aclrtDestroyStream(stream_);
    if (channel_created_) acldvppDestroyChannel(channel_desc_);
    if (channel_desc_ != nullptr) acldvppDestroyChannelDesc(channel_desc_);
  }

  DvppJpegDecodedFrame DecodeDevice(const void* jpeg, std::size_t jpeg_bytes) {
    Check(jpeg != nullptr && jpeg_bytes > 0, "DVPP JPEG input must not be empty");
    CheckAcl(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext JPEGD decode");
    uint32_t predicted = 0;
    Check(jpeg_bytes <= std::numeric_limits<uint32_t>::max(),
          "DVPP JPEG input exceeds the API size limit");
    CheckAcl(acldvppJpegPredictDecSize(static_cast<const uint8_t*>(jpeg),
                                       static_cast<uint32_t>(jpeg_bytes),
                                       PIXEL_FORMAT_YUV_SEMIPLANAR_420, &predicted),
             "acldvppJpegPredictDecSize");
    Check(predicted <= surface_pool_->buffer_bytes(),
          "DVPP JPEGD predicted output exceeds the reusable surface size");
    auto [output_device, owner] = surface_pool_->Acquire();
    acldvppPicDesc* output_desc = acldvppCreatePicDesc();
    Check(output_desc != nullptr, "acldvppCreatePicDesc failed");
    try {
      CheckAcl(acldvppSetPicDescData(output_desc, output_device), "acldvppSetPicDescData");
      CheckAcl(acldvppSetPicDescSize(output_desc,
                                    static_cast<uint32_t>(surface_pool_->buffer_bytes())),
               "acldvppSetPicDescSize");
      CheckAcl(acldvppSetPicDescFormat(output_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420),
               "acldvppSetPicDescFormat");
      CheckAcl(acldvppSetPicDescWidth(output_desc, config_.width), "acldvppSetPicDescWidth");
      CheckAcl(acldvppSetPicDescHeight(output_desc, config_.height), "acldvppSetPicDescHeight");
      CheckAcl(acldvppSetPicDescWidthStride(output_desc, width_stride_),
               "acldvppSetPicDescWidthStride");
      CheckAcl(acldvppSetPicDescHeightStride(output_desc, height_stride_),
               "acldvppSetPicDescHeightStride");
      CheckAcl(acldvppJpegDecodeAsync(channel_desc_, static_cast<const uint8_t*>(jpeg),
                                      static_cast<uint32_t>(jpeg_bytes), output_desc, stream_),
               "acldvppJpegDecodeAsync");
      CheckAcl(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream JPEGD");
      acldvppDestroyPicDesc(output_desc);
      output_desc = nullptr;
      return DvppJpegDecodedFrame(std::move(owner), output_bytes_, width_stride_, height_stride_);
    } catch (...) {
      if (output_desc != nullptr) acldvppDestroyPicDesc(output_desc);
      throw;
    }
  }

  DvppJpegDecodedFrame DecodeDevice(const std::vector<uint8_t>& jpeg) {
    return DecodeDevice(jpeg.data(), jpeg.size());
  }

  std::vector<uint8_t> Decode(const std::vector<uint8_t>& jpeg) {
    DvppJpegDecodedFrame frame = DecodeDevice(jpeg);
    std::vector<uint8_t> output(frame.bytes());
    CheckAcl(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext JPEGD copy");
    CheckAcl(aclrtMemcpy(output.data(), output.size(), frame.device_data(), output.size(),
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "aclrtMemcpy JPEGD output");
    return output;
  }

 private:
  aclrtContext context_ = nullptr;
  DvppJpegDecodeConfig config_;
  int width_stride_ = 0;
  int height_stride_ = 0;
  std::size_t output_bytes_ = 0;
  std::shared_ptr<DvppSurfacePool> surface_pool_;
  acldvppChannelDesc* channel_desc_ = nullptr;
  aclrtStream stream_ = nullptr;
  bool channel_created_ = false;
};

DvppJpegDecoder::DvppJpegDecoder(aclrtContext context, DvppJpegDecodeConfig config)
    : impl_(std::make_unique<Impl>(context, config)) {}

DvppJpegDecoder::~DvppJpegDecoder() = default;

std::vector<uint8_t> DvppJpegDecoder::Decode(const std::vector<uint8_t>& jpeg) {
  return impl_->Decode(jpeg);
}

DvppJpegDecodedFrame DvppJpegDecoder::DecodeDevice(const std::vector<uint8_t>& jpeg) {
  return impl_->DecodeDevice(jpeg);
}

DvppJpegDecodedFrame DvppJpegDecoder::DecodeDevice(const void* jpeg, std::size_t jpeg_bytes) {
  return impl_->DecodeDevice(jpeg, jpeg_bytes);
}

}  // namespace mlvc::io
