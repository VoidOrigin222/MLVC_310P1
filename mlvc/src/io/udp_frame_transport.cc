#include "mlvc/io/udp_frame_transport.h"

#include <acl/ops/acl_dvpp.h>
#include <cstring>
#include <chrono>
#include <limits>
#include <opencv2/imgcodecs.hpp>

#include "mlvc/core/status.h"
#include "mlvc/io/dvpp_jpeg_encoder.h"
#include "mlvc/io/fp16_yuv444_to_nv12.h"
#include "mlvc/io/video_io.h"
#include "mlvc/runtime/acl_runtime.h"
#include "mlvc/runtime/fp16_yuv444_to_nv12_acl.h"

namespace mlvc::io {
namespace {
constexpr uint8_t kHeaderMessage = 1;
constexpr uint8_t kFrameMessage = 2;
constexpr uint8_t kEndMessage = 3;
constexpr uint8_t kRawYuvMessage = 4;

void PutU32(std::vector<uint8_t>* out, uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    out->push_back(static_cast<uint8_t>(value >> shift));
  }
}

void PutI32(std::vector<uint8_t>* out, int32_t value) {
  PutU32(out, static_cast<uint32_t>(value));
}

void PutU64(std::vector<uint8_t>* out, uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8) {
    out->push_back(static_cast<uint8_t>(value >> shift));
  }
}

void PutDouble(std::vector<uint8_t>* out, double value) {
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  PutU64(out, bits);
}

uint32_t ReadU32(const uint8_t* data) {
  return static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8) |
         (static_cast<uint32_t>(data[2]) << 16) | (static_cast<uint32_t>(data[3]) << 24);
}

int32_t ReadI32(const uint8_t* data) {
  return static_cast<int32_t>(ReadU32(data));
}

uint64_t ReadU64(const uint8_t* data) {
  uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 8) {
    value |= static_cast<uint64_t>(data[shift / 8]) << shift;
  }
  return value;
}

double ReadDouble(const uint8_t* data) {
  const uint64_t bits = ReadU64(data);
  double value = 0.0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

}  // namespace

class detail::DeviceDvppSlotQueue::Impl {
 public:
  explicit Impl(std::size_t capacity) : states_(capacity, State::kFree) {
    Check(capacity > 0, "device DVPP slot queue capacity must be positive");
    for (std::size_t slot = 0; slot < capacity; ++slot) free_.push_back(slot);
  }

  std::size_t Acquire() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return error_ != nullptr || stopping_ || !free_.empty(); });
    RethrowIfFailed();
    Check(!stopping_, "device DVPP slot queue is stopping");
    const std::size_t slot = free_.front();
    free_.pop_front();
    Check(states_.at(slot) == State::kFree, "device DVPP free slot state is invalid");
    states_[slot] = State::kAcquired;
    return slot;
  }

  void Enqueue(std::size_t slot, int frame_index) {
    std::lock_guard<std::mutex> lock(mutex_);
    RethrowIfFailed();
    Check(!stopping_, "device DVPP slot queue is stopping");
    Check(slot < states_.size() && states_[slot] == State::kAcquired,
          "device DVPP enqueue requires an acquired slot");
    states_[slot] = State::kQueued;
    pending_.push_back({slot, frame_index});
    cv_.notify_all();
  }

  bool Pop(detail::DeviceDvppPendingSlot* pending) {
    Check(pending != nullptr, "device DVPP pending slot output is required");
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return error_ != nullptr || stopping_ || !pending_.empty(); });
    RethrowIfFailed();
    if (pending_.empty()) return false;
    *pending = pending_.front();
    pending_.pop_front();
    Check(pending->slot < states_.size() && states_[pending->slot] == State::kQueued,
          "device DVPP queued slot state is invalid");
    states_[pending->slot] = State::kProcessing;
    return true;
  }

  void Release(std::size_t slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    Check(slot < states_.size() &&
              (states_[slot] == State::kAcquired || states_[slot] == State::kProcessing),
          "device DVPP release requires an acquired or processing slot");
    states_[slot] = State::kFree;
    free_.push_back(slot);
    cv_.notify_all();
  }

  void Fail(std::exception_ptr error) {
    Check(error != nullptr, "device DVPP slot queue failure must contain an exception");
    std::lock_guard<std::mutex> lock(mutex_);
    if (error_ == nullptr) error_ = error;
    stopping_ = true;
    cv_.notify_all();
  }

  void Shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
    cv_.notify_all();
  }

 private:
  enum class State { kFree, kAcquired, kQueued, kProcessing };

  void RethrowIfFailed() const {
    if (error_ != nullptr) std::rethrow_exception(error_);
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::size_t> free_;
  std::deque<detail::DeviceDvppPendingSlot> pending_;
  std::vector<State> states_;
  std::exception_ptr error_;
  bool stopping_ = false;
};

detail::DeviceDvppSlotQueue::DeviceDvppSlotQueue(std::size_t capacity)
    : impl_(std::make_unique<Impl>(capacity)) {}

detail::DeviceDvppSlotQueue::~DeviceDvppSlotQueue() = default;

std::size_t detail::DeviceDvppSlotQueue::Acquire() { return impl_->Acquire(); }

void detail::DeviceDvppSlotQueue::Enqueue(std::size_t slot, int frame_index) {
  impl_->Enqueue(slot, frame_index);
}

bool detail::DeviceDvppSlotQueue::Pop(DeviceDvppPendingSlot* pending) {
  return impl_->Pop(pending);
}

void detail::DeviceDvppSlotQueue::Release(std::size_t slot) { impl_->Release(slot); }

void detail::DeviceDvppSlotQueue::Fail(std::exception_ptr error) {
  impl_->Fail(error);
}

void detail::DeviceDvppSlotQueue::Shutdown() { impl_->Shutdown(); }

UdpMlvcSender::UdpMlvcSender(const std::string& host, uint16_t port)
    : sender_(host, port, "MLVC") {}

UdpMlvcSender::~UdpMlvcSender() = default;

void UdpMlvcSender::SendHeader(const MlvcBitstreamHeader& header) {
  std::vector<uint8_t> message;
  message.reserve(57);
  message.push_back(kHeaderMessage);
  PutU32(&message, header.version != 0 ? header.version : 3);
  PutI32(&message, header.width);
  PutI32(&message, header.height);
  PutDouble(&message, header.fps);
  PutI32(&message, header.q_index);
  PutI32(&message, header.gop);
  PutI32(&message, header.reset_interval);
  PutI32(&message, header.ltr_start_idx);
  PutI32(&message, header.ltr_period);
  PutI32(&message, header.ltr_qp_shift);
  PutDouble(&message, header.target_bitrate_bps);
  PutU32(&message, header.flags);
  sender_.Send(message);
}

void UdpMlvcSender::SendFrame(int frame_index, mlvc::codec::MlvcFrameType frame_type, int q_index,
                              const std::vector<uint8_t>& payload) {
  Check(payload.size() <= std::numeric_limits<uint32_t>::max(), "MLVC frame payload is too large");
  std::vector<uint8_t> message;
  message.reserve(14 + payload.size());
  message.push_back(kFrameMessage);
  PutI32(&message, frame_index);
  message.push_back(static_cast<uint8_t>(frame_type));
  PutI32(&message, q_index);
  PutU32(&message, static_cast<uint32_t>(payload.size()));
  message.insert(message.end(), payload.begin(), payload.end());
  sender_.Send(message);
}

void UdpMlvcSender::SendEnd() {
  sender_.Send(std::vector<uint8_t>{kEndMessage});
  sender_.Flush();
}

UdpMlvcReceiver::UdpMlvcReceiver(uint16_t port) : receiver_(port, "MLVC") {}

UdpMlvcReceiver::~UdpMlvcReceiver() = default;

MlvcBitstreamHeader UdpMlvcReceiver::ReceiveHeader() {
  const std::vector<uint8_t> message = receiver_.Receive();
  Check(message.size() == 57 && message[0] == kHeaderMessage,
        "invalid MLVC UDP header message");
  const uint8_t* cursor = message.data() + 1;
  MlvcBitstreamHeader header;
  header.version = ReadU32(cursor);
  cursor += 4;
  header.width = ReadI32(cursor);
  cursor += 4;
  header.height = ReadI32(cursor);
  cursor += 4;
  header.fps = ReadDouble(cursor);
  cursor += 8;
  header.q_index = ReadI32(cursor);
  cursor += 4;
  header.gop = ReadI32(cursor);
  cursor += 4;
  header.reset_interval = ReadI32(cursor);
  cursor += 4;
  header.ltr_start_idx = ReadI32(cursor);
  cursor += 4;
  header.ltr_period = ReadI32(cursor);
  cursor += 4;
  header.ltr_qp_shift = ReadI32(cursor);
  cursor += 4;
  header.target_bitrate_bps = ReadDouble(cursor);
  cursor += 8;
  header.flags = ReadU32(cursor);
  return header;
}

bool UdpMlvcReceiver::ReceiveFrame(int* frame_index, mlvc::codec::MlvcFrameType* frame_type,
                                   int* q_index, std::vector<uint8_t>* payload) {
  Check(frame_index && frame_type && q_index && payload, "UDP frame outputs are required");
  const std::vector<uint8_t> message = receiver_.Receive();
  Check(!message.empty(), "empty MLVC UDP message");
  if (message[0] == kEndMessage) {
    Check(message.size() == 1, "invalid MLVC UDP end message");
    return false;
  }
  Check(message[0] == kFrameMessage && message.size() >= 14,
        "invalid MLVC UDP frame message");
  const uint8_t* cursor = message.data() + 1;
  *frame_index = ReadI32(cursor);
  cursor += 4;
  const uint8_t raw_type = *cursor++;
  Check(raw_type <= static_cast<uint8_t>(mlvc::codec::MlvcFrameType::kLtrRecovery),
        "invalid MLVC UDP frame type");
  *frame_type = static_cast<mlvc::codec::MlvcFrameType>(raw_type);
  *q_index = ReadI32(cursor);
  cursor += 4;
  const uint32_t payload_size = ReadU32(cursor);
  cursor += 4;
  Check(payload_size == message.size() - 14, "invalid MLVC UDP frame payload size");
  payload->assign(cursor, cursor + payload_size);
  return true;
}

VideoTransUdpSender::VideoTransUdpSender(const std::string& host, uint16_t port)
    : sender_(host, port, "VideoTrans forwarder") {}

VideoTransUdpSender::~VideoTransUdpSender() = default;

void VideoTransUdpSender::SendFrame(const cv::Mat& bgr, uint8_t channel) {
  Check(!bgr.empty(), "cannot forward an empty decoded frame");
  std::vector<uint8_t> jpeg;
  Check(cv::imencode(".jpg", bgr, jpeg, {cv::IMWRITE_JPEG_QUALITY, 75}),
        "failed to JPEG-encode decoded frame for forwarding");
  SendJpeg(jpeg, channel);
}

void VideoTransUdpSender::SendJpeg(const std::vector<uint8_t>& jpeg, uint8_t channel) {
  Check(jpeg.size() >= 4 && jpeg[0] == 0xff && jpeg[1] == 0xd8 &&
            jpeg[jpeg.size() - 2] == 0xff && jpeg.back() == 0xd9,
        "cannot forward an invalid JPEG payload");
  std::vector<uint8_t> payload;
  payload.reserve(8 + jpeg.size());
  PutU32(&payload, static_cast<uint32_t>(jpeg.size()));
  payload.insert(payload.end(), jpeg.begin(), jpeg.end());
  PutU32(&payload, 0);  // No detections are produced by the MLVC decoder.
  sender_.Send(payload, channel);
}

void VideoTransUdpSender::Flush() { sender_.Flush(); }

class AsyncDvppVideoTransUdpSender::Impl {
 public:
  Impl(const std::string& host, uint16_t port, int width, int height,
       std::size_t queue_capacity, uint32_t jpeg_quality, aclrtContext context)
      : layout_{width, height, AlignUp(width, 16), AlignUp(height, 2)},
        queue_capacity_(queue_capacity),
        jpeg_quality_(jpeg_quality),
        context_(context),
        sender_(host, port) {
    Check(width > 0 && height > 0 && width % 2 == 0 && height % 2 == 0,
          "DVPP async forward dimensions must be positive and even");
    Check(queue_capacity_ > 0, "DVPP async forward queue capacity must be positive");
    Check(jpeg_quality_ >= 1 && jpeg_quality_ <= 100,
          "DVPP async JPEG quality must be in [1, 100]");
    Check(context_ != nullptr, "DVPP async forward requires an ACL context");
    send_thread_ = std::thread(&Impl::SendLoop, this);
  }

  ~Impl() { CloseNoThrow(); }

  void SendFrame(const mlvc::codec::TensorData& tensor, int frame_index) {
    Check(tensor.dtype == mlvc::DataType::kFloat16,
          "DVPP async forward expects an FP16 tensor");
    std::unique_lock<std::mutex> lock(mutex_);
    queue_cv_.wait(lock, [this] {
      return stopping_ || send_error_ != nullptr || queue_.size() < queue_capacity_;
    });
    if (send_error_ != nullptr) std::rethrow_exception(send_error_);
    Check(!stopping_, "DVPP async forwarder is stopping");
    queue_.push_back(PendingFrame{mlvc::codec::CloneTensor(tensor), frame_index});
    ++stats_.enqueued;
    queue_cv_.notify_all();
  }

  void Close() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    queue_cv_.notify_all();
    if (send_thread_.joinable()) send_thread_.join();
    std::exception_ptr error;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      error = send_error_;
    }
    if (error != nullptr) std::rethrow_exception(error);
  }

  DvppForwardStats stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
  }

 private:
  struct PendingFrame {
    mlvc::codec::TensorData tensor;
    int frame_index = 0;
  };

  static int AlignUp(int value, int alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
  }

  void CloseNoThrow() noexcept {
    try {
      Close();
    } catch (...) {
    }
  }

  void SendLoop() {
    try {
      CheckAcl(aclrtSetCurrentContext(context_),
               "aclrtSetCurrentContext for DVPP async forward");
      DvppJpegEncoder encoder(context_, layout_, jpeg_quality_);
      for (;;) {
        PendingFrame pending;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          queue_cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
          if (queue_.empty()) break;
          pending = std::move(queue_.front());
          queue_.pop_front();
          queue_cv_.notify_all();
        }

        const auto conversion_begin = std::chrono::steady_clock::now();
        std::vector<uint8_t> nv12;
        ConvertFp16Yuv444ToNv12(pending.tensor, layout_, &nv12);
        const auto conversion_end = std::chrono::steady_clock::now();
        DvppJpegTiming timing;
        std::vector<uint8_t> jpeg = encoder.Encode(nv12, &timing);
        const auto udp_begin = std::chrono::steady_clock::now();
        sender_.SendJpeg(jpeg);
        const auto udp_end = std::chrono::steady_clock::now();
        {
          std::lock_guard<std::mutex> lock(mutex_);
          ++stats_.encoded;
          ++stats_.sent;
          stats_.conversion_ms +=
              std::chrono::duration<double, std::milli>(conversion_end - conversion_begin).count();
          stats_.h2d_ms += timing.h2d_ms;
          stats_.encode_ms += timing.encode_ms;
          stats_.d2h_ms += timing.d2h_ms;
          stats_.udp_ms +=
              std::chrono::duration<double, std::milli>(udp_end - udp_begin).count();
        }
      }
      sender_.Flush();
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      send_error_ = std::current_exception();
      ++stats_.failed;
      stopping_ = true;
      queue_cv_.notify_all();
    }
  }

  Nv12Layout layout_;
  std::size_t queue_capacity_ = 0;
  uint32_t jpeg_quality_ = 75;
  aclrtContext context_ = nullptr;
  VideoTransUdpSender sender_;
  mutable std::mutex mutex_;
  std::condition_variable queue_cv_;
  std::deque<PendingFrame> queue_;
  std::exception_ptr send_error_;
  bool stopping_ = false;
  DvppForwardStats stats_;
  std::thread send_thread_;
};

AsyncDvppVideoTransUdpSender::AsyncDvppVideoTransUdpSender(
    const std::string& host, uint16_t port, int width, int height,
    std::size_t queue_capacity, uint32_t jpeg_quality, aclrtContext context)
    : impl_(std::make_unique<Impl>(host, port, width, height, queue_capacity,
                                   jpeg_quality, context)) {}

AsyncDvppVideoTransUdpSender::~AsyncDvppVideoTransUdpSender() = default;

void AsyncDvppVideoTransUdpSender::SendFrame(const mlvc::codec::TensorData& tensor,
                                              int frame_index) {
  impl_->SendFrame(tensor, frame_index);
}

void AsyncDvppVideoTransUdpSender::Close() { impl_->Close(); }

DvppForwardStats AsyncDvppVideoTransUdpSender::stats() const { return impl_->stats(); }

class AsyncDeviceDvppVideoTransUdpSender::Impl {
 public:
  Impl(const std::string& host, uint16_t port, int width, int height,
       std::size_t queue_capacity, uint32_t jpeg_quality, aclrtContext context)
      : layout_{width, height, AlignUp(width, 32), AlignUp(height, 2)},
        input_bytes_(Nv12BufferSize(layout_)),
        jpeg_quality_(jpeg_quality),
        context_(context),
        sender_(host, port),
        queue_(queue_capacity),
        slots_(queue_capacity) {
    Check(width > 0 && height > 0 && width % 16 == 0 && height % 2 == 0,
          "device DVPP dimensions require width aligned to 16 and even height");
    Check(queue_capacity >= 2,
          "device DVPP queue capacity must be at least two");
    Check(jpeg_quality_ >= 1 && jpeg_quality_ <= 100,
          "device DVPP JPEG quality must be in [1, 100]");
    Check(context_ != nullptr, "device DVPP forward requires an ACL context");
    Check(Fp16Yuv444ToNv12AclAvailable(),
          "device DVPP forward requires the MlvcFp16Yuv444ToNv12 custom op");
    try {
      CheckAcl(aclrtSetCurrentContext(context_),
               "aclrtSetCurrentContext for device DVPP slots");
      for (Slot& slot : slots_) {
        CheckAcl(acldvppMalloc(&slot.nv12, input_bytes_),
                 "acldvppMalloc device DVPP slot");
        constexpr uint32_t kTimedSyncEvent = ACL_EVENT_SYNC | ACL_EVENT_TIME_LINE;
        CheckAcl(aclrtCreateEventWithFlag(&slot.conversion_start, kTimedSyncEvent),
                 "aclrtCreateEventWithFlag device conversion start");
        CheckAcl(aclrtCreateEventWithFlag(&slot.ready, kTimedSyncEvent),
                 "aclrtCreateEventWithFlag device conversion ready");
      }
      send_thread_ = std::thread(&Impl::SendLoop, this);
    } catch (...) {
      CleanupSlots();
      throw;
    }
  }

  ~Impl() {
    CloseNoThrow();
    CleanupSlots();
  }

  void SendFrame(const void* input_fp16_device,
                 const mlvc::TensorShape& input_shape,
                 aclrtStream decoder_stream, int frame_index,
                 mlvc::Profiler* profiler) {
    Check(input_fp16_device != nullptr && decoder_stream != nullptr,
          "device DVPP frame requires device input and decoder stream");
    const std::size_t slot_index = queue_.Acquire();
    try {
      Slot& slot = slots_.at(slot_index);
      Fp16Yuv444ToNv12Acl(input_fp16_device, input_shape, layout_, slot.nv12,
                          decoder_stream, slot.conversion_start, slot.ready,
                          profiler);
      queue_.Enqueue(slot_index, frame_index);
      std::lock_guard<std::mutex> lock(stats_mutex_);
      ++stats_.enqueued;
    } catch (...) {
      queue_.Release(slot_index);
      throw;
    }
  }

  void Close() {
    bool expected = false;
    if (closed_.compare_exchange_strong(expected, true)) {
      queue_.Shutdown();
      if (send_thread_.joinable()) send_thread_.join();
    }
    std::exception_ptr error;
    {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      error = send_error_;
    }
    if (error != nullptr) std::rethrow_exception(error);
  }

  DvppForwardStats stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
  }

 private:
  struct Slot {
    void* nv12 = nullptr;
    aclrtEvent conversion_start = nullptr;
    aclrtEvent ready = nullptr;
  };

  static int AlignUp(int value, int alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
  }

  void CloseNoThrow() noexcept {
    try {
      Close();
    } catch (...) {
    }
  }

  void CleanupSlots() noexcept {
    if (context_ != nullptr) (void)aclrtSetCurrentContext(context_);
    for (Slot& slot : slots_) {
      if (slot.ready != nullptr) {
        (void)aclrtDestroyEvent(slot.ready);
        slot.ready = nullptr;
      }
      if (slot.conversion_start != nullptr) {
        (void)aclrtDestroyEvent(slot.conversion_start);
        slot.conversion_start = nullptr;
      }
      if (slot.nv12 != nullptr) {
        (void)acldvppFree(slot.nv12);
        slot.nv12 = nullptr;
      }
    }
  }

  void SendLoop() {
    try {
      CheckAcl(aclrtSetCurrentContext(context_),
               "aclrtSetCurrentContext for device DVPP forward");
      DvppJpegEncoder encoder(context_, layout_, jpeg_quality_);
      detail::DeviceDvppPendingSlot pending;
      while (queue_.Pop(&pending)) {
        Slot& slot = slots_.at(pending.slot);
        CheckAcl(aclrtSynchronizeEvent(slot.ready),
                 "aclrtSynchronizeEvent device NV12 conversion");
        float conversion_ms = 0.0F;
        CheckAcl(aclrtEventElapsedTime(&conversion_ms, slot.conversion_start,
                                       slot.ready),
                 "aclrtEventElapsedTime device NV12 conversion");
        DvppJpegTiming timing;
        std::vector<uint8_t> jpeg =
            encoder.EncodeDevice(slot.nv12, input_bytes_, slot.ready, &timing);
        const auto udp_begin = std::chrono::steady_clock::now();
        sender_.SendJpeg(jpeg);
        const auto udp_end = std::chrono::steady_clock::now();
        queue_.Release(pending.slot);
        {
          std::lock_guard<std::mutex> lock(stats_mutex_);
          ++stats_.encoded;
          ++stats_.sent;
          stats_.conversion_ms += conversion_ms;
          stats_.h2d_ms += timing.h2d_ms;
          stats_.encode_ms += timing.encode_ms;
          stats_.d2h_ms += timing.d2h_ms;
          stats_.udp_ms +=
              std::chrono::duration<double, std::milli>(udp_end - udp_begin).count();
        }
      }
      sender_.Flush();
    } catch (...) {
      const std::exception_ptr error = std::current_exception();
      {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        send_error_ = error;
        ++stats_.failed;
      }
      queue_.Fail(error);
    }
  }

  Nv12Layout layout_;
  std::size_t input_bytes_ = 0;
  uint32_t jpeg_quality_ = 75;
  aclrtContext context_ = nullptr;
  VideoTransUdpSender sender_;
  detail::DeviceDvppSlotQueue queue_;
  std::vector<Slot> slots_;
  mutable std::mutex stats_mutex_;
  std::exception_ptr send_error_;
  DvppForwardStats stats_;
  std::atomic<bool> closed_{false};
  std::thread send_thread_;
};

AsyncDeviceDvppVideoTransUdpSender::AsyncDeviceDvppVideoTransUdpSender(
    const std::string& host, uint16_t port, int width, int height,
    std::size_t queue_capacity, uint32_t jpeg_quality, aclrtContext context)
    : impl_(std::make_unique<Impl>(host, port, width, height, queue_capacity,
                                   jpeg_quality, context)) {}

AsyncDeviceDvppVideoTransUdpSender::~AsyncDeviceDvppVideoTransUdpSender() = default;

void AsyncDeviceDvppVideoTransUdpSender::SendFrame(
    const void* input_fp16_device, const mlvc::TensorShape& input_shape,
    aclrtStream decoder_stream, int frame_index, mlvc::Profiler* profiler) {
  impl_->SendFrame(input_fp16_device, input_shape, decoder_stream, frame_index,
                   profiler);
}

void AsyncDeviceDvppVideoTransUdpSender::Close() { impl_->Close(); }

DvppForwardStats AsyncDeviceDvppVideoTransUdpSender::stats() const {
  return impl_->stats();
}

AsyncVideoTransUdpSender::AsyncVideoTransUdpSender(const std::string& host, uint16_t port,
                                                   int width, int height)
    : width_(width), height_(height), sender_(host, port) {
  Check(width_ > 0 && height_ > 0, "invalid async JPEG forward dimensions");
  send_thread_ = std::thread(&AsyncVideoTransUdpSender::SendLoop, this);
}

AsyncVideoTransUdpSender::~AsyncVideoTransUdpSender() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    stopping_ = true;
  }
  queue_cv_.notify_all();
  if (send_thread_.joinable()) {
    send_thread_.join();
  }
}

void AsyncVideoTransUdpSender::SendFrame(const mlvc::codec::TensorData& tensor, int frame_index) {
  Check(tensor.dtype == mlvc::DataType::kFloat16, "async JPEG forward expects an FP16 tensor");
  std::lock_guard<std::mutex> lock(queue_mutex_);
  if (send_error_ != nullptr) {
    std::rethrow_exception(send_error_);
  }
  Check(!stopping_, "async JPEG forwarder is stopping");
  constexpr std::size_t kMaxQueuedFrames = 4;
  if (send_queue_.size() >= kMaxQueuedFrames) {
    ++dropped_frames_;
    return;
  }
  send_queue_.push_back(PendingFrame{mlvc::codec::CloneTensor(tensor), frame_index});
  queue_cv_.notify_one();
}

void AsyncVideoTransUdpSender::SendLoop() {
  for (;;) {
    PendingFrame pending;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock, [this] { return stopping_ || !send_queue_.empty(); });
      if (send_queue_.empty()) {
        return;
      }
      pending = std::move(send_queue_.front());
      send_queue_.pop_front();
    }
    try {
      const cv::Mat bgr = ConvertTensorToBgr(pending.tensor, width_, height_);
      sender_.SendFrame(bgr);
    } catch (...) {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      send_error_ = std::current_exception();
      stopping_ = true;
      queue_cv_.notify_all();
      return;
    }
  }
}

RawYuvUdpSender::RawYuvUdpSender(const std::string& host, uint16_t port)
    : sender_(host, port, "raw YUV forwarder") {
  send_thread_ = std::thread(&RawYuvUdpSender::SendLoop, this);
}

RawYuvUdpSender::~RawYuvUdpSender() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    stopping_ = true;
  }
  queue_cv_.notify_all();
  if (send_thread_.joinable()) {
    send_thread_.join();
  }
}

void RawYuvUdpSender::SendFrame(const mlvc::codec::TensorData& tensor, int frame_index) {
  Check(tensor.dtype == mlvc::DataType::kFloat16, "raw YUV forward expects FP16 tensor");
  Check(tensor.shape.rank() == 4 && tensor.shape.dim(0) == 1 && tensor.shape.dim(1) == 3,
        "raw YUV forward expects NCHW 3-plane tensor");
  Check(tensor.bytes.size() <= std::numeric_limits<uint32_t>::max(),
        "raw YUV tensor is too large");
  std::vector<uint8_t> message;
  message.reserve(26 + tensor.bytes.size());
  message.push_back(kRawYuvMessage);
  PutI32(&message, frame_index);
  PutU32(&message, static_cast<uint32_t>(tensor.shape.dim(3)));
  PutU32(&message, static_cast<uint32_t>(tensor.shape.dim(2)));
  PutU32(&message, static_cast<uint32_t>(tensor.shape.dim(3)));
  PutU32(&message, static_cast<uint32_t>(tensor.shape.dim(2)));
  message.push_back(1);  // FP16 YUV444 NCHW.
  PutU32(&message, static_cast<uint32_t>(tensor.bytes.size()));
  message.insert(message.end(), tensor.bytes.begin(), tensor.bytes.end());
  std::lock_guard<std::mutex> lock(queue_mutex_);
  if (send_error_ != nullptr) {
    std::rethrow_exception(send_error_);
  }
  Check(!stopping_, "raw YUV forwarder is stopping");
  constexpr std::size_t kMaxQueuedFrames = 4;
  if (send_queue_.size() >= kMaxQueuedFrames) {
    ++dropped_frames_;
    return;
  }
  send_queue_.push_back(PendingFrame{std::move(message)});
  queue_cv_.notify_one();
}

void RawYuvUdpSender::SendLoop() {
  for (;;) {
    PendingFrame pending;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock, [this] { return stopping_ || !send_queue_.empty(); });
      if (send_queue_.empty()) {
        return;
      }
      pending = std::move(send_queue_.front());
      send_queue_.pop_front();
    }
    try {
      sender_.Send(pending.message);
    } catch (...) {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      send_error_ = std::current_exception();
      stopping_ = true;
      queue_cv_.notify_all();
      return;
    }
  }
}

}  // namespace mlvc::io
