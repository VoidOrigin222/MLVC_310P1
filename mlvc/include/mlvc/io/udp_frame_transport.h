#ifndef MLVC_IO_UDP_FRAME_TRANSPORT_H_
#define MLVC_IO_UDP_FRAME_TRANSPORT_H_

#include <atomic>
#include <acl/acl.h>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <string>
#include <thread>
#include <vector>

#include "mlvc/codec/tensor_data.h"
#include "mlvc/io/mlvc_bitstream.h"
#include "mlvc/transport/udp_message_transport.h"

namespace mlvc {
class Profiler;
}

namespace mlvc::io {

namespace detail {

struct DeviceDvppPendingSlot {
  std::size_t slot = 0;
  int frame_index = 0;
};

class DeviceDvppSlotQueue {
 public:
  explicit DeviceDvppSlotQueue(std::size_t capacity);
  ~DeviceDvppSlotQueue();

  DeviceDvppSlotQueue(const DeviceDvppSlotQueue&) = delete;
  DeviceDvppSlotQueue& operator=(const DeviceDvppSlotQueue&) = delete;

  std::size_t Acquire();
  void Enqueue(std::size_t slot, int frame_index);
  bool Pop(DeviceDvppPendingSlot* pending);
  void Release(std::size_t slot);
  void Fail(std::exception_ptr error);
  void Shutdown();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace detail

// UDP framing follows VideoTrans: EB90 magic, 1024-byte fragments, and CDDE
// tail. A reassembled message starts with a one-byte MLVC message kind.
class UdpMlvcSender {
 public:
  UdpMlvcSender(const std::string& host, uint16_t port);
  ~UdpMlvcSender();
  void SendHeader(const MlvcBitstreamHeader& header);
  void SendFrame(int frame_index, mlvc::codec::MlvcFrameType frame_type, int q_index,
                 const std::vector<uint8_t>& payload);
 void SendEnd();

 private:
  mlvc::transport::UdpMessageSender sender_;
};

class UdpMlvcReceiver {
 public:
  explicit UdpMlvcReceiver(uint16_t port);
  ~UdpMlvcReceiver();
  MlvcBitstreamHeader ReceiveHeader();
 bool ReceiveFrame(int* frame_index, mlvc::codec::MlvcFrameType* frame_type, int* q_index,
                    std::vector<uint8_t>* payload);

 private:
  mlvc::transport::UdpMessageReceiver receiver_;
};

// VideoTrans-compatible visualization forwarder. It sends a JPEG payload and
// an empty bounding-box list using the same 1024-byte fragment layout.
class VideoTransUdpSender {
 public:
  VideoTransUdpSender(const std::string& host, uint16_t port);
  ~VideoTransUdpSender();
  void SendFrame(const cv::Mat& bgr, uint8_t channel = 0);
  void SendJpeg(const std::vector<uint8_t>& jpeg, uint8_t channel = 0);
  void Flush();

 private:
  mlvc::transport::UdpMessageSender sender_;
};

struct DvppForwardStats {
  uint64_t enqueued = 0;
  uint64_t encoded = 0;
  uint64_t sent = 0;
  uint64_t failed = 0;
  uint64_t dropped = 0;
  double conversion_ms = 0.0;
  double h2d_ms = 0.0;
  double encode_ms = 0.0;
  double d2h_ms = 0.0;
  double udp_ms = 0.0;
};

class AsyncDvppVideoTransUdpSender {
 public:
  AsyncDvppVideoTransUdpSender(const std::string& host, uint16_t port, int width, int height,
                               std::size_t queue_capacity, uint32_t jpeg_quality,
                               aclrtContext context);
  ~AsyncDvppVideoTransUdpSender();
  void SendFrame(const mlvc::codec::TensorData& tensor, int frame_index);
  void Close();
  DvppForwardStats stats() const;
  uint64_t dropped_frames() const { return stats().dropped; }

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class AsyncDeviceDvppVideoTransUdpSender {
 public:
  AsyncDeviceDvppVideoTransUdpSender(const std::string& host, uint16_t port,
                                     int width, int height,
                                     std::size_t queue_capacity,
                                     uint32_t jpeg_quality,
                                     aclrtContext context);
  ~AsyncDeviceDvppVideoTransUdpSender();
  void SendFrame(const void* input_fp16_device, const mlvc::TensorShape& input_shape,
                 aclrtStream decoder_stream, int frame_index,
                 mlvc::Profiler* profiler = nullptr);
  void Close();
  DvppForwardStats stats() const;
  uint64_t dropped_frames() const { return stats().dropped; }

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// JPEG forwarder that also performs FP16 YUV conversion and JPEG encoding on
// a worker thread. The decoder thread only clones the decoded tensor into a
// bounded queue.
class AsyncVideoTransUdpSender {
 public:
  AsyncVideoTransUdpSender(const std::string& host, uint16_t port, int width, int height);
  ~AsyncVideoTransUdpSender();
  void SendFrame(const mlvc::codec::TensorData& tensor, int frame_index);
  uint64_t dropped_frames() const { return dropped_frames_.load(); }

 private:
  struct PendingFrame {
    mlvc::codec::TensorData tensor;
    int frame_index = 0;
  };
  void SendLoop();

  int width_ = 0;
  int height_ = 0;
  VideoTransUdpSender sender_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::deque<PendingFrame> send_queue_;
  std::exception_ptr send_error_;
  bool stopping_ = false;
  std::atomic<uint64_t> dropped_frames_{0};
  std::thread send_thread_;
};

// Raw forwarding path for local-performance measurements. The sender copies
// the decoded FP16 YUV tensor into a queue; the worker emits a framed message
// without RGB conversion or JPEG encoding.
class RawYuvUdpSender {
 public:
  RawYuvUdpSender(const std::string& host, uint16_t port);
  ~RawYuvUdpSender();
  void SendFrame(const mlvc::codec::TensorData& tensor, int frame_index);
 uint64_t dropped_frames() const { return dropped_frames_.load(); }

 private:
  struct PendingFrame {
    std::vector<uint8_t> message;
  };
  void SendLoop();

  mlvc::transport::UdpMessageSender sender_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::deque<PendingFrame> send_queue_;
  std::exception_ptr send_error_;
  bool stopping_ = false;
  std::atomic<uint64_t> dropped_frames_{0};
  std::thread send_thread_;
};

}  // namespace mlvc::io

#endif  // MLVC_IO_UDP_FRAME_TRANSPORT_H_
