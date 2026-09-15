#ifndef MLVC_IO_UDP_FRAME_TRANSPORT_H_
#define MLVC_IO_UDP_FRAME_TRANSPORT_H_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "mlvc/codec/tensor_data.h"
#include "mlvc/io/mlvc_bitstream.h"
#include "mlvc/transport/rtp_message_receiver.h"
#include "mlvc/transport/rtp_mlvc.h"
#include "mlvc/transport/udp_message_transport.h"

namespace mlvc {
class Profiler;
}

namespace mlvc::io {

// UDP framing follows VideoTrans: EB90 magic, 1024-byte fragments, and CDDE
// tail. A reassembled message starts with a one-byte MLVC message kind.
class UdpMlvcSender {
 public:
  UdpMlvcSender(const std::string& host, uint16_t port,
                mlvc::transport::UdpSendOptions options = {});
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
  MlvcBitstreamHeader header_;
  uint64_t max_payload_size_ = 0;
  int expected_frame_index_ = 0;
  bool header_received_ = false;
};

class RtpMlvcSender {
 public:
  RtpMlvcSender(const std::string& host, uint16_t port, uint64_t pacing_rate_bps = 0,
                std::size_t max_burst_bytes = 4096);
  ~RtpMlvcSender();
  void SendHeader(const MlvcBitstreamHeader& header);
  void SendFrame(int frame_index, mlvc::codec::MlvcFrameType frame_type, int q_index,
                 const std::vector<uint8_t>& payload);
  void SendEnd();

 private:
  mlvc::transport::RtpMlvcSender sender_;
};

class RtpMlvcReceiver {
 public:
  explicit RtpMlvcReceiver(uint16_t port);
  ~RtpMlvcReceiver();
  MlvcBitstreamHeader ReceiveHeader();
  bool ReceiveFrame(int* frame_index, mlvc::codec::MlvcFrameType* frame_type, int* q_index,
                    std::vector<uint8_t>* payload);

 private:
  mlvc::transport::RtpMessageReceiver receiver_;
  MlvcBitstreamHeader header_;
  uint64_t max_payload_size_ = 0;
  int expected_frame_index_ = 0;
  bool header_received_ = false;
};

// Raw forwarding path for local-performance measurements. The sender copies
// the decoded FP16 YUV tensor into a queue and emits a framed message.
class RawYuvUdpSender {
 public:
  RawYuvUdpSender(const std::string& host, uint16_t port);
  ~RawYuvUdpSender();
  void SendFrame(const mlvc::codec::TensorData& tensor, int frame_index);
  void Close();
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
  bool closed_ = false;
  std::atomic<uint64_t> dropped_frames_{0};
  std::thread send_thread_;
};

}  // namespace mlvc::io

#endif  // MLVC_IO_UDP_FRAME_TRANSPORT_H_
