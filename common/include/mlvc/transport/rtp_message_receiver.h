#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "mlvc/transport/rtp_mlvc.h"

namespace mlvc::transport {

class RtpMessageReceiver {
 public:
  explicit RtpMessageReceiver(uint16_t port, std::string context = "RTP receiver");
  ~RtpMessageReceiver();

  RtpMessageReceiver(const RtpMessageReceiver&) = delete;
  RtpMessageReceiver& operator=(const RtpMessageReceiver&) = delete;

  bool Receive(std::vector<uint8_t>* unit);
  uint16_t local_port() const { return local_port_; }

  uint64_t lost_packets() const { return lost_packets_; }
  uint64_t duplicate_packets() const { return duplicate_packets_; }
  uint64_t reordered_packets() const { return reordered_packets_; }

 private:
  void ReceiveLoop();
  void TrackSequence(uint16_t sequence);
  void SetReceiveError(std::exception_ptr error);

  int socket_ = -1;
  uint16_t local_port_ = 0;
  std::string context_;
  bool have_sequence_ = false;
  uint16_t highest_sequence_ = 0;
  uint32_t ssrc_ = 0;
  std::atomic<uint64_t> lost_packets_{0};
  std::atomic<uint64_t> duplicate_packets_{0};
  std::atomic<uint64_t> reordered_packets_{0};
  std::set<uint16_t> seen_sequences_;
  std::deque<uint16_t> seen_sequence_order_;
  RtpMlvcReassembler reassembler_;
  static constexpr std::size_t kMaxQueuedUnits = 64;
  std::mutex queue_mutex_;
  std::condition_variable queue_not_empty_;
  std::condition_variable queue_not_full_;
  std::deque<std::vector<uint8_t>> unit_queue_;
  std::exception_ptr receive_error_;
  bool stopping_ = false;
  std::thread receive_thread_;
};

}  // namespace mlvc::transport
