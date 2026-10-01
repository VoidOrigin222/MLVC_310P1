#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <exception>
#include <deque>
#include <mutex>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "mlvc/transport/rtp_mlvc.h"
#include "mlvc/transport/rtcp_session.h"

namespace mlvc::transport {

class RtpMessageReceiver {
 public:
  explicit RtpMessageReceiver(uint16_t port, std::string context = "RTP receiver",
                              uint8_t payload_type = 96);
  ~RtpMessageReceiver();

  RtpMessageReceiver(const RtpMessageReceiver&) = delete;
  RtpMessageReceiver& operator=(const RtpMessageReceiver&) = delete;

  bool Receive(std::vector<uint8_t>* unit);
  // Waits at most timeout for a complete media unit. Returns false only when
  // the deadline expires or the receiver is stopping with an empty queue.
  bool ReceiveFor(std::vector<uint8_t>* unit, std::chrono::milliseconds timeout);
  uint16_t local_port() const { return local_port_; }

  uint64_t lost_packets() const { return lost_packets_; }
  uint64_t duplicate_packets() const { return duplicate_packets_; }
  uint64_t reordered_packets() const { return reordered_packets_; }
  uint64_t rtcp_sender_reports_received() const { return rtcp_sender_reports_received_; }
  uint64_t rtcp_receiver_reports_sent() const { return rtcp_receiver_reports_sent_; }

 private:
  void ReceiveLoop();
  bool TrackSequence(uint16_t sequence,
                     std::chrono::steady_clock::time_point arrival_time);
  void SetReceiveError(std::exception_ptr error);
  void RtcpLoop();
  void UpdateRtcpReceiveStats(const RtpPacket& packet);
  RtcpReportBlock SnapshotReportBlock(uint32_t source_ssrc);
  std::vector<RtcpNack> SnapshotPendingNacks(uint32_t sender_ssrc,
                                              uint32_t media_ssrc,
                                              std::chrono::steady_clock::time_point now);

  int socket_ = -1;
  uint16_t local_port_ = 0;
  std::string context_;
  uint8_t payload_type_ = 96;
  bool have_sequence_ = false;
  uint16_t highest_sequence_ = 0;
  std::atomic<uint32_t> ssrc_{0};
  std::atomic<uint64_t> lost_packets_{0};
  std::atomic<uint64_t> duplicate_packets_{0};
  std::atomic<uint64_t> reordered_packets_{0};
  std::set<uint16_t> seen_sequences_;
  std::deque<uint16_t> seen_sequence_order_;
  std::set<uint16_t> missing_sequences_;
  std::deque<uint16_t> missing_sequence_order_;
  std::map<uint16_t, std::chrono::steady_clock::time_point> missing_sequence_times_;
  mutable std::mutex sequence_mutex_;
  RtpMlvcReassembler reassembler_;
  std::deque<std::pair<uint32_t, uint32_t>> config_frame_limits_;
  static constexpr std::size_t kMaxQueuedUnits = 64;
  static constexpr std::size_t kMaxQueuedUnitBytes = 128u * 1024u * 1024u;
  std::mutex queue_mutex_;
  std::condition_variable queue_not_empty_;
  std::condition_variable queue_not_full_;
  std::deque<std::vector<uint8_t>> unit_queue_;
  std::size_t unit_queue_bytes_ = 0;
  std::exception_ptr receive_error_;
  bool stopping_ = false;
  std::thread receive_thread_;
  std::unique_ptr<RtcpUdpEndpoint> rtcp_endpoint_;
  std::atomic<bool> rtcp_stopping_{false};
  std::thread rtcp_thread_;
  std::atomic<uint64_t> rtcp_sender_reports_received_{0};
  std::atomic<uint64_t> rtcp_receiver_reports_sent_{0};
  std::mutex rtcp_stats_mutex_;
  bool rtcp_have_sequence_ = false;
  uint16_t rtcp_max_sequence_ = 0;
  uint64_t rtcp_sequence_cycles_ = 0;
  uint64_t rtcp_base_sequence_ = 0;
  uint64_t rtcp_highest_sequence_ = 0;
  uint64_t rtcp_received_unique_ = 0;
  uint64_t rtcp_prior_expected_ = 0;
  uint64_t rtcp_prior_received_ = 0;
  bool rtcp_have_transit_ = false;
  uint32_t rtcp_prior_transit_ = 0;
  double rtcp_jitter_ = 0.0;
  uint32_t rtcp_last_sender_report_ = 0;
  std::chrono::steady_clock::time_point rtcp_last_sender_report_at_{};
};

}  // namespace mlvc::transport
