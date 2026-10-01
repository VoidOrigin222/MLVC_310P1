#ifndef MLVC_TRANSPORT_RTP_MLVC_H_
#define MLVC_TRANSPORT_RTP_MLVC_H_
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include "mlvc/transport/rtcp_session.h"
#include <chrono>

#include "mlvc/transport/udp_pacer.h"
#include "mlvc/transport/mlvc_media_unit.h"
namespace mlvc::transport {
struct RtpTransportStats {
  uint64_t wire_bytes = 0;
  uint64_t packets = 0;
  uint64_t max_burst_bytes = 0;
  uint64_t max_queue_delay_us = 0;
  uint64_t average_queue_delay_us = 0;
  uint64_t socket_block_us = 0;
  uint64_t rtcp_packets_sent = 0;
  uint64_t rtcp_packets_received = 0;
};
using RtpUnitType = MlvcMediaUnitType;
constexpr uint8_t kRtpMlvcStart = 0x80;
constexpr uint8_t kRtpMlvcEnd = 0x40;
constexpr uint8_t kRtpMlvcExtension = 0x20;
struct RtpMlvcFragment {
  RtpUnitType unit_type;
  uint8_t flags;
  uint32_t config_id;
  uint32_t unit_id;
  uint32_t fragment_offset;
  uint32_t unit_length;
  std::vector<uint8_t> payload;
  // Filled by RtpMessageReceiver from the RTP header.  They are deliberately
  // not serialized in the MLVC descriptor, but are part of the reassembly key.
  uint32_t timestamp = 0;
  uint32_t ssrc = 0;
};
std::vector<uint8_t> EncodeRtpMlvcFragment(const RtpMlvcFragment& f);
RtpMlvcFragment DecodeRtpMlvcFragment(const std::vector<uint8_t>& bytes);
class RtpMlvcReassembler {
 public:
  explicit RtpMlvcReassembler(std::size_t max_unit_bytes = 128u * 1024u * 1024u,
                              std::size_t max_pending_units = 64,
                              std::chrono::milliseconds fragment_timeout = std::chrono::seconds(2));
  bool Push(const RtpMlvcFragment& fragment, std::vector<uint8_t>* unit);
  void Reset();

 private:
  struct Pending {
    RtpUnitType type;
    uint32_t length;
    std::vector<uint8_t> data;
    std::map<uint32_t, uint32_t> fragment_ranges;
    std::map<uint32_t, std::vector<uint8_t>> pre_start_fragments;
    std::size_t count = 0;
    std::size_t pre_start_bytes = 0;
    std::chrono::steady_clock::time_point last_updated;
  };
  std::size_t max_unit_bytes_;
  std::size_t max_pending_units_;
  std::size_t max_completed_units_;
  std::size_t max_pending_bytes_;
  std::size_t pending_bytes_ = 0;
  std::size_t completed_bytes_ = 0;
  std::size_t fragment_range_count_ = 0;
  std::size_t pre_start_fragment_bytes_ = 0;
  std::chrono::milliseconds fragment_timeout_;
  std::chrono::steady_clock::time_point last_cleanup_;
  using UnitKey = std::tuple<uint32_t, uint32_t, uint8_t, uint32_t, uint32_t>;
  std::map<UnitKey, Pending> pending_;
  std::map<UnitKey, std::vector<uint8_t>> completed_;
};
class RtpMlvcSender {
 public:
  RtpMlvcSender(const std::string& host, uint16_t port, uint32_t ssrc = 0,
                uint64_t pacing_rate_bps = 0, std::size_t max_burst_bytes = 4096,
                std::size_t max_queue_bytes = 4u * 1024u * 1024u,
                uint64_t max_queue_delay_ms = 1000, uint8_t payload_type = 96,
                std::chrono::milliseconds rtcp_interval = std::chrono::seconds(5));
  ~RtpMlvcSender();
  RtpMlvcSender(const RtpMlvcSender&) = delete;
  RtpMlvcSender& operator=(const RtpMlvcSender&) = delete;
  void SendUnit(RtpUnitType type, uint8_t flags, uint32_t config_id, uint32_t unit_id,
                uint32_t timestamp, const std::vector<uint8_t>& unit);
  void SetSessionConfig(uint32_t config_id, const std::vector<uint8_t>& config_unit,
                        uint8_t flags = 0, uint32_t timestamp = 0);
  void ResendSessionConfig(uint32_t timestamp);
  void Flush();
  void Close();
  bool ConsumeRandomAccessRequest();
  bool PopMlvcControl(MlvcControlMessage* message);
  RtpTransportStats Stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct RtpPacket {
  uint16_t sequence = 0;
  uint32_t timestamp = 0;
  uint32_t ssrc = 0;
  bool marker = false;
  std::vector<uint8_t> payload;
};
// The default dynamic payload type is 96.  Applications that negotiate a
// different dynamic PT through SDP pass it explicitly instead of treating 96
// as a wire-level protocol constant.
RtpPacket DecodeRtpPacket(const std::vector<uint8_t>& bytes,
                          uint8_t expected_payload_type = 96);
}  // namespace mlvc::transport
#endif
