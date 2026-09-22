#ifndef MLVC_TRANSPORT_RTP_MLVC_H_
#define MLVC_TRANSPORT_RTP_MLVC_H_
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <chrono>

#include "mlvc/transport/udp_pacer.h"
namespace mlvc::transport {
struct RtpTransportStats {
  uint64_t wire_bytes = 0;
  uint64_t packets = 0;
  uint64_t max_burst_bytes = 0;
  uint64_t max_queue_delay_us = 0;
  uint64_t average_queue_delay_us = 0;
  uint64_t socket_block_us = 0;
};
enum class RtpUnitType : uint8_t { kScu = 1, kEfu = 2, kEos = 3 };
struct RtpMlvcFragment {
  RtpUnitType unit_type;
  uint8_t flags;
  uint32_t config_id;
  uint32_t unit_id;
  uint32_t fragment_offset;
  uint32_t unit_length;
  std::vector<uint8_t> payload;
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
    uint8_t flags;
    uint32_t length;
    std::vector<uint8_t> data;
    std::vector<uint8_t> seen;
    std::size_t count = 0;
  };
  std::size_t max_unit_bytes_;
  std::size_t max_pending_units_;
  std::chrono::milliseconds fragment_timeout_;
  std::chrono::steady_clock::time_point last_cleanup_;
  std::map<std::pair<uint32_t, uint32_t>, Pending> pending_;
};
class RtpMlvcSender {
 public:
  RtpMlvcSender(const std::string& host, uint16_t port, uint32_t ssrc = 0x4d4c5643u,
                uint64_t pacing_rate_bps = 0, std::size_t max_burst_bytes = 4096,
                std::size_t max_queue_bytes = 4u * 1024u * 1024u,
                uint64_t max_queue_delay_ms = 1000);
  ~RtpMlvcSender();
  RtpMlvcSender(const RtpMlvcSender&) = delete;
  RtpMlvcSender& operator=(const RtpMlvcSender&) = delete;
  void SendUnit(RtpUnitType type, uint8_t flags, uint32_t config_id, uint32_t unit_id,
                uint32_t timestamp, const std::vector<uint8_t>& unit);
  void SetSessionConfig(uint32_t config_id, const std::vector<uint8_t>& config_unit,
                        uint8_t flags = 0);
  void ResendSessionConfig(uint32_t timestamp);
  void Flush();
  void Close();
  RtpTransportStats Stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct RtcpConfigRequest {
  uint32_t config_id = 0;
  uint32_t ssrc = 0;
};
std::vector<uint8_t> EncodeRtcpConfigRequest(const RtcpConfigRequest& request);
RtcpConfigRequest DecodeRtcpConfigRequest(const std::vector<uint8_t>& packet);
struct RtpPacket {
  uint16_t sequence = 0;
  uint32_t timestamp = 0;
  uint32_t ssrc = 0;
  std::vector<uint8_t> payload;
};
RtpPacket DecodeRtpPacket(const std::vector<uint8_t>& bytes);
}  // namespace mlvc::transport
#endif
