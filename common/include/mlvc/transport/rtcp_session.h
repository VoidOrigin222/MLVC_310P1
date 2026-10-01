#ifndef MLVC_TRANSPORT_RTCP_SESSION_H_
#define MLVC_TRANSPORT_RTCP_SESSION_H_

#include <cstdint>
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mlvc::transport {

struct RtcpReportBlock {
  uint32_t source_ssrc = 0;
  uint8_t fraction_lost = 0;
  int32_t cumulative_lost = 0;
  uint32_t highest_sequence = 0;
  uint32_t jitter = 0;
  uint32_t last_sender_report = 0;
  uint32_t delay_since_last_sender_report = 0;
};

struct RtcpSenderReport {
  uint32_t sender_ssrc = 0;
  uint32_t ntp_seconds = 0;
  uint32_t ntp_fraction = 0;
  uint32_t rtp_timestamp = 0;
  uint32_t packet_count = 0;
  uint32_t octet_count = 0;
  std::vector<RtcpReportBlock> reports;
};

struct RtcpReceiverReport {
  uint32_t sender_ssrc = 0;
  std::vector<RtcpReportBlock> reports;
};

struct RtcpNack {
  uint32_t sender_ssrc = 0;
  uint32_t media_ssrc = 0;
  uint16_t pid = 0;
  uint16_t blp = 0;
};

struct RtcpPli {
  uint32_t sender_ssrc = 0;
  uint32_t media_ssrc = 0;
};

struct RtcpFir {
  uint32_t sender_ssrc = 0;
  uint32_t media_ssrc = 0;
  uint8_t sequence_number = 0;
};

struct RtcpTmmbr {
  uint32_t sender_ssrc = 0;
  uint32_t media_ssrc = 0;
  uint32_t bitrate_bps = 0;
  uint16_t overhead_bytes = 0;
};

enum class MlvcControlType : uint8_t {
  kCommand = 1,
  kAck = 2,
  kReject = 3,
  kStatus = 4,
  kCapabilities = 5,
};

constexpr uint16_t kMlvcControlAtomic = 0x0001;

struct MlvcControlTlv {
  uint16_t type = 0;
  std::vector<uint8_t> value;
};

struct MlvcControlMessage {
  uint8_t version = 1;
  MlvcControlType type = MlvcControlType::kCommand;
  uint16_t flags = 0;
  uint32_t transaction_id = 0;
  uint32_t media_ssrc = 0;
  uint32_t apply_after_frame_id = 0xffffffffu;
  std::vector<MlvcControlTlv> tlvs;
  // SSRC from the RTCP APP header.  It is populated by the decoder and is
  // intentionally kept out of the 16-byte MLVC control envelope.
  uint32_t control_ssrc = 0;
};

struct RtcpCompoundContents {
  std::optional<RtcpSenderReport> sender_report;
  std::optional<RtcpReceiverReport> receiver_report;
  std::string cname;
  std::vector<uint32_t> bye_sources;
  std::vector<RtcpNack> nacks;
  std::vector<RtcpPli> plis;
  std::vector<RtcpFir> firs;
  std::vector<RtcpTmmbr> tmmbrs;
  std::vector<MlvcControlMessage> mlvc_controls;
};

// Build and parse the RTCP packets used by the MLVC v1 control plane. The
// compound builder includes RR/SR and SDES CNAME before optional feedback or
// APP packets, as required by the RFC 3550 compound profile.
std::vector<uint8_t> EncodeRtcpSenderReport(const RtcpSenderReport& report,
                                            const std::string& cname,
                                            const std::vector<std::vector<uint8_t>>& extra = {});
std::vector<uint8_t> EncodeRtcpReceiverReport(const RtcpReceiverReport& report,
                                              const std::string& cname,
                                              const std::vector<std::vector<uint8_t>>& extra = {});
std::vector<uint8_t> EncodeRtcpNack(const RtcpNack& nack);
std::vector<uint8_t> EncodeRtcpPli(const RtcpPli& pli);
std::vector<uint8_t> EncodeRtcpFir(const RtcpFir& fir);
std::vector<uint8_t> EncodeRtcpTmmbr(const RtcpTmmbr& tmmbr);
std::vector<uint8_t> EncodeMlvcRtcpApp(uint32_t control_ssrc,
                                       const MlvcControlMessage& message);
MlvcControlMessage DecodeMlvcRtcpApp(const std::vector<uint8_t>& packet);
RtcpCompoundContents DecodeRtcpCompound(const std::vector<uint8_t>& packet);

// UDP transport for compound RTCP.  It deliberately has no worker thread:
// callers decide when to poll and can keep RTCP I/O outside the codec thread.
// A local port of zero requests an ephemeral port; a remote port of zero makes
// Send unavailable while still allowing local reception.
class RtcpUdpEndpoint {
 public:
  RtcpUdpEndpoint(uint16_t local_port, std::string remote_host = {},
                  uint16_t remote_port = 0);
  ~RtcpUdpEndpoint();
  RtcpUdpEndpoint(const RtcpUdpEndpoint&) = delete;
  RtcpUdpEndpoint& operator=(const RtcpUdpEndpoint&) = delete;

  uint16_t local_port() const;
  void Send(const std::vector<uint8_t>& compound_packet);
  // Pins the source of the most recent successful Receive as the reply peer.
  // Call only after validating its session identity.
  void AcceptLastPeer();
  // Replies to the peer accepted by AcceptLastPeer.
  void SendToLastPeer(const std::vector<uint8_t>& compound_packet);
  bool Receive(std::vector<uint8_t>* compound_packet,
               std::chrono::milliseconds timeout);
  void Close();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Builds and validates the minimal RFC 3550 compound RTCP packet used to end
// an RTP sender session: RR + SDES/CNAME + BYE.  The media stream does not use
// the optional MLVC EOS media unit as its RTP session terminator.
std::vector<uint8_t> EncodeRtcpSessionBye(uint32_t ssrc, const std::string& cname);
uint32_t DecodeRtcpSessionBye(const std::vector<uint8_t>& packet);

}  // namespace mlvc::transport

#endif  // MLVC_TRANSPORT_RTCP_SESSION_H_
