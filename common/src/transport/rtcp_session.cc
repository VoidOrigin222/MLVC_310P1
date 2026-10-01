#include "mlvc/transport/rtcp_session.h"

#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#ifdef MLVC_HAS_OPENSSL
#include <openssl/hmac.h>
#endif
#ifndef _WIN32
#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <netdb.h>
#include <limits>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace mlvc::transport {
namespace {

#ifdef MLVC_HAS_OPENSSL
constexpr uint8_t kAuthMagic[] = {0x4d, 0x4c, 0x56, 0x41};
constexpr std::size_t kAuthTrailerBytes = 4 + 1 + 8 + 16;
std::vector<uint8_t> AuthenticateRtcp(const std::vector<uint8_t>& packet, uint64_t index,
                                      const std::string& key) {
  std::vector<uint8_t> out = packet;
  out.insert(out.end(), std::begin(kAuthMagic), std::end(kAuthMagic));
  out.push_back(1);
  for (int shift = 56; shift >= 0; shift -= 8) out.push_back(static_cast<uint8_t>(index >> shift));
  unsigned char digest[EVP_MAX_MD_SIZE]{};
  unsigned int digest_len = 0;
  HMAC_CTX* ctx = HMAC_CTX_new();
  HMAC_Init_ex(ctx, key.data(), static_cast<int>(key.size()), EVP_sha256(), nullptr);
  HMAC_Update(ctx, out.data(), out.size());
  HMAC_Final(ctx, digest, &digest_len);
  HMAC_CTX_free(ctx);
  out.insert(out.end(), digest, digest + 16);
  return out;
}
bool VerifyAndStripRtcp(std::vector<uint8_t>* packet, const std::string& key,
                        uint64_t* highest, uint64_t* window) {
  if (packet->size() < kAuthTrailerBytes) return false;
  const std::size_t begin = packet->size() - kAuthTrailerBytes;
  if (!std::equal(std::begin(kAuthMagic), std::end(kAuthMagic), packet->begin() + begin) ||
      (*packet)[begin + 4] != 1) return false;
  uint64_t index = 0;
  for (int i = 0; i < 8; ++i) index = (index << 8) | (*packet)[begin + 5 + i];
  const auto expected = AuthenticateRtcp(
      std::vector<uint8_t>(packet->begin(), packet->begin() + begin), index, key);
  if (!std::equal(expected.end() - 16, expected.end(), packet->end() - 16)) return false;
  if (index > *highest) {
    const uint64_t shift = index - *highest;
    *window = shift >= 64 ? 1ull : ((*window << shift) | 1ull);
    *highest = index;
  } else {
    const uint64_t delta = *highest - index;
    if (delta >= 64 || ((*window >> delta) & 1ull) != 0) return false;
    *window |= 1ull << delta;
  }
  packet->resize(begin);
  return true;
}
#endif

constexpr uint8_t kRtcpVersion = 2;
constexpr uint8_t kPtSenderReport = 200;
constexpr uint8_t kPtReceiverReport = 201;
constexpr uint8_t kPtSdes = 202;
constexpr uint8_t kPtBye = 203;
constexpr uint8_t kPtApp = 204;
constexpr uint8_t kPtRtpFeedback = 205;
constexpr uint8_t kPtPayloadFeedback = 206;
constexpr uint8_t kFmtNack = 1;
constexpr uint8_t kFmtTmmbr = 3;
constexpr uint8_t kFmtPli = 1;
constexpr uint8_t kFmtFir = 4;
constexpr uint32_t kMlvcAppName = 0x4d4c5643u;  // "MLVC"

void PutRtcpLength(std::vector<uint8_t>* bytes, std::size_t packet_offset);

void Put32(std::vector<uint8_t>* bytes, uint32_t value) {
  bytes->push_back(static_cast<uint8_t>(value >> 24));
  bytes->push_back(static_cast<uint8_t>(value >> 16));
  bytes->push_back(static_cast<uint8_t>(value >> 8));
  bytes->push_back(static_cast<uint8_t>(value));
}

void Put16(std::vector<uint8_t>* bytes, uint16_t value) {
  bytes->push_back(static_cast<uint8_t>(value >> 8));
  bytes->push_back(static_cast<uint8_t>(value));
}

uint16_t Get16(const std::vector<uint8_t>& bytes, std::size_t offset) {
  return static_cast<uint16_t>((static_cast<uint16_t>(bytes[offset]) << 8) |
                               bytes[offset + 1]);
}

uint32_t Get32(const std::vector<uint8_t>& bytes, std::size_t offset) {
  return (static_cast<uint32_t>(bytes[offset]) << 24) |
         (static_cast<uint32_t>(bytes[offset + 1]) << 16) |
         (static_cast<uint32_t>(bytes[offset + 2]) << 8) |
         static_cast<uint32_t>(bytes[offset + 3]);
}

void Require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

void RequireNonZeroSsrc(uint32_t ssrc, const char* message) {
  Require(ssrc != 0, message);
}

void PutReportBlock(std::vector<uint8_t>* bytes, const RtcpReportBlock& report) {
  RequireNonZeroSsrc(report.source_ssrc, "RTCP report block has a zero SSRC");
  Require(report.cumulative_lost >= -8388608 && report.cumulative_lost <= 8388607,
          "RTCP cumulative lost is outside the signed 24-bit range");
  Put32(bytes, report.source_ssrc);
  bytes->push_back(report.fraction_lost);
  const uint32_t lost = static_cast<uint32_t>(report.cumulative_lost) & 0x00ffffffu;
  bytes->push_back(static_cast<uint8_t>(lost >> 16));
  bytes->push_back(static_cast<uint8_t>(lost >> 8));
  bytes->push_back(static_cast<uint8_t>(lost));
  Put32(bytes, report.highest_sequence);
  Put32(bytes, report.jitter);
  Put32(bytes, report.last_sender_report);
  Put32(bytes, report.delay_since_last_sender_report);
}

RtcpReportBlock GetReportBlock(const std::vector<uint8_t>& bytes, std::size_t offset) {
  const uint32_t lost24 = (static_cast<uint32_t>(bytes[offset + 5]) << 16) |
                          (static_cast<uint32_t>(bytes[offset + 6]) << 8) |
                          static_cast<uint32_t>(bytes[offset + 7]);
  const int32_t lost = (lost24 & 0x00800000u) != 0
                           ? static_cast<int32_t>(lost24 | 0xff000000u)
                           : static_cast<int32_t>(lost24);
  return RtcpReportBlock{Get32(bytes, offset), bytes[offset + 4], lost,
                         Get32(bytes, offset + 8), Get32(bytes, offset + 12),
                         Get32(bytes, offset + 16), Get32(bytes, offset + 20)};
}

struct RtcpPacketView {
  std::size_t offset = 0;
  std::size_t end = 0;
  std::size_t payload_end = 0;
  uint8_t count = 0;
  uint8_t type = 0;
  uint8_t padding = 0;
};

RtcpPacketView ReadPacket(const std::vector<uint8_t>& packet, std::size_t offset,
                          bool allow_padding) {
  Require(offset <= packet.size() && packet.size() - offset >= 4,
          "truncated RTCP packet header");
  Require((packet[offset] >> 6) == kRtcpVersion, "invalid RTCP version");
  const uint8_t padding_bit = static_cast<uint8_t>(packet[offset] & 0x20u);
  const std::size_t packet_bytes = (static_cast<std::size_t>(Get16(packet, offset + 2)) + 1u) * 4u;
  Require(packet_bytes >= 4 && packet_bytes <= packet.size() - offset,
          "RTCP length exceeds compound packet");
  const std::size_t end = offset + packet_bytes;
  uint8_t padding = 0;
  if (padding_bit != 0) {
    Require(allow_padding && end == packet.size(),
            "RTCP padding is only allowed on the final packet");
    padding = packet[end - 1];
    Require(padding != 0 && padding <= packet_bytes - 4, "invalid RTCP padding");
  }
  return RtcpPacketView{offset, end, end - padding, static_cast<uint8_t>(packet[offset] & 0x1fu),
                        packet[offset + 1], padding};
}

void ValidateReportCount(uint8_t count) {
  Require(count <= 31, "RTCP report count exceeds the five-bit field");
}

RtcpSenderReport ParseSenderReport(const std::vector<uint8_t>& packet,
                                   const RtcpPacketView& view) {
  ValidateReportCount(view.count);
  const std::size_t expected = 4u + 24u + 24u * view.count;
  Require(view.payload_end - view.offset == expected, "invalid RTCP sender report length");
  const std::size_t body = view.offset + 4;
  RtcpSenderReport report;
  report.sender_ssrc = Get32(packet, body);
  RequireNonZeroSsrc(report.sender_ssrc, "RTCP sender report has a zero SSRC");
  report.ntp_seconds = Get32(packet, body + 4);
  report.ntp_fraction = Get32(packet, body + 8);
  report.rtp_timestamp = Get32(packet, body + 12);
  report.packet_count = Get32(packet, body + 16);
  report.octet_count = Get32(packet, body + 20);
  report.reports.reserve(view.count);
  std::size_t cursor = body + 24;
  for (uint8_t i = 0; i < view.count; ++i) {
    report.reports.push_back(GetReportBlock(packet, cursor));
    RequireNonZeroSsrc(report.reports.back().source_ssrc,
                       "RTCP report block has a zero SSRC");
    cursor += 24;
  }
  return report;
}

RtcpReceiverReport ParseReceiverReport(const std::vector<uint8_t>& packet,
                                       const RtcpPacketView& view) {
  ValidateReportCount(view.count);
  const std::size_t expected = 4u + 4u + 24u * view.count;
  Require(view.payload_end - view.offset == expected, "invalid RTCP receiver report length");
  const std::size_t body = view.offset + 4;
  RtcpReceiverReport report;
  report.sender_ssrc = Get32(packet, body);
  RequireNonZeroSsrc(report.sender_ssrc, "RTCP receiver report has a zero SSRC");
  report.reports.reserve(view.count);
  std::size_t cursor = body + 4;
  for (uint8_t i = 0; i < view.count; ++i) {
    report.reports.push_back(GetReportBlock(packet, cursor));
    RequireNonZeroSsrc(report.reports.back().source_ssrc,
                       "RTCP report block has a zero SSRC");
    cursor += 24;
  }
  return report;
}

void AppendSdesCname(std::vector<uint8_t>* packet, uint32_t ssrc,
                     const std::string& cname) {
  RequireNonZeroSsrc(ssrc, "RTCP CNAME requires a non-zero SSRC");
  Require(!cname.empty() && cname.size() <= 255,
          "RTCP CNAME requires 1..255 bytes");
  const std::size_t offset = packet->size();
  packet->insert(packet->end(), {0x81, kPtSdes, 0, 0});
  Put32(packet, ssrc);
  packet->push_back(1);
  packet->push_back(static_cast<uint8_t>(cname.size()));
  packet->insert(packet->end(), cname.begin(), cname.end());
  packet->push_back(0);
  while (((packet->size() - offset) & 3u) != 0) packet->push_back(0);
  PutRtcpLength(packet, offset);
}

std::vector<uint8_t> BuildCompound(std::vector<uint8_t> report, uint32_t ssrc,
                                   const std::string& cname,
                                   const std::vector<std::vector<uint8_t>>& extra) {
  Require(!report.empty() && report.size() % 4 == 0,
          "RTCP report packet must be word aligned");
  std::vector<uint8_t> compound = std::move(report);
  AppendSdesCname(&compound, ssrc, cname);
  for (const auto& item : extra) {
    Require(item.size() >= 4 && item.size() % 4 == 0 && (item[0] >> 6) == kRtcpVersion,
            "RTCP compound extension is malformed");
    const std::size_t declared = (static_cast<std::size_t>(Get16(item, 2)) + 1u) * 4u;
    Require(declared == item.size(), "RTCP compound extension length does not match bytes");
    Require((item[0] & 0x20u) == 0,
            "RTCP compound extensions with padding are not supported");
    compound.insert(compound.end(), item.begin(), item.end());
  }
  return compound;
}

uint32_t EncodeTmmbrBitrate(uint32_t bitrate_bps) {
  uint8_t exponent = 0;
  uint64_t mantissa = bitrate_bps;
  while (mantissa > 0x1ffffu && exponent < 63) {
    mantissa = (mantissa + 1u) >> 1;
    ++exponent;
  }
  return (static_cast<uint32_t>(exponent) << 26) |
         (static_cast<uint32_t>(mantissa) << 9);
}

uint32_t DecodeTmmbrBitrate(uint32_t word) {
  const uint32_t exponent = word >> 26;
  const uint32_t mantissa = (word >> 9) & 0x1ffffu;
  const uint64_t value = static_cast<uint64_t>(mantissa) << exponent;
  return static_cast<uint32_t>(std::min<uint64_t>(value, std::numeric_limits<uint32_t>::max()));
}

void ValidateControlMessage(const MlvcControlMessage& message) {
  Require(message.version == 1, "unsupported MLVC RTCP control version");
  const uint8_t type = static_cast<uint8_t>(message.type);
  Require(type >= static_cast<uint8_t>(MlvcControlType::kCommand) &&
              type <= static_cast<uint8_t>(MlvcControlType::kCapabilities),
          "unknown MLVC RTCP control message type");
  Require((message.flags & ~kMlvcControlAtomic) == 0,
          "unknown MLVC RTCP control flags");
  if (message.type == MlvcControlType::kCommand) {
    Require((message.flags & kMlvcControlAtomic) != 0,
            "MLVC COMMAND must set the ATOMIC flag");
    Require(message.transaction_id != 0,
            "MLVC COMMAND requires a non-zero transaction ID");
  }
}

std::size_t PaddedTlvBytes(std::size_t length) {
  Require(length <= std::numeric_limits<uint16_t>::max(),
          "MLVC RTCP TLV is too large");
  return (length + 3u) & ~std::size_t{3};
}

void AppendControlTlvs(std::vector<uint8_t>* packet,
                       const std::vector<MlvcControlTlv>& tlvs) {
  for (const auto& tlv : tlvs) {
    const std::size_t padded = PaddedTlvBytes(tlv.value.size());
    Put16(packet, tlv.type);
    Put16(packet, static_cast<uint16_t>(tlv.value.size()));
    packet->insert(packet->end(), tlv.value.begin(), tlv.value.end());
    packet->insert(packet->end(), padded - tlv.value.size(), 0);
  }
}

void ParseControlTlvs(const std::vector<uint8_t>& packet, std::size_t begin,
                      std::size_t end, std::vector<MlvcControlTlv>* tlvs) {
  while (begin < end) {
    Require(end - begin >= 4, "truncated MLVC RTCP control TLV header");
    const uint16_t type = Get16(packet, begin);
    const uint16_t length = Get16(packet, begin + 2);
    const std::size_t padded = PaddedTlvBytes(length);
    Require(padded <= end - begin - 4,
            "MLVC RTCP control TLV exceeds APP payload");
    const std::size_t value_begin = begin + 4;
    const std::size_t value_end = value_begin + length;
    for (std::size_t i = value_end; i < value_begin + padded; ++i) {
      Require(packet[i] == 0, "MLVC RTCP control TLV padding is non-zero");
    }
    MlvcControlTlv tlv;
    tlv.type = type;
    tlv.value.assign(packet.begin() + static_cast<std::ptrdiff_t>(value_begin),
                     packet.begin() + static_cast<std::ptrdiff_t>(value_end));
    tlvs->push_back(std::move(tlv));
    begin += 4 + padded;
  }
  Require(begin == end, "MLVC RTCP control TLVs are not word aligned");
}

void ParseSdes(const std::vector<uint8_t>& packet, const RtcpPacketView& view,
               uint32_t sender_ssrc, std::string* cname) {
  Require(view.count != 0, "RTCP SDES packet has no chunks");
  std::size_t cursor = view.offset + 4;
  for (uint8_t chunk = 0; chunk < view.count; ++chunk) {
    Require(view.payload_end - cursor >= 4, "truncated RTCP SDES source");
    const uint32_t source_ssrc = Get32(packet, cursor);
    RequireNonZeroSsrc(source_ssrc, "RTCP SDES chunk has a zero SSRC");
    cursor += 4;
    bool ended = false;
    while (cursor < view.payload_end) {
      const uint8_t item_type = packet[cursor++];
      if (item_type == 0) {
        ended = true;
        while (((cursor - view.offset) & 3u) != 0) {
          Require(cursor < view.payload_end, "truncated RTCP SDES chunk padding");
          Require(packet[cursor++] == 0, "invalid RTCP SDES chunk padding");
        }
        break;
      }
      Require(cursor < view.payload_end, "truncated RTCP SDES item length");
      const uint8_t item_length = packet[cursor++];
      Require(item_length <= view.payload_end - cursor, "truncated RTCP SDES item");
      if (item_type == 1 && source_ssrc == sender_ssrc) {
        Require(item_length != 0, "RTCP CNAME item is empty");
        const std::string value(packet.begin() + static_cast<std::ptrdiff_t>(cursor),
                                packet.begin() + static_cast<std::ptrdiff_t>(cursor + item_length));
        if (cname->empty()) *cname = value;
        else Require(*cname == value, "RTCP sender has conflicting CNAME items");
      }
      cursor += item_length;
    }
    Require(ended, "RTCP SDES chunk lacks a terminator");
  }
  Require(cursor == view.payload_end, "trailing bytes in RTCP SDES packet");
}

void ParseBye(const std::vector<uint8_t>& packet, const RtcpPacketView& view,
              std::vector<uint32_t>* sources) {
  Require(view.count != 0, "RTCP BYE packet has no sources");
  const std::size_t source_bytes = static_cast<std::size_t>(view.count) * 4u;
  Require(view.payload_end - (view.offset + 4) >= source_bytes,
          "truncated RTCP BYE source list");
  std::size_t cursor = view.offset + 4;
  for (uint8_t i = 0; i < view.count; ++i) {
    const uint32_t source = Get32(packet, cursor);
    RequireNonZeroSsrc(source, "RTCP BYE source is zero");
    sources->push_back(source);
    cursor += 4;
  }
  if (cursor != view.payload_end) {
    const uint8_t reason_length = packet[cursor++];
    Require(reason_length <= view.payload_end - cursor, "truncated RTCP BYE reason");
    cursor += reason_length;
    Require(cursor == view.payload_end, "RTCP BYE has trailing reason bytes");
  }
}

void ParseNack(const std::vector<uint8_t>& packet, const RtcpPacketView& view,
               std::vector<RtcpNack>* nacks) {
  Require(view.count == kFmtNack, "unsupported RTPFB feedback format");
  Require(view.payload_end - (view.offset + 4) >= 12,
          "truncated RTCP Generic NACK packet");
  const std::size_t body = view.offset + 4;
  const std::size_t fci_begin = body + 8;
  Require((view.payload_end - fci_begin) % 4 == 0,
          "RTCP Generic NACK FCI is not word aligned");
  const uint32_t sender_ssrc = Get32(packet, body);
  const uint32_t media_ssrc = Get32(packet, body + 4);
  RequireNonZeroSsrc(sender_ssrc, "RTCP Generic NACK sender SSRC is zero");
  for (std::size_t cursor = fci_begin; cursor < view.payload_end; cursor += 4) {
    nacks->push_back(RtcpNack{sender_ssrc, media_ssrc, Get16(packet, cursor),
                              Get16(packet, cursor + 2)});
  }
}

void ParsePli(const std::vector<uint8_t>& packet, const RtcpPacketView& view,
              std::vector<RtcpPli>* plis) {
  Require(view.count == kFmtPli, "unsupported PSFB feedback format");
  Require(view.payload_end - view.offset == 12, "invalid RTCP PLI length");
  const std::size_t body = view.offset + 4;
  const uint32_t sender_ssrc = Get32(packet, body);
  RequireNonZeroSsrc(sender_ssrc, "RTCP PLI sender SSRC is zero");
  plis->push_back(RtcpPli{sender_ssrc, Get32(packet, body + 4)});
}

void ParseFir(const std::vector<uint8_t>& packet, const RtcpPacketView& view,
              std::vector<RtcpFir>* firs) {
  Require(view.count == kFmtFir, "unsupported PSFB feedback format");
  Require(view.payload_end - (view.offset + 4) >= 16,
          "truncated RTCP FIR packet");
  const std::size_t body = view.offset + 4;
  const uint32_t sender_ssrc = Get32(packet, body);
  RequireNonZeroSsrc(sender_ssrc, "RTCP FIR sender SSRC is zero");
  Require(Get32(packet, body + 4) == 0, "RTCP FIR media SSRC must be zero");
  const std::size_t fci_begin = body + 8;
  Require((view.payload_end - fci_begin) % 8 == 0,
          "RTCP FIR FCI is not entry aligned");
  for (std::size_t cursor = fci_begin; cursor < view.payload_end; cursor += 8) {
    const uint32_t target = Get32(packet, cursor);
    RequireNonZeroSsrc(target, "RTCP FIR target SSRC is zero");
    Require(packet[cursor + 5] == 0 && packet[cursor + 6] == 0 && packet[cursor + 7] == 0,
            "RTCP FIR reserved bits are non-zero");
    firs->push_back(RtcpFir{sender_ssrc, target, packet[cursor + 4]});
  }
}

void ParseTmmbr(const std::vector<uint8_t>& packet, const RtcpPacketView& view,
                std::vector<RtcpTmmbr>* tmmbrs) {
  Require(view.count == kFmtTmmbr, "unsupported TMMBR feedback format");
  Require(view.payload_end - (view.offset + 4) >= 16,
          "truncated RTCP TMMBR packet");
  const std::size_t body = view.offset + 4;
  const uint32_t sender_ssrc = Get32(packet, body);
  RequireNonZeroSsrc(sender_ssrc, "RTCP TMMBR sender SSRC is zero");
  Require(Get32(packet, body + 4) == 0, "RTCP TMMBR common media SSRC must be zero");
  const std::size_t fci_begin = body + 8;
  Require((view.payload_end - fci_begin) % 8 == 0,
          "RTCP TMMBR FCI is not entry aligned");
  for (std::size_t cursor = fci_begin; cursor < view.payload_end; cursor += 8) {
    const uint32_t target = Get32(packet, cursor);
    RequireNonZeroSsrc(target, "RTCP TMMBR target SSRC is zero");
    const uint32_t word = Get32(packet, cursor + 4);
    tmmbrs->push_back(RtcpTmmbr{sender_ssrc, target, DecodeTmmbrBitrate(word),
                                static_cast<uint16_t>(word & 0x1ffu)});
  }
}

}  // namespace

std::vector<uint8_t> EncodeRtcpSenderReport(
    const RtcpSenderReport& report, const std::string& cname,
    const std::vector<std::vector<uint8_t>>& extra) {
  RequireNonZeroSsrc(report.sender_ssrc, "RTCP sender report SSRC is zero");
  Require(report.reports.size() <= 31, "RTCP sender report has too many report blocks");
  std::vector<uint8_t> packet;
  packet.insert(packet.end(), {static_cast<uint8_t>(0x80u | report.reports.size()),
                               kPtSenderReport, 0, 0});
  Put32(&packet, report.sender_ssrc);
  Put32(&packet, report.ntp_seconds);
  Put32(&packet, report.ntp_fraction);
  Put32(&packet, report.rtp_timestamp);
  Put32(&packet, report.packet_count);
  Put32(&packet, report.octet_count);
  for (const auto& block : report.reports) PutReportBlock(&packet, block);
  PutRtcpLength(&packet, 0);
  return BuildCompound(std::move(packet), report.sender_ssrc, cname, extra);
}

std::vector<uint8_t> EncodeRtcpReceiverReport(
    const RtcpReceiverReport& report, const std::string& cname,
    const std::vector<std::vector<uint8_t>>& extra) {
  RequireNonZeroSsrc(report.sender_ssrc, "RTCP receiver report SSRC is zero");
  Require(report.reports.size() <= 31, "RTCP receiver report has too many report blocks");
  std::vector<uint8_t> packet;
  packet.insert(packet.end(), {static_cast<uint8_t>(0x80u | report.reports.size()),
                               kPtReceiverReport, 0, 0});
  Put32(&packet, report.sender_ssrc);
  for (const auto& block : report.reports) PutReportBlock(&packet, block);
  PutRtcpLength(&packet, 0);
  return BuildCompound(std::move(packet), report.sender_ssrc, cname, extra);
}

std::vector<uint8_t> EncodeRtcpNack(const RtcpNack& nack) {
  RequireNonZeroSsrc(nack.sender_ssrc, "RTCP Generic NACK sender SSRC is zero");
  std::vector<uint8_t> packet;
  packet.insert(packet.end(), {static_cast<uint8_t>(0x80u | kFmtNack), kPtRtpFeedback, 0, 0});
  Put32(&packet, nack.sender_ssrc);
  Put32(&packet, nack.media_ssrc);
  Put16(&packet, nack.pid);
  Put16(&packet, nack.blp);
  PutRtcpLength(&packet, 0);
  return packet;
}

std::vector<uint8_t> EncodeRtcpPli(const RtcpPli& pli) {
  RequireNonZeroSsrc(pli.sender_ssrc, "RTCP PLI sender SSRC is zero");
  std::vector<uint8_t> packet;
  packet.insert(packet.end(), {static_cast<uint8_t>(0x80u | kFmtPli), kPtPayloadFeedback, 0, 0});
  Put32(&packet, pli.sender_ssrc);
  Put32(&packet, pli.media_ssrc);
  PutRtcpLength(&packet, 0);
  return packet;
}

std::vector<uint8_t> EncodeRtcpFir(const RtcpFir& fir) {
  RequireNonZeroSsrc(fir.sender_ssrc, "RTCP FIR sender SSRC is zero");
  RequireNonZeroSsrc(fir.media_ssrc, "RTCP FIR target SSRC is zero");
  std::vector<uint8_t> packet;
  packet.insert(packet.end(), {static_cast<uint8_t>(0x80u | kFmtFir), kPtPayloadFeedback, 0, 0});
  Put32(&packet, fir.sender_ssrc);
  Put32(&packet, 0);
  Put32(&packet, fir.media_ssrc);
  packet.insert(packet.end(), {fir.sequence_number, 0, 0, 0});
  PutRtcpLength(&packet, 0);
  return packet;
}

std::vector<uint8_t> EncodeRtcpTmmbr(const RtcpTmmbr& tmmbr) {
  RequireNonZeroSsrc(tmmbr.sender_ssrc, "RTCP TMMBR sender SSRC is zero");
  RequireNonZeroSsrc(tmmbr.media_ssrc, "RTCP TMMBR target SSRC is zero");
  Require(tmmbr.overhead_bytes <= 511, "RTCP TMMBR overhead exceeds nine bits");
  std::vector<uint8_t> packet;
  packet.insert(packet.end(), {static_cast<uint8_t>(0x80u | kFmtTmmbr), kPtRtpFeedback, 0, 0});
  Put32(&packet, tmmbr.sender_ssrc);
  Put32(&packet, 0);
  Put32(&packet, tmmbr.media_ssrc);
  Put32(&packet, EncodeTmmbrBitrate(tmmbr.bitrate_bps) | tmmbr.overhead_bytes);
  PutRtcpLength(&packet, 0);
  return packet;
}

std::vector<uint8_t> EncodeMlvcRtcpApp(uint32_t control_ssrc,
                                       const MlvcControlMessage& message) {
  RequireNonZeroSsrc(control_ssrc, "MLVC RTCP APP control SSRC is zero");
  if (message.control_ssrc != 0) {
    Require(message.control_ssrc == control_ssrc,
            "MLVC RTCP APP control SSRC does not match message metadata");
  }
  ValidateControlMessage(message);
  std::vector<uint8_t> packet;
  packet.insert(packet.end(), {0x80, kPtApp, 0, 0});
  Put32(&packet, control_ssrc);
  Put32(&packet, kMlvcAppName);
  packet.push_back(message.version);
  packet.push_back(static_cast<uint8_t>(message.type));
  Put16(&packet, message.flags);
  Put32(&packet, message.transaction_id);
  Put32(&packet, message.media_ssrc);
  Put32(&packet, message.apply_after_frame_id);
  AppendControlTlvs(&packet, message.tlvs);
  PutRtcpLength(&packet, 0);
  return packet;
}

MlvcControlMessage DecodeMlvcRtcpApp(const std::vector<uint8_t>& packet) {
  Require(!packet.empty() && packet.size() % 4 == 0,
          "MLVC RTCP APP packet is empty or unaligned");
  const RtcpPacketView view = ReadPacket(packet, 0, false);
  Require(view.end == packet.size(), "MLVC RTCP APP decoder expects one packet");
  Require(view.type == kPtApp && view.count == 0,
          "packet is not an MLVC APP subtype zero");
  Require(view.padding == 0, "MLVC RTCP APP padding is not supported");
  Require(view.payload_end - view.offset >= 4 + 4 + 16,
          "truncated MLVC RTCP APP envelope");
  const std::size_t body = view.offset + 4;
  const uint32_t control_ssrc = Get32(packet, body);
  RequireNonZeroSsrc(control_ssrc, "MLVC RTCP APP control SSRC is zero");
  Require(Get32(packet, body + 4) == kMlvcAppName,
          "RTCP APP name is not MLVC");
  const std::size_t envelope = body + 8;
  MlvcControlMessage message;
  message.version = packet[envelope];
  message.type = static_cast<MlvcControlType>(packet[envelope + 1]);
  message.flags = Get16(packet, envelope + 2);
  message.transaction_id = Get32(packet, envelope + 4);
  message.media_ssrc = Get32(packet, envelope + 8);
  message.apply_after_frame_id = Get32(packet, envelope + 12);
  message.control_ssrc = control_ssrc;
  ValidateControlMessage(message);
  ParseControlTlvs(packet, envelope + 16, view.payload_end, &message.tlvs);
  return message;
}

RtcpCompoundContents DecodeRtcpCompound(const std::vector<uint8_t>& packet) {
  Require(!packet.empty() && packet.size() % 4 == 0,
          "RTCP compound packet is empty or unaligned");
  RtcpCompoundContents contents;
  std::size_t offset = 0;
  uint32_t sender_ssrc = 0;
  bool first_packet = true;
  bool saw_sdes_cname = false;
  while (offset < packet.size()) {
    const RtcpPacketView view = ReadPacket(packet, offset, true);
    if (first_packet) {
      Require(view.type == kPtSenderReport || view.type == kPtReceiverReport,
              "RTCP compound packet must begin with SR or RR");
      first_packet = false;
    }
    if (view.type == kPtSenderReport) {
      Require(!contents.sender_report.has_value(), "RTCP compound has duplicate sender reports");
      contents.sender_report = ParseSenderReport(packet, view);
      if (sender_ssrc == 0) sender_ssrc = contents.sender_report->sender_ssrc;
      else Require(sender_ssrc == contents.sender_report->sender_ssrc,
                   "RTCP compound report SSRCs do not match");
    } else if (view.type == kPtReceiverReport) {
      Require(!contents.receiver_report.has_value(), "RTCP compound has duplicate receiver reports");
      contents.receiver_report = ParseReceiverReport(packet, view);
      if (sender_ssrc == 0) sender_ssrc = contents.receiver_report->sender_ssrc;
      else Require(sender_ssrc == contents.receiver_report->sender_ssrc,
                   "RTCP compound report SSRCs do not match");
    } else if (view.type == kPtSdes) {
      ParseSdes(packet, view, sender_ssrc, &contents.cname);
      saw_sdes_cname = saw_sdes_cname || !contents.cname.empty();
    } else if (view.type == kPtBye) {
      ParseBye(packet, view, &contents.bye_sources);
    } else if (view.type == kPtRtpFeedback) {
      if (view.count == kFmtNack) ParseNack(packet, view, &contents.nacks);
      else if (view.count == kFmtTmmbr) ParseTmmbr(packet, view, &contents.tmmbrs);
    } else if (view.type == kPtPayloadFeedback) {
      if (view.count == kFmtPli) ParsePli(packet, view, &contents.plis);
      else if (view.count == kFmtFir) ParseFir(packet, view, &contents.firs);
    } else if (view.type == kPtApp) {
      Require(view.padding == 0, "MLVC RTCP APP padding is not supported");
      const std::vector<uint8_t> app(packet.begin() + static_cast<std::ptrdiff_t>(view.offset),
                                     packet.begin() + static_cast<std::ptrdiff_t>(view.end));
      if (view.end - view.offset >= 12 && Get32(app, 8) == kMlvcAppName) {
        contents.mlvc_controls.push_back(DecodeMlvcRtcpApp(app));
      }
    }
    offset = view.end;
  }
  Require(!first_packet && sender_ssrc != 0,
          "RTCP compound packet has no sender report or receiver report");
  Require(saw_sdes_cname, "RTCP compound packet is missing sender CNAME");
  return contents;
}

std::vector<uint8_t> EncodeRtcpSessionBye(uint32_t ssrc, const std::string& cname) {
  RtcpReceiverReport report;
  report.sender_ssrc = ssrc;
  std::vector<uint8_t> packet = EncodeRtcpReceiverReport(report, cname);
  const std::size_t offset = packet.size();
  packet.insert(packet.end(), {0x81, kPtBye, 0, 0});
  Put32(&packet, ssrc);
  PutRtcpLength(&packet, offset);
  return packet;
}

uint32_t DecodeRtcpSessionBye(const std::vector<uint8_t>& packet) {
  const RtcpCompoundContents contents = DecodeRtcpCompound(packet);
  Require(!contents.bye_sources.empty(), "RTCP compound packet is missing BYE");
  const uint32_t sender_ssrc = contents.sender_report.has_value()
                                   ? contents.sender_report->sender_ssrc
                                   : contents.receiver_report->sender_ssrc;
  Require(std::find(contents.bye_sources.begin(), contents.bye_sources.end(), sender_ssrc) !=
              contents.bye_sources.end(),
          "RTCP BYE does not include the reporting SSRC");
  return sender_ssrc;
}

#ifndef _WIN32

class RtcpUdpEndpoint::Impl {
 public:
  int socket = -1;
  sockaddr_storage remote{};
  socklen_t remote_length = 0;
  sockaddr_storage last_peer{};
  socklen_t last_peer_length = 0;
  sockaddr_storage reply_peer{};
  socklen_t reply_peer_length = 0;
  uint16_t local_port = 0;
#ifdef MLVC_HAS_OPENSSL
  std::string auth_key;
  uint64_t tx_index = 0;
  uint64_t rx_highest = 0;
  uint64_t rx_window = 0;
#endif

  ~Impl() { Close(); }

  void Close() {
    if (socket >= 0) {
      ::close(socket);
      socket = -1;
    }
  }
};

RtcpUdpEndpoint::RtcpUdpEndpoint(uint16_t local_port, std::string remote_host,
                                 uint16_t remote_port)
    : impl_(std::make_unique<Impl>()) {
  if (remote_port != 0) {
    if (remote_host.empty()) throw std::invalid_argument("RTCP remote host is empty");
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    addrinfo* result = nullptr;
    const std::string service = std::to_string(remote_port);
    if (getaddrinfo(remote_host.c_str(), service.c_str(), &hints, &result) != 0 ||
        result == nullptr) {
      throw std::runtime_error("failed to resolve RTCP remote endpoint");
    }
    std::memcpy(&impl_->remote, result->ai_addr, result->ai_addrlen);
    impl_->remote_length = static_cast<socklen_t>(result->ai_addrlen);
    freeaddrinfo(result);
  }
  impl_->socket = ::socket(AF_INET, SOCK_DGRAM, 0);
#ifdef MLVC_HAS_OPENSSL
  if (const char* key = std::getenv("MLVC_RTCP_KEY"); key != nullptr) impl_->auth_key = key;
#endif
  if (impl_->socket < 0) throw std::runtime_error("failed to create RTCP socket");
  int receive_buffer = 1024 * 1024;
  (void)::setsockopt(impl_->socket, SOL_SOCKET, SO_RCVBUF, &receive_buffer,
                     sizeof(receive_buffer));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  address.sin_port = htons(local_port);
  if (::bind(impl_->socket, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
    impl_->Close();
    throw std::runtime_error("failed to bind RTCP socket");
  }
  sockaddr_in bound{};
  socklen_t bound_length = sizeof(bound);
  if (::getsockname(impl_->socket, reinterpret_cast<sockaddr*>(&bound), &bound_length) < 0) {
    impl_->Close();
    throw std::runtime_error("failed to query RTCP socket port");
  }
  impl_->local_port = ntohs(bound.sin_port);
}

RtcpUdpEndpoint::~RtcpUdpEndpoint() = default;

uint16_t RtcpUdpEndpoint::local_port() const {
  if (!impl_ || impl_->socket < 0) return 0;
  return impl_->local_port;
}

void RtcpUdpEndpoint::Send(const std::vector<uint8_t>& compound_packet) {
  if (!impl_ || impl_->socket < 0) throw std::runtime_error("RTCP socket is closed");
  if (impl_->remote_length == 0) throw std::runtime_error("RTCP remote endpoint is not configured");
  (void)DecodeRtcpCompound(compound_packet);
  std::vector<uint8_t> wire = compound_packet;
#ifdef MLVC_HAS_OPENSSL
  if (!impl_->auth_key.empty()) wire = AuthenticateRtcp(compound_packet, impl_->tx_index++, impl_->auth_key);
#endif
  const ssize_t sent = ::sendto(impl_->socket, wire.data(), wire.size(), 0,
                                reinterpret_cast<const sockaddr*>(&impl_->remote),
                                impl_->remote_length);
  if (sent < 0 || static_cast<std::size_t>(sent) != wire.size()) {
    throw std::runtime_error("failed to send RTCP packet");
  }
}

void RtcpUdpEndpoint::AcceptLastPeer() {
  if (!impl_ || impl_->socket < 0) throw std::runtime_error("RTCP socket is closed");
  if (impl_->last_peer_length == 0) throw std::runtime_error("RTCP peer has not sent a packet");
  impl_->reply_peer = impl_->last_peer;
  impl_->reply_peer_length = impl_->last_peer_length;
}

void RtcpUdpEndpoint::SendToLastPeer(const std::vector<uint8_t>& compound_packet) {
  if (!impl_ || impl_->socket < 0) throw std::runtime_error("RTCP socket is closed");
  if (impl_->reply_peer_length == 0) throw std::runtime_error("RTCP reply peer is not accepted");
  (void)DecodeRtcpCompound(compound_packet);
  std::vector<uint8_t> wire = compound_packet;
#ifdef MLVC_HAS_OPENSSL
  if (!impl_->auth_key.empty()) wire = AuthenticateRtcp(compound_packet, impl_->tx_index++, impl_->auth_key);
#endif
  const ssize_t sent = ::sendto(impl_->socket, wire.data(), wire.size(), 0,
                                reinterpret_cast<const sockaddr*>(&impl_->reply_peer),
                                impl_->reply_peer_length);
  if (sent < 0 || static_cast<std::size_t>(sent) != wire.size()) {
    throw std::runtime_error("failed to send RTCP reply");
  }
}

bool RtcpUdpEndpoint::Receive(std::vector<uint8_t>* compound_packet,
                              std::chrono::milliseconds timeout) {
  if (compound_packet == nullptr) throw std::invalid_argument("RTCP output is null");
  if (!impl_ || impl_->socket < 0) throw std::runtime_error("RTCP socket is closed");
  if (timeout.count() < 0) throw std::invalid_argument("RTCP timeout is negative");
  timeval tv{};
  tv.tv_sec = static_cast<time_t>(timeout.count() / 1000);
  tv.tv_usec = static_cast<suseconds_t>((timeout.count() % 1000) * 1000);
  (void)::setsockopt(impl_->socket, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  std::vector<uint8_t> buffer(65536);
  sockaddr_storage peer{};
  socklen_t peer_length = sizeof(peer);
  const ssize_t received = ::recvfrom(impl_->socket, buffer.data(), buffer.size(), 0,
                                      reinterpret_cast<sockaddr*>(&peer), &peer_length);
  if (received < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) return false;
    throw std::runtime_error("failed to receive RTCP packet");
  }
  if (received == 0) return false;
  buffer.resize(static_cast<std::size_t>(received));
  if (impl_->remote_length != 0) {
    if (peer.ss_family != AF_INET || impl_->remote.ss_family != AF_INET) return false;
    const auto* actual = reinterpret_cast<const sockaddr_in*>(&peer);
    const auto* expected = reinterpret_cast<const sockaddr_in*>(&impl_->remote);
    if (actual->sin_addr.s_addr != expected->sin_addr.s_addr ||
        actual->sin_port != expected->sin_port) {
      return false;
    }
  }
#ifdef MLVC_HAS_OPENSSL
  if (!impl_->auth_key.empty() &&
      !VerifyAndStripRtcp(&buffer, impl_->auth_key, &impl_->rx_highest, &impl_->rx_window)) return false;
#endif
  (void)DecodeRtcpCompound(buffer);
  impl_->last_peer = peer;
  impl_->last_peer_length = peer_length;
  *compound_packet = std::move(buffer);
  return true;
}

void RtcpUdpEndpoint::Close() {
  if (impl_) impl_->Close();
}

#else

class RtcpUdpEndpoint::Impl {};

RtcpUdpEndpoint::RtcpUdpEndpoint(uint16_t, std::string, uint16_t)
    : impl_(std::make_unique<Impl>()) {
  throw std::runtime_error("RTCP UDP endpoint is not supported on this build");
}
RtcpUdpEndpoint::~RtcpUdpEndpoint() = default;
uint16_t RtcpUdpEndpoint::local_port() const { return 0; }
void RtcpUdpEndpoint::Send(const std::vector<uint8_t>&) {
  throw std::runtime_error("RTCP UDP endpoint is not supported on this build");
}
void RtcpUdpEndpoint::SendToLastPeer(const std::vector<uint8_t>&) {
  throw std::runtime_error("RTCP UDP endpoint is not supported on this build");
}
void RtcpUdpEndpoint::AcceptLastPeer() {
  throw std::runtime_error("RTCP UDP endpoint is not supported on this build");
}
bool RtcpUdpEndpoint::Receive(std::vector<uint8_t>*, std::chrono::milliseconds) {
  throw std::runtime_error("RTCP UDP endpoint is not supported on this build");
}
void RtcpUdpEndpoint::Close() {}

#endif

namespace {

void PutRtcpLength(std::vector<uint8_t>* bytes, std::size_t packet_offset) {
  const std::size_t packet_bytes = bytes->size() - packet_offset;
  Require(packet_bytes >= 4 && packet_bytes % 4 == 0 && packet_bytes / 4 - 1 <= 0xffffu,
          "invalid RTCP packet size");
  const uint16_t words_minus_one = static_cast<uint16_t>(packet_bytes / 4 - 1);
  (*bytes)[packet_offset + 2] = static_cast<uint8_t>(words_minus_one >> 8);
  (*bytes)[packet_offset + 3] = static_cast<uint8_t>(words_minus_one);
}

}  // namespace

}  // namespace mlvc::transport
