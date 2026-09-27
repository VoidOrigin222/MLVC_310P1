#include "mlvc/transport/rtcp_session.h"

#include <algorithm>
#include <stdexcept>

namespace mlvc::transport {
namespace {

void Put32(std::vector<uint8_t>* bytes, uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8) {
    bytes->push_back(static_cast<uint8_t>(value >> shift));
  }
}

uint16_t Get16(const std::vector<uint8_t>& bytes, std::size_t offset) {
  return static_cast<uint16_t>((static_cast<uint16_t>(bytes[offset]) << 8) | bytes[offset + 1]);
}

uint32_t Get32(const std::vector<uint8_t>& bytes, std::size_t offset) {
  return (static_cast<uint32_t>(bytes[offset]) << 24) |
         (static_cast<uint32_t>(bytes[offset + 1]) << 16) |
         (static_cast<uint32_t>(bytes[offset + 2]) << 8) |
         static_cast<uint32_t>(bytes[offset + 3]);
}

void PutRtcpLength(std::vector<uint8_t>* bytes, std::size_t packet_offset) {
  const std::size_t packet_bytes = bytes->size() - packet_offset;
  if (packet_bytes == 0 || packet_bytes % 4 != 0 || packet_bytes / 4 - 1 > 0xffffu) {
    throw std::runtime_error("invalid RTCP packet size");
  }
  const uint16_t words_minus_one = static_cast<uint16_t>(packet_bytes / 4 - 1);
  (*bytes)[packet_offset + 2] = static_cast<uint8_t>(words_minus_one >> 8);
  (*bytes)[packet_offset + 3] = static_cast<uint8_t>(words_minus_one);
}

}  // namespace

std::vector<uint8_t> EncodeRtcpSessionBye(uint32_t ssrc, const std::string& cname) {
  if (ssrc == 0 || cname.empty() || cname.size() > 255) {
    throw std::invalid_argument("RTCP BYE requires a non-zero SSRC and a 1..255 byte CNAME");
  }

  std::vector<uint8_t> packet;
  // Receiver Report with no report blocks.
  packet.insert(packet.end(), {0x80, 201, 0, 1});
  Put32(&packet, ssrc);

  // One SDES chunk with a CNAME item and the required zero terminator/padding.
  const std::size_t sdes_offset = packet.size();
  packet.insert(packet.end(), {0x81, 202, 0, 0});
  Put32(&packet, ssrc);
  packet.push_back(1);  // CNAME item type.
  packet.push_back(static_cast<uint8_t>(cname.size()));
  packet.insert(packet.end(), cname.begin(), cname.end());
  packet.push_back(0);  // End of SDES items.
  while (((packet.size() - sdes_offset) & 3u) != 0) packet.push_back(0);
  PutRtcpLength(&packet, sdes_offset);

  // BYE for the same source.
  const std::size_t bye_offset = packet.size();
  packet.insert(packet.end(), {0x81, 203, 0, 0});
  Put32(&packet, ssrc);
  PutRtcpLength(&packet, bye_offset);
  return packet;
}

uint32_t DecodeRtcpSessionBye(const std::vector<uint8_t>& packet) {
  if (packet.size() < 20 || (packet.size() & 3u) != 0) {
    throw std::runtime_error("truncated or unaligned RTCP compound packet");
  }

  std::size_t offset = 0;
  uint32_t sender_ssrc = 0;
  bool first_packet = true;
  bool saw_sdes_cname = false;
  bool saw_bye = false;
  while (offset < packet.size()) {
    if (packet.size() - offset < 4 || (packet[offset] >> 6) != 2) {
      throw std::runtime_error("invalid RTCP packet header");
    }
    const uint8_t count = packet[offset] & 0x1fu;
    const uint8_t payload_type = packet[offset + 1];
    const std::size_t packet_bytes = (static_cast<std::size_t>(Get16(packet, offset + 2)) + 1) * 4;
    if (packet_bytes < 4 || packet_bytes > packet.size() - offset) {
      throw std::runtime_error("RTCP length exceeds compound packet");
    }
    const std::size_t packet_end = offset + packet_bytes;
    if ((packet[offset] & 0x20u) != 0) {
      if (packet_end != packet.size()) throw std::runtime_error("RTCP padding is only valid on the final packet");
      const uint8_t padding = packet[packet_end - 1];
      if (padding == 0 || padding > packet_bytes - 4) throw std::runtime_error("invalid RTCP padding");
    }

    if (first_packet) {
      if ((payload_type != 200 && payload_type != 201) || packet_bytes < 8) {
        throw std::runtime_error("RTCP compound packet must begin with SR or RR");
      }
      sender_ssrc = Get32(packet, offset + 4);
      if (sender_ssrc == 0) throw std::runtime_error("RTCP sender SSRC is zero");
      first_packet = false;
    } else if (payload_type == 202) {
      std::size_t cursor = offset + 4;
      for (uint8_t chunk = 0; chunk < count; ++chunk) {
        if (packet_end - cursor < 4 || Get32(packet, cursor) != sender_ssrc) {
          throw std::runtime_error("RTCP SDES source does not match sender SSRC");
        }
        cursor += 4;
        bool ended = false;
        while (cursor < packet_end) {
          const uint8_t item_type = packet[cursor++];
          if (item_type == 0) {
            ended = true;
            while (((cursor - (offset + 4)) & 3u) != 0 && cursor < packet_end) {
              if (packet[cursor++] != 0) throw std::runtime_error("invalid RTCP SDES padding");
            }
            break;
          }
          if (cursor >= packet_end) throw std::runtime_error("truncated RTCP SDES item length");
          const uint8_t item_length = packet[cursor++];
          if (item_length > packet_end - cursor) throw std::runtime_error("truncated RTCP SDES item");
          if (item_type == 1 && item_length != 0) saw_sdes_cname = true;
          cursor += item_length;
        }
        if (!ended) throw std::runtime_error("RTCP SDES chunk lacks terminator");
      }
      if (cursor != packet_end) throw std::runtime_error("trailing bytes in RTCP SDES packet");
    } else if (payload_type == 203) {
      std::size_t cursor = offset + 4;
      if (packet_end - cursor < static_cast<std::size_t>(count) * 4) {
        throw std::runtime_error("truncated RTCP BYE source list");
      }
      for (uint8_t source = 0; source < count; ++source) {
        if (Get32(packet, cursor) == sender_ssrc) saw_bye = true;
        cursor += 4;
      }
      if (cursor < packet_end) {
        const uint8_t reason_length = packet[cursor++];
        if (reason_length > packet_end - cursor) throw std::runtime_error("truncated RTCP BYE reason");
        cursor += reason_length;
        while (cursor < packet_end) {
          if (packet[cursor++] != 0) throw std::runtime_error("invalid RTCP BYE padding");
        }
      }
    }
    offset = packet_end;
  }
  if (first_packet || !saw_sdes_cname || !saw_bye) {
    throw std::runtime_error("RTCP compound packet is missing CNAME or BYE");
  }
  return sender_ssrc;
}

}  // namespace mlvc::transport
