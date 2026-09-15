#include "mlvc/transport/rtp_mlvc.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>
namespace mlvc::transport {
namespace {
void Put(std::vector<uint8_t>& o, uint32_t v) {
  for (int i = 3; i >= 0; --i) o.push_back((v >> (i * 8)) & 255);
}
uint32_t Get(const std::vector<uint8_t>& b, size_t& p) {
  if (p + 4 > b.size()) throw std::runtime_error("truncated RTP MLVC header");
  uint32_t v = 0;
  for (int i = 0; i < 4; ++i) v = (v << 8) | b[p++];
  return v;
}
}  // namespace
std::vector<uint8_t> EncodeRtpMlvcFragment(const RtpMlvcFragment& f) {
  if (f.payload.size() > 65535 || uint64_t(f.fragment_offset) + f.payload.size() > f.unit_length)
    throw std::runtime_error("invalid RTP MLVC fragment");
  std::vector<uint8_t> o = {'M', 'L', 1, (uint8_t)f.unit_type, f.flags, 0};
  Put(o, f.config_id);
  Put(o, f.unit_id);
  Put(o, f.fragment_offset);
  Put(o, f.unit_length);
  o.push_back(f.payload.size() >> 8);
  o.push_back(f.payload.size());
  o.insert(o.end(), f.payload.begin(), f.payload.end());
  return o;
}
RtpMlvcFragment DecodeRtpMlvcFragment(const std::vector<uint8_t>& b) {
  if (b.size() < 24 || b[0] != 'M' || b[1] != 'L' || b[2] != 1)
    throw std::runtime_error("invalid RTP MLVC header");
  size_t p = 3;
  uint8_t type = b[p++];
  uint8_t flags = b[p++];
  ++p;
  RtpMlvcFragment f{(RtpUnitType)type, flags, 0, 0, 0, 0, {}};
  f.config_id = Get(b, p);
  f.unit_id = Get(b, p);
  f.fragment_offset = Get(b, p);
  f.unit_length = Get(b, p);
  if (p + 2 > b.size()) throw std::runtime_error("truncated RTP MLVC length");
  size_t hi = b[p++], lo = b[p++];
  size_t n = (hi << 8) | lo;
  if (n != b.size() - p || uint64_t(f.fragment_offset) + n > f.unit_length)
    throw std::runtime_error("invalid RTP MLVC payload");
  f.payload.assign(b.begin() + p, b.end());
  return f;
}
RtpMlvcReassembler::RtpMlvcReassembler(std::size_t max_unit_bytes)
    : max_unit_bytes_(max_unit_bytes) {}
bool RtpMlvcReassembler::Push(const RtpMlvcFragment& f, std::vector<uint8_t>* unit) {
  if (!unit || f.unit_length > max_unit_bytes_ ||
      uint64_t(f.fragment_offset) + f.payload.size() > f.unit_length)
    throw std::runtime_error("invalid RTP unit");
  auto key = std::make_pair(f.config_id, f.unit_id);
  auto it = pending_.find(key);
  if (it == pending_.end()) {
    Pending p{f.unit_type,
              f.flags,
              f.unit_length,
              std::vector<uint8_t>(f.unit_length),
              std::vector<uint8_t>(f.unit_length),
              0};
    it = pending_.emplace(key, std::move(p)).first;
  }
  auto& p = it->second;
  if (p.type != f.unit_type || p.length != f.unit_length)
    throw std::runtime_error("conflicting RTP fragment");
  for (size_t i = 0; i < f.payload.size(); ++i) {
    size_t n = f.fragment_offset + i;
    if (p.seen[n] && p.data[n] != f.payload[i])
      throw std::runtime_error("conflicting duplicate RTP fragment");
    if (!p.seen[n]) {
      p.seen[n] = 1;
      p.data[n] = f.payload[i];
      ++p.count;
    }
  }
  if (p.count == p.length) {
    *unit = std::move(p.data);
    pending_.erase(it);
    return true;
  }
  return false;
}
void RtpMlvcReassembler::Reset() { pending_.clear(); }
struct RtpMlvcSender::Impl {
  int socket = -1;
  sockaddr_storage address{};
  socklen_t address_length = 0;
  uint16_t sequence = 0;
  uint32_t ssrc = 0;
  std::unique_ptr<UdpPacer> pacer;
  ~Impl() {
    if (socket >= 0) ::close(socket);
  }
  void Send(RtpUnitType type, uint8_t flags, uint32_t config_id, uint32_t unit_id,
            uint32_t timestamp, const std::vector<uint8_t>& unit) {
    constexpr std::size_t kRtpPayloadBytes = 1200;
    constexpr std::size_t kFragmentHeaderBytes = 24;
    if (unit.size() > UINT32_MAX) throw std::runtime_error("RTP MLVC unit is too large");
    const std::size_t chunk = kRtpPayloadBytes - kFragmentHeaderBytes;
    const std::size_t count = std::max<std::size_t>(1, (unit.size() + chunk - 1) / chunk);
    for (std::size_t i = 0; i < count; ++i) {
      const std::size_t offset = i * chunk;
      const std::size_t size = std::min(chunk, unit.size() - offset);
      RtpMlvcFragment fragment{
          type,
          flags,
          config_id,
          unit_id,
          static_cast<uint32_t>(offset),
          static_cast<uint32_t>(unit.size()),
          std::vector<uint8_t>(unit.begin() + offset, unit.begin() + offset + size)};
      const auto payload = EncodeRtpMlvcFragment(fragment);
      std::vector<uint8_t> packet(12 + payload.size());
      packet[0] = 0x80;
      packet[1] = static_cast<uint8_t>(96 | (i + 1 == count ? 0x80 : 0));
      packet[2] = static_cast<uint8_t>(sequence >> 8);
      packet[3] = static_cast<uint8_t>(sequence++);
      packet[4] = static_cast<uint8_t>(timestamp >> 24);
      packet[5] = static_cast<uint8_t>(timestamp >> 16);
      packet[6] = static_cast<uint8_t>(timestamp >> 8);
      packet[7] = static_cast<uint8_t>(timestamp);
      packet[8] = static_cast<uint8_t>(ssrc >> 24);
      packet[9] = static_cast<uint8_t>(ssrc >> 16);
      packet[10] = static_cast<uint8_t>(ssrc >> 8);
      packet[11] = static_cast<uint8_t>(ssrc);
      std::copy(payload.begin(), payload.end(), packet.begin() + 12);
      if (pacer) {
        const auto delay = pacer->ConsumeAndGetDelay(packet.size());
        if (delay.count() > 0) std::this_thread::sleep_for(delay);
      }
      if (::sendto(socket, packet.data(), packet.size(), 0,
                   reinterpret_cast<const sockaddr*>(&address), address_length) < 0) {
        throw std::runtime_error("failed to send RTP packet");
      }
    }
  }
};
RtpMlvcSender::RtpMlvcSender(const std::string& host, uint16_t port, uint32_t ssrc,
                             uint64_t pacing_rate_bps, std::size_t max_burst_bytes) {
  impl_ = std::make_unique<Impl>();
  impl_->ssrc = ssrc;
  impl_->pacer = std::make_unique<UdpPacer>(pacing_rate_bps, max_burst_bytes);
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_DGRAM;
  addrinfo* result = nullptr;
  const std::string service = std::to_string(port);
  if (getaddrinfo(host.c_str(), service.c_str(), &hints, &result) != 0 || result == nullptr)
    throw std::runtime_error("failed to resolve RTP destination");
  std::memcpy(&impl_->address, result->ai_addr, result->ai_addrlen);
  impl_->address_length = static_cast<socklen_t>(result->ai_addrlen);
  freeaddrinfo(result);
  impl_->socket = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (impl_->socket < 0) throw std::runtime_error("failed to create RTP socket");
}
RtpMlvcSender::~RtpMlvcSender() = default;
void RtpMlvcSender::SendUnit(RtpUnitType type, uint8_t flags, uint32_t config_id, uint32_t unit_id,
                             uint32_t timestamp, const std::vector<uint8_t>& unit) {
  impl_->Send(type, flags, config_id, unit_id, timestamp, unit);
}
RtpPacket DecodeRtpPacket(const std::vector<uint8_t>& b) {
  if (b.size() < 12 || (b[0] >> 6) != 2) throw std::runtime_error("invalid RTP packet");
  if ((b[0] & 0x0f) != 0 || ((b[0] & 0x10) != 0))
    throw std::runtime_error("RTP CSRC/extensions unsupported");
  if ((b[1] & 0x7f) != 96) throw std::runtime_error("unsupported RTP payload type");
  RtpPacket p;
  p.sequence = (b[2] << 8) | b[3];
  p.timestamp = (uint32_t(b[4]) << 24) | (uint32_t(b[5]) << 16) | (uint32_t(b[6]) << 8) | b[7];
  p.ssrc = (uint32_t(b[8]) << 24) | (uint32_t(b[9]) << 16) | (uint32_t(b[10]) << 8) | b[11];
  p.payload.assign(b.begin() + 12, b.end());
  return p;
}
}  // namespace mlvc::transport
