#include "mlvc/transport/rtp_mlvc.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <random>
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
RtpMlvcReassembler::RtpMlvcReassembler(std::size_t max_unit_bytes,
                                       std::size_t max_pending_units,
                                       std::chrono::milliseconds fragment_timeout)
    : max_unit_bytes_(max_unit_bytes),
      max_pending_units_(max_pending_units),
      fragment_timeout_(fragment_timeout),
      last_cleanup_(std::chrono::steady_clock::now()) {
  if (max_pending_units_ == 0 || fragment_timeout_.count() <= 0)
    throw std::runtime_error("invalid RTP reassembler limits");
}
bool RtpMlvcReassembler::Push(const RtpMlvcFragment& f, std::vector<uint8_t>* unit) {
  if (!unit || f.unit_length > max_unit_bytes_ ||
      uint64_t(f.fragment_offset) + f.payload.size() > f.unit_length)
    throw std::runtime_error("invalid RTP unit");
  const auto now = std::chrono::steady_clock::now();
  if (now - last_cleanup_ >= fragment_timeout_) {
    pending_.clear();
    last_cleanup_ = now;
  }
  auto key = std::make_pair(f.config_id, f.unit_id);
  auto it = pending_.find(key);
  if (it == pending_.end()) {
    if (pending_.size() >= max_pending_units_) {
      auto oldest = pending_.begin();
      pending_.erase(oldest);
    }
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
  struct PendingPacket {
    std::vector<uint8_t> bytes;
    std::chrono::steady_clock::time_point enqueued_at;
  };
  int socket = -1;
  sockaddr_storage address{};
  socklen_t address_length = 0;
  uint16_t sequence = 0;
  uint32_t ssrc = 0;
  std::unique_ptr<UdpPacer> pacer;
  uint64_t pacing_rate_bps = 0;
  std::size_t max_queue_bytes = 0;
  uint64_t max_queue_delay_ms = 0;
  std::size_t queued_bytes = 0;
  std::deque<PendingPacket> queue;
  std::mutex queue_mutex;
  std::condition_variable queue_cv;
  std::exception_ptr send_error;
  bool stopping = false;
  bool sending = false;
  std::thread send_thread;
  uint32_t session_config_id = 0;
  std::vector<uint8_t> session_config;
  uint8_t session_config_flags = 0;
  std::atomic<uint64_t> wire_bytes_sent{0};
  std::atomic<uint64_t> packets_sent{0};
  std::atomic<uint64_t> burst_bytes{0};
  std::atomic<uint64_t> max_burst_bytes_observed{0};
  std::atomic<uint64_t> max_queue_delay_us{0};
  std::atomic<uint64_t> queue_delay_total_us{0};
  std::atomic<uint64_t> queue_delay_samples{0};
  std::atomic<uint64_t> socket_block_us{0};

  ~Impl() {
    try {
      Flush();
    } catch (...) {
    }
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      stopping = true;
    }
    queue_cv.notify_all();
    if (send_thread.joinable()) send_thread.join();
    if (socket >= 0) ::close(socket);
  }

  void Enqueue(std::vector<PendingPacket> packets) {
    std::size_t bytes = 0;
    for (const auto& packet : packets) bytes += packet.bytes.size();
    if (bytes > max_queue_bytes) throw std::runtime_error("RTP unit exceeds send queue bytes");
    std::unique_lock<std::mutex> lock(queue_mutex);
    queue_cv.wait(lock, [this, bytes] {
      if (send_error != nullptr || stopping) return true;
      if (queued_bytes + bytes > max_queue_bytes) return false;
      if (max_queue_delay_ms == 0 || pacing_rate_bps == 0) return true;
      const uint64_t estimated_us = static_cast<uint64_t>(queued_bytes) * 8000000ull /
                                    std::max<uint64_t>(1, pacing_rate_bps);
      return estimated_us <= max_queue_delay_ms * 1000ull;
    });
    if (send_error != nullptr) std::rethrow_exception(send_error);
    if (stopping) throw std::runtime_error("RTP sender is stopping");
    for (auto& packet : packets) {
      queued_bytes += packet.bytes.size();
      queue.push_back(std::move(packet));
    }
    queue_cv.notify_one();
  }

  void SendUnit(RtpUnitType type, uint8_t flags, uint32_t config_id, uint32_t unit_id,
                uint32_t timestamp, const std::vector<uint8_t>& unit) {
    constexpr std::size_t kRtpPayloadBytes = 1200;
    constexpr std::size_t kFragmentHeaderBytes = 24;
    if (unit.size() > UINT32_MAX) throw std::runtime_error("RTP MLVC unit is too large");
    const std::size_t chunk = kRtpPayloadBytes - kFragmentHeaderBytes;
    const std::size_t count = std::max<std::size_t>(1, (unit.size() + chunk - 1) / chunk);
    std::vector<PendingPacket> packets;
    packets.reserve(count);
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
      packets.push_back(PendingPacket{std::move(packet), std::chrono::steady_clock::now()});
    }
    Enqueue(std::move(packets));
  }

  void SetSessionConfig(uint32_t config_id, const std::vector<uint8_t>& config_unit, uint8_t flags) {
    session_config_id = config_id;
    session_config = config_unit;
    session_config_flags = flags;
  }

  void Flush() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    queue_cv.wait(lock, [this] { return (queue.empty() && !sending) || send_error != nullptr; });
    if (send_error != nullptr) std::rethrow_exception(send_error);
  }

  void SendLoop() {
    for (;;) {
      PendingPacket pending;
      {
        std::unique_lock<std::mutex> lock(queue_mutex);
        queue_cv.wait(lock, [this] { return stopping || !queue.empty(); });
        if (queue.empty()) return;
        pending = std::move(queue.front());
        queue.pop_front();
        queued_bytes -= pending.bytes.size();
        sending = true;
      }
      queue_cv.notify_all();
      try {
        const auto queue_delay = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - pending.enqueued_at);
        max_queue_delay_us.store(std::max<uint64_t>(max_queue_delay_us.load(),
                                                    static_cast<uint64_t>(queue_delay.count())));
        queue_delay_total_us.fetch_add(static_cast<uint64_t>(queue_delay.count()));
        queue_delay_samples.fetch_add(1);
        const auto delay = pacer->ConsumeAndGetDelay(pending.bytes.size());
        if (delay.count() > 0) std::this_thread::sleep_for(delay);
        const auto send_begin = std::chrono::steady_clock::now();
        if (::sendto(socket, pending.bytes.data(), pending.bytes.size(), 0,
                     reinterpret_cast<const sockaddr*>(&address), address_length) < 0) {
          throw std::runtime_error("failed to send RTP packet");
        }
        const auto blocked = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - send_begin);
        socket_block_us.fetch_add(static_cast<uint64_t>(blocked.count()));
        wire_bytes_sent.fetch_add(pending.bytes.size());
        packets_sent.fetch_add(1);
        const uint64_t burst = delay.count() > 0
                                   ? pending.bytes.size()
                                   : burst_bytes.fetch_add(pending.bytes.size()) + pending.bytes.size();
        if (delay.count() > 0) burst_bytes.store(pending.bytes.size());
        uint64_t previous = max_burst_bytes_observed.load();
        while (burst > previous &&
               !max_burst_bytes_observed.compare_exchange_weak(previous, burst)) {
        }
        {
          std::lock_guard<std::mutex> lock(queue_mutex);
          sending = false;
        }
        queue_cv.notify_all();
      } catch (...) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        send_error = std::current_exception();
        sending = false;
        stopping = true;
        queue_cv.notify_all();
        return;
      }
    }
  }
};
RtpMlvcSender::RtpMlvcSender(const std::string& host, uint16_t port, uint32_t ssrc,
                             uint64_t pacing_rate_bps, std::size_t max_burst_bytes,
                             std::size_t max_queue_bytes, uint64_t max_queue_delay_ms) {
  impl_ = std::make_unique<Impl>();
  if (ssrc == 0) {
    std::random_device rd;
    ssrc = (static_cast<uint32_t>(rd()) << 16) ^ static_cast<uint32_t>(rd());
    if (ssrc == 0) ssrc = 1;
  }
  impl_->ssrc = ssrc;
  impl_->pacer = std::make_unique<UdpPacer>(pacing_rate_bps, max_burst_bytes);
  impl_->pacing_rate_bps = pacing_rate_bps;
  impl_->max_queue_bytes = max_queue_bytes;
  impl_->max_queue_delay_ms = max_queue_delay_ms;
  if (max_queue_bytes == 0) throw std::runtime_error("RTP send queue bytes must be positive");
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
  impl_->send_thread = std::thread([impl = impl_.get()] { impl->SendLoop(); });
}
RtpMlvcSender::~RtpMlvcSender() = default;
void RtpMlvcSender::SendUnit(RtpUnitType type, uint8_t flags, uint32_t config_id, uint32_t unit_id,
                             uint32_t timestamp, const std::vector<uint8_t>& unit) {
  impl_->SendUnit(type, flags, config_id, unit_id, timestamp, unit);
}
void RtpMlvcSender::Flush() { impl_->Flush(); }
void RtpMlvcSender::Close() { impl_->Flush(); }
void RtpMlvcSender::SetSessionConfig(uint32_t config_id, const std::vector<uint8_t>& config_unit,
                                     uint8_t flags) {
  impl_->SetSessionConfig(config_id, config_unit, flags);
  impl_->SendUnit(RtpUnitType::kScu, flags, config_id, 0, 0, config_unit);
}
void RtpMlvcSender::ResendSessionConfig(uint32_t timestamp) {
  if (impl_->session_config.empty()) throw std::runtime_error("RTP session config is not set");
  impl_->SendUnit(RtpUnitType::kScu, impl_->session_config_flags, impl_->session_config_id, 0,
                  timestamp, impl_->session_config);
}
RtpTransportStats RtpMlvcSender::Stats() const {
  RtpTransportStats stats;
  stats.wire_bytes = impl_->wire_bytes_sent.load();
  stats.packets = impl_->packets_sent.load();
  stats.max_burst_bytes = impl_->max_burst_bytes_observed.load();
  stats.max_queue_delay_us = impl_->max_queue_delay_us.load();
  const uint64_t samples = impl_->queue_delay_samples.load();
  stats.average_queue_delay_us = samples == 0 ? 0 : impl_->queue_delay_total_us.load() / samples;
  stats.socket_block_us = impl_->socket_block_us.load();
  return stats;
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

std::vector<uint8_t> EncodeRtcpConfigRequest(const RtcpConfigRequest& request) {
  std::vector<uint8_t> packet = {0x80, 204, 0, 2, 'M', 'L', 'C', '1'};
  Put(packet, request.ssrc);
  Put(packet, request.config_id);
  return packet;
}

RtcpConfigRequest DecodeRtcpConfigRequest(const std::vector<uint8_t>& packet) {
  if (packet.size() != 16 || packet[0] != 0x80 || packet[1] != 204 || packet[2] != 0 ||
      packet[3] != 2 || packet[4] != 'M' || packet[5] != 'L' || packet[6] != 'C' || packet[7] != '1')
    throw std::runtime_error("invalid MLVC RTCP config request");
  std::size_t p = 8;
  return RtcpConfigRequest{Get(packet, p), Get(packet, p)};
}
}  // namespace mlvc::transport
