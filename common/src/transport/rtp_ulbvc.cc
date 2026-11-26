#include "ulbvc/transport/rtp_ulbvc.h"

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
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>

#include "ulbvc/transport/rtcp_session.h"
#include "ulbvc/transport/ulbvc_media_unit.h"
namespace ulbvc::transport {
namespace {
constexpr std::size_t kIpv4UdpOverheadBytes = 28;
// Bound interval-map metadata separately from media bytes. Tiny attacker-
// controlled fragments can otherwise consume much more memory than unit data.
constexpr std::size_t kMaxPendingFragmentRanges = 131072;
constexpr std::size_t kMaxPreStartFragmentBytes = 1024u * 1024u;

void Put(std::vector<uint8_t>& o, uint32_t v) {
  for (int i = 3; i >= 0; --i) o.push_back((v >> (i * 8)) & 255);
}
uint32_t Get(const std::vector<uint8_t>& b, size_t& p) {
  if (p + 4 > b.size()) throw std::runtime_error("truncated RTP ULBVC header");
  uint32_t v = 0;
  for (int i = 0; i < 4; ++i) v = (v << 8) | b[p++];
  return v;
}

bool IsValidUnitType(uint8_t type) { return type != 0; }

std::size_t MaxUnitBytesForType(RtpUnitType type) {
  if (type == RtpUnitType::kScu) return kUlbvcScuMaxBytes;
  if (type == RtpUnitType::kEos) return kUlbvcMediaUnitCommonHeaderBytes;
  return kUlbvcMediaUnitMaxBytes;
}

void ValidateUnitLengthForType(RtpUnitType type, uint32_t length) {
  if (length < kUlbvcMediaUnitCommonHeaderBytes || length > MaxUnitBytesForType(type) ||
      (type == RtpUnitType::kEos && length != kUlbvcMediaUnitCommonHeaderBytes)) {
    throw std::runtime_error("RTP ULBVC unit length exceeds the type-specific limit");
  }
}

void ValidateFragmentIdentity(RtpUnitType type, uint32_t config_id, uint32_t unit_id) {
  if (type != RtpUnitType::kScu && type != RtpUnitType::kEfu &&
      type != RtpUnitType::kEos) {
    return;
  }
  if (config_id == 0) throw std::runtime_error("RTP ULBVC fragment has a zero config ID");
  if (type == RtpUnitType::kScu && unit_id != config_id) {
    throw std::runtime_error("RTP ULBVC SCU unit ID must equal its config ID");
  }
  if (type == RtpUnitType::kEfu && unit_id == kUlbvcNoReference) {
    throw std::runtime_error("RTP ULBVC EFU has an invalid unit ID");
  }
  if (type == RtpUnitType::kEos && unit_id != kUlbvcNoReference &&
      unit_id > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("RTP ULBVC EOS frame boundary is outside the application range");
  }
}
}  // namespace
std::vector<uint8_t> EncodeRtpUlbvcFragment(const RtpUlbvcFragment& f) {
  const uint8_t packet_flags = f.flags;
  if (!IsValidUnitType(static_cast<uint8_t>(f.unit_type)) || f.payload.empty() ||
      (packet_flags & (0x1fu | kRtpUlbvcExtension)) != 0 ||
      uint64_t(f.fragment_offset) + f.payload.size() > f.unit_length ||
      ((packet_flags & kRtpUlbvcStart) != 0) != (f.fragment_offset == 0) ||
      ((packet_flags & kRtpUlbvcEnd) != 0) !=
          (uint64_t(f.fragment_offset) + f.payload.size() == f.unit_length))
    throw std::runtime_error("invalid RTP ULBVC fragment");
  ValidateUnitLengthForType(f.unit_type, f.unit_length);
  ValidateFragmentIdentity(f.unit_type, f.config_id, f.unit_id);
  std::vector<uint8_t> o;
  o.reserve(24 + f.payload.size());
  o.push_back(1);
  o.push_back(static_cast<uint8_t>(f.unit_type));
  o.push_back(packet_flags);
  o.push_back(0);
  o.push_back(0);
  o.push_back(24);
  o.push_back(0);
  o.push_back(0);
  Put(o, f.config_id);
  Put(o, f.unit_id);
  Put(o, f.fragment_offset);
  Put(o, f.unit_length);
  o.insert(o.end(), f.payload.begin(), f.payload.end());
  return o;
}
RtpUlbvcFragment DecodeRtpUlbvcFragment(const std::vector<uint8_t>& b) {
  if (b.size() < 24 || b[0] != 1)
    throw std::runtime_error("invalid RTP ULBVC header");
  size_t p = 1;
  uint8_t type = b[p++];
  uint8_t flags = b[p++];
  if (!IsValidUnitType(type) || (flags & 0x1fu) != 0 || b[p++] != 0)
    throw std::runtime_error("invalid RTP ULBVC header");
  const uint16_t descriptor_length = static_cast<uint16_t>(b[p] << 8 | b[p + 1]);
  p += 2;
  if (descriptor_length < 24 || (descriptor_length & 3u) != 0 ||
      descriptor_length > b.size() || b[p] != 0 || b[p + 1] != 0 ||
      (((flags & kRtpUlbvcExtension) == 0) != (descriptor_length == 24))) {
    throw std::runtime_error("invalid RTP ULBVC descriptor");
  }
  p += 2;
  // Version 1 does not interpret descriptor extensions.  Once their bounded,
  // aligned length is checked, strip X before handing the fragment to the
  // version-1 reassembler.
  RtpUlbvcFragment f{(RtpUnitType)type,
                    static_cast<uint8_t>(flags & ~kRtpUlbvcExtension), 0, 0, 0, 0, {}};
  f.config_id = Get(b, p);
  f.unit_id = Get(b, p);
  f.fragment_offset = Get(b, p);
  f.unit_length = Get(b, p);
  if (f.unit_length < kUlbvcMediaUnitCommonHeaderBytes ||
      uint64_t(f.fragment_offset) + (b.size() - descriptor_length) > f.unit_length ||
      ((flags & kRtpUlbvcStart) != 0) != (f.fragment_offset == 0) ||
      ((flags & kRtpUlbvcEnd) != 0) !=
          (uint64_t(f.fragment_offset) + (b.size() - descriptor_length) == f.unit_length) ||
      b.size() == descriptor_length)
    throw std::runtime_error("invalid RTP ULBVC payload");
  ValidateUnitLengthForType(f.unit_type, f.unit_length);
  ValidateFragmentIdentity(f.unit_type, f.config_id, f.unit_id);
  f.payload.assign(b.begin() + descriptor_length, b.end());
  return f;
}
RtpUlbvcReassembler::RtpUlbvcReassembler(std::size_t max_unit_bytes,
                                       std::size_t max_pending_units,
                                       std::chrono::milliseconds fragment_timeout)
    : max_unit_bytes_(max_unit_bytes),
      max_pending_units_(max_pending_units),
      max_completed_units_(max_pending_units),
      max_pending_bytes_(max_unit_bytes),
      fragment_timeout_(fragment_timeout),
      last_cleanup_(std::chrono::steady_clock::now()) {
  if (max_unit_bytes_ == 0 || max_pending_units_ == 0 || fragment_timeout_.count() <= 0)
    throw std::runtime_error("invalid RTP reassembler limits");
}
bool RtpUlbvcReassembler::Push(const RtpUlbvcFragment& f, std::vector<uint8_t>* unit) {
  if (!unit || !IsValidUnitType(static_cast<uint8_t>(f.unit_type)) || f.unit_length == 0 ||
      f.payload.empty() || f.unit_length > max_unit_bytes_ ||
      (f.flags & 0x1fu) != 0 ||
      uint64_t(f.fragment_offset) + f.payload.size() > f.unit_length ||
      ((f.flags & kRtpUlbvcStart) != 0) != (f.fragment_offset == 0) ||
      ((f.flags & kRtpUlbvcEnd) != 0) !=
          (uint64_t(f.fragment_offset) + f.payload.size() == f.unit_length))
    throw std::runtime_error("invalid RTP unit");
  ValidateUnitLengthForType(f.unit_type, f.unit_length);
  ValidateFragmentIdentity(f.unit_type, f.config_id, f.unit_id);
  const auto now = std::chrono::steady_clock::now();
  const auto erase_pending = [this](auto it) {
    pending_bytes_ -= it->second.length;
    fragment_range_count_ -= it->second.fragment_ranges.size();
    pre_start_fragment_bytes_ -= it->second.pre_start_bytes;
    return pending_.erase(it);
  };
  if (now - last_cleanup_ >= fragment_timeout_) {
    for (auto it = pending_.begin(); it != pending_.end();) {
      if (now - it->second.last_updated >= fragment_timeout_) {
        it = erase_pending(it);
      } else {
        ++it;
      }
    }
    last_cleanup_ = now;
  }
  auto key = std::make_tuple(f.ssrc, f.timestamp, static_cast<uint8_t>(f.unit_type),
                             f.config_id, f.unit_id);
  const auto completed = completed_.find(key);
  if (completed != completed_.end()) {
    const uint64_t fragment_end = static_cast<uint64_t>(f.fragment_offset) + f.payload.size();
    if (fragment_end > completed->second.size() ||
        !std::equal(f.payload.begin(), f.payload.end(),
                    completed->second.begin() + f.fragment_offset)) {
      throw std::runtime_error("conflicting duplicate RTP media unit");
    }
    return false;
  }
  auto it = pending_.find(key);
  if (it == pending_.end()) {
    if (f.unit_length > max_pending_bytes_)
      throw std::runtime_error("RTP unit exceeds aggregate reassembly byte limit");
    while (completed_bytes_ > max_pending_bytes_ - f.unit_length && !completed_.empty()) {
      completed_bytes_ -= completed_.begin()->second.size();
      completed_.erase(completed_.begin());
    }
    while (!pending_.empty() &&
           (pending_.size() >= max_pending_units_ ||
            f.unit_length > max_pending_bytes_ - pending_bytes_ - completed_bytes_)) {
      auto oldest = std::min_element(pending_.begin(), pending_.end(),
                                     [](const auto& left, const auto& right) {
                                       return left.second.last_updated < right.second.last_updated;
                                     });
      erase_pending(oldest);
    }
    Pending p{f.unit_type, f.unit_length, {}, {}, {}, 0, 0, now};
    if ((f.flags & kRtpUlbvcStart) != 0) p.data.resize(f.unit_length);
    it = pending_.emplace(key, std::move(p)).first;
    pending_bytes_ += f.unit_length;
  }
  auto& p = it->second;
  if (p.type != f.unit_type || p.length != f.unit_length)
    throw std::runtime_error("conflicting RTP fragment");
  const uint32_t fragment_end =
      static_cast<uint32_t>(f.fragment_offset + static_cast<uint32_t>(f.payload.size()));
  auto next_range = p.fragment_ranges.lower_bound(f.fragment_offset);
  const auto matches_existing_payload = [&] {
    if (p.data.empty()) {
      const auto existing = p.pre_start_fragments.find(f.fragment_offset);
      return existing != p.pre_start_fragments.end() && existing->second == f.payload;
    }
    return std::equal(f.payload.begin(), f.payload.end(), p.data.begin() + f.fragment_offset);
  };
  if (next_range != p.fragment_ranges.end() && next_range->first < fragment_end) {
    if (f.fragment_offset == next_range->first && fragment_end == next_range->second) {
      if (!matches_existing_payload())
        throw std::runtime_error("conflicting duplicate RTP fragment");
      return false;
    }
    throw std::runtime_error("overlapping RTP fragments");
  }
  if (next_range != p.fragment_ranges.begin()) {
    const auto previous_range = std::prev(next_range);
    if (previous_range->second > f.fragment_offset) {
      if (f.fragment_offset == previous_range->first && fragment_end == previous_range->second) {
        if (!matches_existing_payload())
          throw std::runtime_error("conflicting duplicate RTP fragment");
        return false;
      }
      throw std::runtime_error("overlapping RTP fragments");
    }
  }
  if (fragment_range_count_ >= kMaxPendingFragmentRanges &&
      p.count + f.payload.size() < p.length) {
    erase_pending(it);
    throw std::runtime_error("RTP reassembly fragment-range limit exceeded");
  }
  if (p.data.empty() && (f.flags & kRtpUlbvcStart) != 0) {
    p.data.resize(p.length);
    for (const auto& [offset, payload] : p.pre_start_fragments) {
      std::copy(payload.begin(), payload.end(), p.data.begin() + offset);
    }
    pre_start_fragment_bytes_ -= p.pre_start_bytes;
    p.pre_start_bytes = 0;
    p.pre_start_fragments.clear();
  }
  if (p.data.empty()) {
    const std::size_t pending_budget =
        max_pending_bytes_ - pending_bytes_ - completed_bytes_;
    if (f.payload.size() > kMaxPreStartFragmentBytes - pre_start_fragment_bytes_ ||
        f.payload.size() > pending_budget) {
      erase_pending(it);
      return false;
    }
    p.pre_start_fragments.emplace(f.fragment_offset, f.payload);
    p.pre_start_bytes += f.payload.size();
    pre_start_fragment_bytes_ += f.payload.size();
  } else {
    std::copy(f.payload.begin(), f.payload.end(), p.data.begin() + f.fragment_offset);
  }
  p.count += f.payload.size();
  if (!f.payload.empty()) {
    p.fragment_ranges.emplace(f.fragment_offset, fragment_end);
    ++fragment_range_count_;
    p.last_updated = now;
  }
  if (p.count == p.length) {
    *unit = std::move(p.data);
    pending_bytes_ -= p.length;
    fragment_range_count_ -= p.fragment_ranges.size();
    completed_bytes_ += unit->size();
    completed_[key] = *unit;
    while ((completed_bytes_ + pending_bytes_ > max_pending_bytes_ ||
            completed_.size() > max_completed_units_) &&
           !completed_.empty()) {
      completed_bytes_ -= completed_.begin()->second.size();
      completed_.erase(completed_.begin());
    }
    pending_.erase(it);
    return true;
  }
  return false;
}
void RtpUlbvcReassembler::Reset() {
  pending_.clear();
  pending_bytes_ = 0;
  fragment_range_count_ = 0;
  pre_start_fragment_bytes_ = 0;
  completed_.clear();
  completed_bytes_ = 0;
}
struct RtpUlbvcSender::Impl {
  struct PendingPacket {
    std::vector<uint8_t> bytes;
    std::chrono::steady_clock::time_point enqueued_at;
    // Only the first fragment of a newly queued ULBVC media unit carries this
    // value. Retransmissions leave it at zero so a media unit is counted once.
    std::size_t media_unit_bytes = 0;
    bool retransmission = false;
  };
  int socket = -1;
  sockaddr_storage address{};
  socklen_t address_length = 0;
  uint16_t sequence = 0;
  uint32_t ssrc = 0;
  uint8_t payload_type = 96;
  std::unique_ptr<UdpPacer> pacer;
  std::unique_ptr<RtcpUdpEndpoint> rtcp_endpoint;
  std::chrono::milliseconds rtcp_interval{5000};
  std::atomic<bool> rtcp_stopping{false};
  std::thread rtcp_thread;
  std::atomic<uint64_t> pacing_rate_bps{0};
  uint64_t applied_pacing_rate = 0;
  std::size_t max_queue_bytes = 0;
  uint64_t max_queue_delay_ms = 0;
  std::size_t queued_bytes = 0;
  std::mutex packetize_mutex;
  std::deque<PendingPacket> queue;
  std::mutex queue_mutex;
  std::condition_variable queue_cv;
  std::exception_ptr send_error;
  bool stopping = false;
  bool sending = false;
  std::chrono::steady_clock::time_point sending_since{};
  std::thread send_thread;
  uint32_t session_config_id = 0;
  uint32_t pending_random_access_config_id = 0;
  std::vector<uint8_t> session_config;
  uint8_t session_config_flags = 0;
  std::atomic<uint64_t> wire_bytes_sent{0};
  std::atomic<uint64_t> serialized_wire_bytes{0};
  std::atomic<uint64_t> media_unit_bytes_sent{0};
  std::atomic<uint64_t> packets_sent{0};
  std::atomic<uint64_t> burst_bytes{0};
  std::atomic<uint64_t> max_burst_bytes_observed{0};
  std::atomic<uint64_t> max_queue_delay_us{0};
  std::atomic<uint64_t> queue_delay_total_us{0};
  std::atomic<uint64_t> queue_delay_samples{0};
  std::atomic<uint64_t> socket_block_us{0};
  std::atomic<uint64_t> rtp_octets_sent{0};
  std::atomic<uint32_t> last_rtp_timestamp{0};
  std::atomic<uint64_t> rtcp_packets_sent{0};
  std::atomic<uint64_t> rtcp_packets_received{0};
  std::atomic<uint64_t> retransmitted_wire_bytes{0};
  std::atomic<bool> rate_feedback_enabled{false};
  std::atomic<uint32_t> feedback_interval_ms{500};
  uint64_t rate_session_id = 0;
  mutable std::mutex feedback_mutex;
  RateFeedbackSnapshot feedback_snapshot;
  std::deque<uint64_t> retired_receiver_instances;
  std::deque<std::pair<uint32_t, std::chrono::steady_clock::time_point>> sender_report_times;
  std::atomic<uint64_t> random_access_requests{0};
  std::mutex control_mutex;
  std::deque<UlbvcControlMessage> control_queue;
  std::set<uint32_t> seen_control_transactions;
  std::deque<uint32_t> control_transaction_order;
  std::map<uint32_t, std::vector<uint8_t>> control_response_cache;
  std::deque<std::vector<uint8_t>> control_responses;
  std::atomic<uint32_t> remote_control_ssrc{0};
  std::mutex packet_history_mutex;
  std::map<uint16_t, std::vector<uint8_t>> packet_history;
  std::map<uint16_t, std::chrono::steady_clock::time_point> packet_history_times;
  std::deque<uint16_t> packet_history_order;
  std::map<uint16_t, std::chrono::steady_clock::time_point> last_retransmit_time;
  static constexpr std::size_t kPacketHistoryLimit = 2048;

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
    rtcp_stopping.store(true);
    if (rtcp_thread.joinable()) rtcp_thread.join();
    if (rtcp_endpoint) rtcp_endpoint->Close();
    if (socket >= 0) ::close(socket);
  }

  void ReserveQueueBytes(std::size_t bytes) {
    if (bytes > max_queue_bytes) throw std::runtime_error("RTP unit exceeds send queue bytes");
    std::unique_lock<std::mutex> lock(queue_mutex);
    queue_cv.wait(lock, [this, bytes] {
      if (send_error != nullptr || stopping) return true;
      if (queued_bytes > max_queue_bytes || bytes > max_queue_bytes - queued_bytes) return false;
      if (queued_bytes == 0) return true;
      if (max_queue_delay_ms == 0 || pacing_rate_bps == 0) return true;
      const long double estimated_us = static_cast<long double>(queued_bytes + bytes) *
                                       8000000.0L / pacing_rate_bps;
      return estimated_us <= static_cast<long double>(max_queue_delay_ms) * 1000.0L;
    });
    if (send_error != nullptr) std::rethrow_exception(send_error);
    if (stopping) throw std::runtime_error("RTP sender is stopping");
    queued_bytes += bytes;
  }

  void ReleaseQueueReservation(std::size_t bytes) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    if (bytes <= queued_bytes) queued_bytes -= bytes;
    queue_cv.notify_all();
  }

  void RememberRtpPacket(const std::vector<uint8_t>& packet) {
    if (packet.size() < 12) return;
    const uint16_t packet_sequence = static_cast<uint16_t>(
        (static_cast<uint16_t>(packet[2]) << 8) | packet[3]);
    std::lock_guard<std::mutex> lock(packet_history_mutex);
    const auto existing = packet_history.find(packet_sequence);
    if (existing == packet_history.end()) packet_history_order.push_back(packet_sequence);
    packet_history[packet_sequence] = packet;
    packet_history_times[packet_sequence] = std::chrono::steady_clock::now();
    while (packet_history_order.size() > kPacketHistoryLimit) {
      const uint16_t expired = packet_history_order.front();
      packet_history_order.pop_front();
      packet_history.erase(expired);
      packet_history_times.erase(expired);
      last_retransmit_time.erase(expired);
    }
  }

  void QueueRetransmissions(const std::vector<uint16_t>& requested_sequences) {
    const auto now = std::chrono::steady_clock::now();
    std::vector<PendingPacket> packets;
    packets.reserve(std::min<std::size_t>(requested_sequences.size(), 64));
    {
      std::lock_guard<std::mutex> lock(packet_history_mutex);
      for (uint16_t requested : requested_sequences) {
        if (packets.size() >= 64) break;
        const auto history = packet_history.find(requested);
        if (history == packet_history.end()) continue;
        const auto original = packet_history_times.find(requested);
        if (original == packet_history_times.end() ||
            now - original->second > std::chrono::milliseconds(std::min<uint64_t>(
                                         2000, std::max<uint64_t>(200, max_queue_delay_ms + 500))))
          continue;
        const auto last = last_retransmit_time.find(requested);
        if (last != last_retransmit_time.end() &&
            now - last->second < std::chrono::milliseconds(100)) {
          continue;
        }
        last_retransmit_time[requested] = now;
        packets.push_back(PendingPacket{history->second, now, 0, true});
      }
    }
    if (packets.empty()) return;
    std::size_t reservation = 0;
    for (const auto& packet : packets) {
      const std::size_t packet_bytes = packet.bytes.size() + kIpv4UdpOverheadBytes;
      if (reservation > std::numeric_limits<std::size_t>::max() - packet_bytes) return;
      reservation += packet_bytes;
    }
    std::lock_guard<std::mutex> lock(queue_mutex);
    if (stopping || send_error != nullptr || reservation > max_queue_bytes ||
        queued_bytes > max_queue_bytes - reservation) {
      return;
    }
    if (pacing_rate_bps > 0 && max_queue_delay_ms > 0) {
      const long double estimated_us = static_cast<long double>(queued_bytes + reservation) *
                                       8000000.0L / pacing_rate_bps;
      if (estimated_us > static_cast<long double>(max_queue_delay_ms) * 1000.0L) return;
    }
    queued_bytes += reservation;
    for (auto& packet : packets) queue.push_back(std::move(packet));
    queue_cv.notify_one();
  }

  void HandleRtcpPacket(const std::vector<uint8_t>& packet) {
    const RtcpCompoundContents compound = DecodeRtcpCompound(packet);
    uint32_t receiver_ssrc = compound.receiver_report.has_value()
                                 ? compound.receiver_report->sender_ssrc
                                 : 0;
    if (receiver_ssrc == 0 && !compound.plis.empty()) receiver_ssrc = compound.plis.front().sender_ssrc;
    if (receiver_ssrc == 0 && !compound.firs.empty()) receiver_ssrc = compound.firs.front().sender_ssrc;
    if (receiver_ssrc == 0) return;
    if (!rtcp_endpoint->LastPeerMatchesRemoteHost()) return;
    std::optional<RateFeedback> rate_feedback;
    for (const auto& control : compound.ulbvc_controls) {
      if (control.type != UlbvcControlType::kStatus || control.control_ssrc != receiver_ssrc ||
          control.media_ssrc != ssrc)
        continue;
      for (const auto& tlv : control.tlvs) {
        if (tlv.type != kRateFeedbackTlv) continue;
        if (rate_feedback) return;
        auto decoded = DecodeRateFeedback(tlv);
        if (!rate_feedback_enabled.load() || decoded.session_id != rate_session_id) return;
        rate_feedback = decoded;
      }
    }
    uint32_t expected_receiver_ssrc = remote_control_ssrc.load();
    if (expected_receiver_ssrc == 0) {
      remote_control_ssrc.compare_exchange_strong(expected_receiver_ssrc, receiver_ssrc);
      expected_receiver_ssrc = remote_control_ssrc.load();
    }
    bool changed_instance = false;
    if (rate_feedback) {
      std::lock_guard<std::mutex> lock(feedback_mutex);
      changed_instance =
          feedback_snapshot.valid &&
          rate_feedback->receiver_instance != feedback_snapshot.feedback.receiver_instance;
    }
    if (receiver_ssrc != expected_receiver_ssrc || changed_instance) {
      if (!rate_feedback) return;
      std::lock_guard<std::mutex> lock(feedback_mutex);
      if (std::find(retired_receiver_instances.begin(), retired_receiver_instances.end(),
                    rate_feedback->receiver_instance) != retired_receiver_instances.end())
        return;
      if (feedback_snapshot.valid) {
        retired_receiver_instances.push_back(feedback_snapshot.feedback.receiver_instance);
        while (retired_receiver_instances.size() > 32) retired_receiver_instances.pop_front();
      }
      feedback_snapshot = {};
      remote_control_ssrc.store(receiver_ssrc);
      std::lock_guard<std::mutex> control_lock(control_mutex);
      seen_control_transactions.clear();
      control_transaction_order.clear();
      control_queue.clear();
      control_response_cache.clear();
      control_responses.clear();
    }
    if (compound.receiver_report.has_value() &&
        std::none_of(compound.receiver_report->reports.begin(),
                     compound.receiver_report->reports.end(), [this](const auto& report) {
                       return report.source_ssrc == ssrc;
                     }) && compound.plis.empty() && compound.firs.empty()) {
      return;
    }
    rtcp_endpoint->AcceptLastPeer();
    ++rtcp_packets_received;
    if (rate_feedback) {
      std::lock_guard<std::mutex> lock(feedback_mutex);
      if (feedback_snapshot.valid) {
        const auto& previous = feedback_snapshot.feedback;
        const uint32_t delta = rate_feedback->sequence - previous.sequence;
        if (rate_feedback->receiver_instance != previous.receiver_instance || !delta ||
            delta >= 0x80000000u || rate_feedback->sample_time_ms <= previous.sample_time_ms ||
            rate_feedback->received_packets < previous.received_packets ||
            rate_feedback->received_wire_bytes < previous.received_wire_bytes ||
            rate_feedback->missing_packets < previous.missing_packets ||
            rate_feedback->recovered_packets < previous.recovered_packets)
          return;
      }
      feedback_snapshot.valid = true;
      feedback_snapshot.feedback = *rate_feedback;
      feedback_snapshot.received_at = std::chrono::steady_clock::now();
      if (compound.receiver_report)
        for (const auto& report : compound.receiver_report->reports) {
          if (report.source_ssrc != ssrc || !report.last_sender_report) continue;
          for (const auto& sent : sender_report_times) {
            if (sent.first != report.last_sender_report) continue;
            const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                                     feedback_snapshot.received_at - sent.second)
                                     .count();
            const int64_t dlsr_us =
                int64_t(report.delay_since_last_sender_report) * 1000000 / 65536;
            if (elapsed >= dlsr_us && elapsed - dlsr_us <= 10000000)
              feedback_snapshot.rtt_us = static_cast<uint32_t>(elapsed - dlsr_us);
          }
        }
    }
    {
      std::lock_guard<std::mutex> lock(control_mutex);
      for (const auto& control : compound.ulbvc_controls) {
        if (control.type == UlbvcControlType::kCommand && control.control_ssrc == receiver_ssrc &&
            (control.media_ssrc == 0 || control.media_ssrc == ssrc) &&
            control.transaction_id != 0 &&
            seen_control_transactions.insert(control.transaction_id).second) {
          control_queue.push_back(control);
          control_transaction_order.push_back(control.transaction_id);
          while (control_transaction_order.size() > 1024) {
            seen_control_transactions.erase(control_transaction_order.front());
            control_response_cache.erase(control_transaction_order.front());
            control_transaction_order.pop_front();
          }
        } else if (control.type == UlbvcControlType::kCommand &&
                   control.control_ssrc == receiver_ssrc &&
                   (control.media_ssrc == 0 || control.media_ssrc == ssrc)) {
          const auto cached = control_response_cache.find(control.transaction_id);
          if (cached != control_response_cache.end() && control_responses.size() < 32)
            control_responses.push_back(cached->second);
        }
      }
      while (control_queue.size() > 32) control_queue.pop_front();
    }
    for (const auto& pli : compound.plis) {
      if (pli.sender_ssrc == receiver_ssrc && pli.media_ssrc == ssrc)
        random_access_requests.fetch_add(1);
    }
    for (const auto& fir : compound.firs) {
      if (fir.sender_ssrc == receiver_ssrc && fir.media_ssrc == ssrc)
        random_access_requests.fetch_add(1);
    }
    std::vector<uint16_t> requested_sequences;
    for (const auto& nack : compound.nacks) {
      if (nack.sender_ssrc != receiver_ssrc || nack.media_ssrc != ssrc) continue;
      requested_sequences.push_back(nack.pid);
      for (uint8_t bit = 0; bit < 16; ++bit) {
        if ((nack.blp & (1u << bit)) != 0) {
          requested_sequences.push_back(static_cast<uint16_t>(nack.pid + bit + 1u));
        }
      }
    }
    if (!requested_sequences.empty()) QueueRetransmissions(requested_sequences);
  }

  void SendUnitLocked(RtpUnitType type, uint8_t flags, uint32_t config_id, uint32_t unit_id,
                      uint32_t timestamp, const std::vector<uint8_t>& unit) {
    (void)flags;  // Unit flags live in the serialized media-unit common header.
    if (type != RtpUnitType::kScu && type != RtpUnitType::kEfu &&
        type != RtpUnitType::kEos) {
      throw std::runtime_error("RTP sender cannot emit an undefined ULBVC media unit type");
    }
    constexpr std::size_t kRtpPacketBytes = 1200;
    constexpr std::size_t kFragmentHeaderBytes = 24;
    constexpr std::size_t kMaxUnitBytes = 128u * 1024u * 1024u;
    if (unit.empty() || unit.size() > kMaxUnitBytes || unit.size() > UINT32_MAX)
      throw std::runtime_error("RTP ULBVC unit is empty or too large");
    const auto media_header = ParseMediaUnitHeader(unit);
    if (media_header.unit_type != type || media_header.config_id != config_id ||
        media_header.unit_id != unit_id || media_header.unit_length != unit.size()) {
      throw std::runtime_error("RTP ULBVC descriptor metadata does not match media unit");
    }
    if (type == RtpUnitType::kScu) {
      (void)ParseScu(unit);
    } else if (session_config_id == 0 || config_id != session_config_id) {
      throw std::runtime_error("RTP media unit is sent before its active SCU");
    }
    ValidateMediaUnit(unit);
    if (pending_random_access_config_id != 0) {
      if (type == RtpUnitType::kScu) {
        if (config_id != pending_random_access_config_id || unit != session_config) {
          throw std::runtime_error("RTP configuration refresh does not match the active SCU");
        }
      } else if (type != RtpUnitType::kEfu) {
        throw std::runtime_error("RTP configuration switch requires a random-access EFU next");
      } else {
        const auto efu = ParseEfu(unit);
        if (efu.config_id != pending_random_access_config_id || efu.frame_type != 0 ||
            (efu.unit_flags & (kEfuRandomAccess | kEfuResetReference)) !=
                (kEfuRandomAccess | kEfuResetReference)) {
          throw std::runtime_error("RTP configuration switch requires a random-access EFU next");
        }
      }
    }
    const std::size_t chunk = kRtpPacketBytes - kFragmentHeaderBytes;
    const std::size_t count = std::max<std::size_t>(1, (unit.size() + chunk - 1) / chunk);
    const std::size_t reserved_bytes = unit.size() + count * (36 + kIpv4UdpOverheadBytes);
    if (reserved_bytes > max_queue_bytes)
      throw std::runtime_error("RTP unit exceeds send queue bytes");
    std::vector<PendingPacket> packets;
    packets.reserve(count);
      for (std::size_t i = 0; i < count; ++i) {
        const std::size_t offset = i * chunk;
        const std::size_t size = std::min(chunk, unit.size() - offset);
        const uint8_t packet_flags =
            static_cast<uint8_t>((offset == 0 ? kRtpUlbvcStart : 0) |
                                 (offset + size == unit.size() ? kRtpUlbvcEnd : 0));
        RtpUlbvcFragment fragment{
            type,
            packet_flags,
            config_id,
            unit_id,
            static_cast<uint32_t>(offset),
            static_cast<uint32_t>(unit.size()),
            std::vector<uint8_t>(unit.begin() + offset, unit.begin() + offset + size)};
        const auto payload = EncodeRtpUlbvcFragment(fragment);
        std::vector<uint8_t> packet(12 + payload.size());
        packet[0] = 0x80;
        packet[1] = static_cast<uint8_t>(
            payload_type | (type == RtpUnitType::kEfu && (packet_flags & kRtpUlbvcEnd) != 0 ? 0x80 : 0));
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
        packets.push_back(PendingPacket{std::move(packet), std::chrono::steady_clock::now(),
                                        i == 0 ? unit.size() : 0});
      }
    // Stream complete units through a bounded packet queue. A large I-frame
    // can take longer to serialize without allocating an unbounded queue or
    // failing merely because its full size exceeds the normal delay budget.
    for (auto& packet : packets) {
      const auto bytes = packet.bytes.size() + kIpv4UdpOverheadBytes;
      ReserveQueueBytes(bytes);
      try {
        std::lock_guard<std::mutex> lock(queue_mutex);
        if (send_error) std::rethrow_exception(send_error);
        if (stopping) throw std::runtime_error("RTP sender stopped during unit admission");
        packet.enqueued_at = std::chrono::steady_clock::now();
        queue.push_back(std::move(packet));
      } catch (...) {
        ReleaseQueueReservation(bytes);
        throw;
      }
      queue_cv.notify_one();
    }
    serialized_wire_bytes.fetch_add(reserved_bytes);
    if (pending_random_access_config_id != 0 && type == RtpUnitType::kEfu) {
      pending_random_access_config_id = 0;
    }
  }

  void SendUnit(RtpUnitType type, uint8_t flags, uint32_t config_id, uint32_t unit_id,
                uint32_t timestamp, const std::vector<uint8_t>& unit) {
    std::lock_guard<std::mutex> packetize_lock(packetize_mutex);
    SendUnitLocked(type, flags, config_id, unit_id, timestamp, unit);
  }

  void SetSessionConfig(uint32_t config_id, const std::vector<uint8_t>& config_unit, uint8_t flags,
                        uint32_t timestamp) {
    std::lock_guard<std::mutex> packetize_lock(packetize_mutex);
    const auto parsed = ParseScu(config_unit);
    if (parsed.config_id != config_id) {
      throw std::runtime_error("RTP session config ID does not match SCU");
    }
    if (config_id == session_config_id && !session_config.empty() &&
        session_config != config_unit) {
      throw std::runtime_error("RTP configuration ID was reused with different SCU bytes");
    }
    const UlbvcSerial32Order config_order =
        CompareUlbvcSerial32(config_id, session_config_id);
    if (session_config_id != 0 &&
        (config_order == UlbvcSerial32Order::kOlder ||
         config_order == UlbvcSerial32Order::kAmbiguous)) {
      throw std::runtime_error("RTP configuration IDs must not move backwards");
    }
    const uint32_t old_config_id = session_config_id;
    const std::vector<uint8_t> old_session_config = session_config;
    const uint8_t old_session_config_flags = session_config_flags;
    const uint32_t old_pending_config_id = pending_random_access_config_id;
    if (old_pending_config_id != 0 && config_order == UlbvcSerial32Order::kNewer) {
      throw std::runtime_error("RTP configuration switch is already waiting for a random-access EFU");
    }
    session_config_id = config_id;
    session_config = config_unit;
    session_config_flags = flags;
    const uint32_t new_pending_config_id =
        old_pending_config_id != 0
            ? old_pending_config_id
            : (old_config_id != 0 && config_order == UlbvcSerial32Order::kNewer ? config_id : 0);
    // The SCU itself must be queued before the pending random-access gate is
    // armed; otherwise the gate would reject the SCU that establishes it.
    pending_random_access_config_id = 0;
    try {
      SendUnitLocked(RtpUnitType::kScu, flags, config_id, config_id, timestamp, config_unit);
      pending_random_access_config_id = new_pending_config_id;
    } catch (...) {
      session_config_id = old_config_id;
      session_config = old_session_config;
      session_config_flags = old_session_config_flags;
      pending_random_access_config_id = old_pending_config_id;
      throw;
    }
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
        sending = true;
        sending_since = pending.enqueued_at;
      }
      queue_cv.notify_all();
      try {
        const std::size_t estimated_wire_packet_size = pending.bytes.size() + kIpv4UdpOverheadBytes;
        auto rate = pacing_rate_bps.load();
        if (rate != applied_pacing_rate) {
          pacer->UpdateRate(rate);
          applied_pacing_rate = rate;
        }
        auto delay = pacer->ConsumeAndGetDelay(estimated_wire_packet_size);
        const bool paced = delay.count() > 0;
        while (delay.count() > 0) {
          std::unique_lock<std::mutex> lock(queue_mutex);
          queue_cv.wait_for(lock, delay,
                            [this, rate] { return stopping || pacing_rate_bps.load() != rate; });
          if (stopping) return;
          rate = pacing_rate_bps.load();
          if (rate != applied_pacing_rate) {
            pacer->UpdateRate(rate);
            applied_pacing_rate = rate;
          }
          delay = pacer->RemainingDelay();
        }
        const auto queue_delay = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - pending.enqueued_at);
        max_queue_delay_us.store(std::max<uint64_t>(max_queue_delay_us.load(), queue_delay.count()));
        queue_delay_total_us.fetch_add(queue_delay.count());
        queue_delay_samples.fetch_add(1);
        const auto send_begin = std::chrono::steady_clock::now();
        if (::sendto(socket, pending.bytes.data(), pending.bytes.size(), 0,
                     reinterpret_cast<const sockaddr*>(&address), address_length) < 0) {
          throw std::runtime_error("failed to send RTP packet");
        }
        const auto blocked = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - send_begin);
        socket_block_us.fetch_add(static_cast<uint64_t>(blocked.count()));
        wire_bytes_sent.fetch_add(estimated_wire_packet_size);
        if (pending.retransmission) retransmitted_wire_bytes.fetch_add(estimated_wire_packet_size);
        media_unit_bytes_sent.fetch_add(pending.media_unit_bytes);
        packets_sent.fetch_add(1);
        rtp_octets_sent.fetch_add(pending.bytes.size() >= 12 ? pending.bytes.size() - 12 : 0);
        last_rtp_timestamp.store((static_cast<uint32_t>(pending.bytes[4]) << 24) |
                                 (static_cast<uint32_t>(pending.bytes[5]) << 16) |
                                 (static_cast<uint32_t>(pending.bytes[6]) << 8) |
                                 static_cast<uint32_t>(pending.bytes[7]));
        if (!pending.retransmission) RememberRtpPacket(pending.bytes);
        const uint64_t burst =
            paced ? estimated_wire_packet_size
                  : burst_bytes.fetch_add(estimated_wire_packet_size) + estimated_wire_packet_size;
        if (paced) burst_bytes.store(estimated_wire_packet_size);
        uint64_t previous = max_burst_bytes_observed.load();
        while (burst > previous &&
               !max_burst_bytes_observed.compare_exchange_weak(previous, burst)) {
        }
        {
          std::lock_guard<std::mutex> lock(queue_mutex);
          queued_bytes -= estimated_wire_packet_size;
          sending = false;
        }
        queue_cv.notify_all();
      } catch (...) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        send_error = std::current_exception();
        queued_bytes = 0;
        queue.clear();
        sending = false;
        stopping = true;
        queue_cv.notify_all();
        return;
      }
    }
  }

  void SendRtcpLoop() {
    const auto cname = std::string("ulbvc-") + std::to_string(ssrc);
    auto next_report = std::chrono::steady_clock::now();
    while (!rtcp_stopping.load()) {
      const auto now = std::chrono::steady_clock::now();
      try {
        std::vector<uint8_t> response;
        {
          std::lock_guard<std::mutex> lock(control_mutex);
          if (!control_responses.empty()) {
            response = std::move(control_responses.front());
            control_responses.pop_front();
          }
        }
        if (!response.empty()) {
          if (rtcp_endpoint->TrySend(response, true))
            ++rtcp_packets_sent;
          else {
            std::lock_guard<std::mutex> lock(control_mutex);
            if (control_responses.size() < 32) control_responses.push_front(std::move(response));
          }
        }
        if (now >= next_report) {
          bool report_sent = false;
          if (packets_sent.load() != 0) {
            const auto wall_now = std::chrono::system_clock::now().time_since_epoch();
            const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(wall_now);
            const auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
                wall_now - seconds);
            constexpr uint64_t kNtpEpochOffset = 2208988800ull;
            RtcpSenderReport report;
            report.sender_ssrc = ssrc;
            report.ntp_seconds = static_cast<uint32_t>(
                static_cast<uint64_t>(seconds.count()) + kNtpEpochOffset);
            report.ntp_fraction = static_cast<uint32_t>(
                (static_cast<uint64_t>(nanoseconds.count()) << 32) / 1000000000ull);
            report.rtp_timestamp = last_rtp_timestamp.load();
            report.packet_count = static_cast<uint32_t>(packets_sent.load());
            report.octet_count = static_cast<uint32_t>(rtp_octets_sent.load());
            std::vector<std::vector<uint8_t>> extra;
            if (rate_feedback_enabled.load()) {
              UlbvcControlMessage capabilities;
              capabilities.type = UlbvcControlType::kCapabilities;
              capabilities.media_ssrc = ssrc;
              capabilities.tlvs.push_back(
                  EncodeRateCapabilities({rate_session_id, feedback_interval_ms.load(),
                                          static_cast<uint32_t>(std::clamp<uint64_t>(
                                              pacing_rate_bps.load() / 40, 2048, 2500000))}));
              extra.push_back(EncodeUlbvcRtcpApp(ssrc, capabilities));
            }
            report_sent = rtcp_endpoint->TrySend(EncodeRtcpSenderReport(report, cname, extra));
            if (report_sent) {
              const uint32_t compact =
                  ((report.ntp_seconds & 0xffffu) << 16) | (report.ntp_fraction >> 16);
              {
                std::lock_guard<std::mutex> lock(feedback_mutex);
                sender_report_times.emplace_back(compact, std::chrono::steady_clock::now());
                while (sender_report_times.size() > 32) sender_report_times.pop_front();
              }
              ++rtcp_packets_sent;
            }
          }
          next_report = std::chrono::steady_clock::now() +
                        (report_sent ? rtcp_interval : std::chrono::milliseconds(100));
        }
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            next_report - std::chrono::steady_clock::now());
        const auto timeout = std::max(std::chrono::milliseconds(1),
                                      std::min(std::chrono::milliseconds(100), remaining));
        std::vector<uint8_t> packet;
        if (!rtcp_endpoint->Receive(&packet, timeout)) continue;
        HandleRtcpPacket(packet);
      } catch (...) {
        if (rtcp_stopping.load()) return;
        // Malformed, unexpected or temporarily undeliverable RTCP must not
        // terminate the media sender. The next report interval retries.
      }
    }
  }
};
RtpUlbvcSender::RtpUlbvcSender(const std::string& host, uint16_t port, uint32_t ssrc,
                             uint64_t pacing_rate_bps, std::size_t max_burst_bytes,
                             std::size_t max_queue_bytes, uint64_t max_queue_delay_ms,
                             uint8_t payload_type, std::chrono::milliseconds rtcp_interval) {
  impl_ = std::make_unique<Impl>();
  if (payload_type < 96 || payload_type > 127)
    throw std::invalid_argument("RTP payload type must be in [96, 127]");
  if (port == std::numeric_limits<uint16_t>::max())
    throw std::invalid_argument("RTP destination port 65535 has no adjacent RTCP port");
  if (rtcp_interval.count() <= 0)
    throw std::invalid_argument("RTCP sender-report interval must be positive");
  std::random_device rd;
  if (ssrc == 0) {
    ssrc = (static_cast<uint32_t>(rd()) << 16) ^ static_cast<uint32_t>(rd());
    if (ssrc == 0) ssrc = 1;
  }
  impl_->sequence = static_cast<uint16_t>(rd());
  impl_->ssrc = ssrc;
  impl_->rate_session_id = (uint64_t(rd()) << 32) ^ rd();
  if (!impl_->rate_session_id) impl_->rate_session_id = 1;
  impl_->payload_type = payload_type;
  impl_->rtcp_interval = rtcp_interval;
  impl_->pacer = std::make_unique<UdpPacer>(pacing_rate_bps, max_burst_bytes);
  impl_->pacing_rate_bps = pacing_rate_bps;
  impl_->applied_pacing_rate = pacing_rate_bps;
  impl_->max_queue_bytes = max_queue_bytes;
  impl_->max_queue_delay_ms = max_queue_delay_ms;
  if (max_queue_bytes == 0) throw std::runtime_error("RTP send queue bytes must be positive");
  // A full RTP packet carries a 24-byte ULBVC descriptor in a 1200-byte
  // packetization budget, plus the 12-byte RTP header and IPv4/UDP overhead.
  constexpr std::size_t kMaxRtpWirePacketBytes = 12 + 1200 + kIpv4UdpOverheadBytes;
  if (max_burst_bytes < kMaxRtpWirePacketBytes)
    throw std::runtime_error("RTP max burst bytes must be at least one RTP packet");
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
  std::exception_ptr last_bind_error;
  for (int attempt = 0; attempt < 32; ++attempt) {
    const int candidate = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (candidate < 0) throw std::runtime_error("failed to create RTP socket");
    sockaddr_in local{};
    local.sin_family = AF_INET;
    local.sin_addr.s_addr = htonl(INADDR_ANY);
    local.sin_port = 0;
    if (::bind(candidate, reinterpret_cast<sockaddr*>(&local), sizeof(local)) < 0) {
      ::close(candidate);
      continue;
    }
    socklen_t local_length = sizeof(local);
    if (::getsockname(candidate, reinterpret_cast<sockaddr*>(&local), &local_length) < 0) {
      ::close(candidate);
      continue;
    }
    const uint16_t local_rtp_port = ntohs(local.sin_port);
    if ((local_rtp_port & 1u) != 0 || local_rtp_port == 0 || local_rtp_port >= 65534) {
      ::close(candidate);
      continue;
    }
    try {
      impl_->rtcp_endpoint = std::make_unique<RtcpUdpEndpoint>(
          static_cast<uint16_t>(local_rtp_port + 1), host,
          static_cast<uint16_t>(port + 1));
      impl_->socket = candidate;
      break;
    } catch (...) {
      last_bind_error = std::current_exception();
      ::close(candidate);
    }
  }
  if (impl_->socket < 0) {
    if (last_bind_error) std::rethrow_exception(last_bind_error);
    throw std::runtime_error("failed to bind a paired RTP/RTCP UDP port");
  }
  impl_->send_thread = std::thread([impl = impl_.get()] { impl->SendLoop(); });
  impl_->rtcp_thread = std::thread([impl = impl_.get()] { impl->SendRtcpLoop(); });
}
RtpUlbvcSender::~RtpUlbvcSender() = default;
void RtpUlbvcSender::SendUnit(RtpUnitType type, uint8_t flags, uint32_t config_id, uint32_t unit_id,
                             uint32_t timestamp, const std::vector<uint8_t>& unit) {
  impl_->SendUnit(type, flags, config_id, unit_id, timestamp, unit);
}
void RtpUlbvcSender::Flush() { impl_->Flush(); }
void RtpUlbvcSender::Close() { impl_->Flush(); }
void RtpUlbvcSender::SetSessionConfig(uint32_t config_id, const std::vector<uint8_t>& config_unit,
                                     uint8_t flags, uint32_t timestamp) {
  const auto scu = ParseScu(config_unit);
  if (scu.config_id != config_id) throw std::runtime_error("RTP SCU config ID mismatch");
  impl_->SetSessionConfig(config_id, config_unit, flags, timestamp);
}
void RtpUlbvcSender::ResendSessionConfig(uint32_t timestamp) {
  if (impl_->session_config.empty()) throw std::runtime_error("RTP session config is not set");
  impl_->SendUnit(RtpUnitType::kScu, impl_->session_config_flags, impl_->session_config_id,
                  impl_->session_config_id,
                  timestamp, impl_->session_config);
}
RtpTransportStats RtpUlbvcSender::Stats() const {
  RtpTransportStats stats;
  stats.media_unit_bytes = impl_->media_unit_bytes_sent.load();
  stats.rtp_payload_bytes = impl_->rtp_octets_sent.load();
  stats.wire_bytes = impl_->wire_bytes_sent.load();
  stats.packets = impl_->packets_sent.load();
  stats.max_burst_bytes = impl_->max_burst_bytes_observed.load();
  stats.max_queue_delay_us = impl_->max_queue_delay_us.load();
  const uint64_t samples = impl_->queue_delay_samples.load();
  stats.average_queue_delay_us = samples == 0 ? 0 : impl_->queue_delay_total_us.load() / samples;
  stats.socket_block_us = impl_->socket_block_us.load();
  stats.rtcp_packets_sent = impl_->rtcp_packets_sent.load();
  stats.rtcp_packets_received = impl_->rtcp_packets_received.load();
  stats.rtcp_wire_bytes = impl_->rtcp_endpoint ? impl_->rtcp_endpoint->sent_wire_bytes() : 0;
  stats.retransmitted_wire_bytes = impl_->retransmitted_wire_bytes.load();
  stats.pacing_rate_bps = impl_->pacing_rate_bps.load();
  stats.serialized_wire_bytes = impl_->serialized_wire_bytes.load();
  {
    std::lock_guard<std::mutex> lock(impl_->queue_mutex);
    stats.queued_wire_bytes = impl_->queued_bytes;
    const auto oldest = impl_->sending
                            ? impl_->sending_since
                            : (!impl_->queue.empty() ? impl_->queue.front().enqueued_at
                                                     : std::chrono::steady_clock::time_point{});
    if (oldest != std::chrono::steady_clock::time_point{})
      stats.oldest_queue_delay_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now() - oldest)
                                        .count();
  }
  return stats;
}
void RtpUlbvcSender::EnableRateFeedback(uint32_t interval_ms) {
  (void)EncodeRateCapabilities({impl_->rate_session_id, interval_ms});
  impl_->feedback_interval_ms.store(interval_ms);
  impl_->rate_feedback_enabled.store(true);
  impl_->rtcp_endpoint->SetSendBudget(std::max<uint64_t>(1024, impl_->pacing_rate_bps.load() / 39));
}
void RtpUlbvcSender::SetPacingRate(uint64_t wire_bps) {
  if (!wire_bps || wire_bps > 100000000)
    throw std::invalid_argument("invalid RTP dynamic pacing rate");
  impl_->pacing_rate_bps.store(wire_bps);
  impl_->rtcp_endpoint->SetSendBudget(std::max<uint64_t>(1024, wire_bps / 39));
  impl_->queue_cv.notify_all();
}
RateFeedbackSnapshot RtpUlbvcSender::Feedback() const {
  std::lock_guard<std::mutex> lock(impl_->feedback_mutex);
  return impl_->feedback_snapshot;
}
bool RtpUlbvcSender::ConsumeRandomAccessRequest() {
  return impl_->random_access_requests.exchange(0) != 0;
}
bool RtpUlbvcSender::PopUlbvcControl(UlbvcControlMessage* message) {
  if (message == nullptr) return false;
  std::lock_guard<std::mutex> lock(impl_->control_mutex);
  if (impl_->control_queue.empty()) return false;
  *message = std::move(impl_->control_queue.front());
  impl_->control_queue.pop_front();
  return true;
}
uint16_t RtpUlbvcSender::rtcp_local_port() const {
  return impl_->rtcp_endpoint == nullptr ? 0 : impl_->rtcp_endpoint->local_port();
}
void RtpUlbvcSender::SendUlbvcControlResponse(const UlbvcControlMessage& request, bool accepted,
                                            const std::string& reason) {
  UlbvcControlMessage response;
  response.type = accepted ? UlbvcControlType::kAck : UlbvcControlType::kReject;
  response.transaction_id = request.transaction_id;
  response.apply_after_frame_id = request.apply_after_frame_id;
  response.media_ssrc = impl_->ssrc;
  response.tlvs.push_back(UlbvcControlTlv{1, std::vector<uint8_t>(reason.begin(), reason.end())});
  RtcpReceiverReport report;
  report.sender_ssrc = impl_->ssrc;
  const auto app = EncodeUlbvcRtcpApp(impl_->ssrc, response);
  const auto compound = EncodeRtcpReceiverReport(report, std::string("ulbvc-") + std::to_string(impl_->ssrc), {app});
  std::lock_guard<std::mutex> lock(impl_->control_mutex);
  if (impl_->seen_control_transactions.count(request.transaction_id))
    impl_->control_response_cache[request.transaction_id] = compound;
  if (impl_->control_responses.size() < 32) impl_->control_responses.push_back(compound);
}
RtpPacket DecodeRtpPacket(const std::vector<uint8_t>& b, uint8_t expected_payload_type) {
  if (b.size() < 12 || (b[0] >> 6) != 2) throw std::runtime_error("invalid RTP packet");
  if ((b[1] & 0x7f) != expected_payload_type || expected_payload_type < 96 ||
      expected_payload_type > 127)
    throw std::runtime_error("unsupported RTP payload type");
  const std::size_t csrc_bytes = static_cast<std::size_t>(b[0] & 0x0f) * 4;
  std::size_t payload_offset = 12 + csrc_bytes;
  if (payload_offset > b.size()) throw std::runtime_error("truncated RTP CSRC list");
  if ((b[0] & 0x10) != 0) {
    if (payload_offset + 4 > b.size()) throw std::runtime_error("truncated RTP extension header");
    const std::size_t extension_bytes =
        static_cast<std::size_t>((static_cast<uint16_t>(b[payload_offset + 2]) << 8) |
                                  b[payload_offset + 3]) *
        4;
    payload_offset += 4;
    if (extension_bytes > b.size() - payload_offset)
      throw std::runtime_error("truncated RTP extension payload");
    payload_offset += extension_bytes;
  }
  std::size_t payload_end = b.size();
  if ((b[0] & 0x20) != 0) {
    const uint8_t padding_bytes = b.back();
    if (padding_bytes == 0 || padding_bytes > payload_end - payload_offset)
      throw std::runtime_error("invalid RTP padding");
    payload_end -= padding_bytes;
  }
  if (payload_offset > payload_end) throw std::runtime_error("RTP header exceeds packet");
  RtpPacket p;
  p.sequence = (b[2] << 8) | b[3];
  p.timestamp = (uint32_t(b[4]) << 24) | (uint32_t(b[5]) << 16) | (uint32_t(b[6]) << 8) | b[7];
  p.ssrc = (uint32_t(b[8]) << 24) | (uint32_t(b[9]) << 16) | (uint32_t(b[10]) << 8) | b[11];
  p.marker = (b[1] & 0x80) != 0;
  p.payload.assign(b.begin() + static_cast<std::ptrdiff_t>(payload_offset),
                   b.begin() + static_cast<std::ptrdiff_t>(payload_end));
  return p;
}

}  // namespace ulbvc::transport
