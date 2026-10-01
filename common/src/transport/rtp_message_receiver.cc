#include "mlvc/transport/rtp_message_receiver.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <random>
#include <utility>

namespace mlvc::transport {

namespace {
constexpr std::size_t kMaxSequenceHistory = 4096;

uint32_t RandomSsrc() {
  std::random_device random;
  uint32_t value = (static_cast<uint32_t>(random()) << 16) ^
                   static_cast<uint32_t>(random());
  return value == 0 ? 1 : value;
}

std::string CnameForSsrc(uint32_t ssrc) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string cname = "mlvc-";
  for (int shift = 28; shift >= 0; shift -= 4) cname.push_back(kHex[(ssrc >> shift) & 0xf]);
  return cname;
}

uint32_t CompactNtp(const RtcpSenderReport& report) {
  return ((report.ntp_seconds & 0xffffu) << 16) | (report.ntp_fraction >> 16);
}
}

RtpMessageReceiver::RtpMessageReceiver(uint16_t port, std::string context, uint8_t payload_type)
    : context_(std::move(context)), payload_type_(payload_type) {
  if (payload_type_ < 96 || payload_type_ > 127) {
    throw std::invalid_argument(context_ + ": RTP payload type must be in [96, 127]");
  }
  socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (socket_ < 0) {
    throw std::runtime_error(context_ + ": failed to create RTP socket");
  }

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  address.sin_port = htons(port);
  if (::bind(socket_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
    ::close(socket_);
    socket_ = -1;
    throw std::runtime_error(context_ + ": failed to bind RTP socket");
  }

  // RTP frames are fragmented into many datagrams and decoding can briefly
  // stop reading while an accelerator operation is in flight.  Keep enough
  // kernel queue for a complete short burst so a transient scheduling pause
  // does not turn into a permanently incomplete MLVC unit.
  int receive_buffer = 16 * 1024 * 1024;
  (void)::setsockopt(socket_, SOL_SOCKET, SO_RCVBUF, &receive_buffer, sizeof(receive_buffer));
  timeval receive_timeout{};
  // Model initialization on Ascend can take tens of seconds before the first
  // packet is consumed; retain the bounded wait while allowing that startup.
  receive_timeout.tv_sec = 60;
  (void)::setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO, &receive_timeout, sizeof(receive_timeout));

  sockaddr_in bound_address{};
  socklen_t bound_length = sizeof(bound_address);
  if (::getsockname(socket_, reinterpret_cast<sockaddr*>(&bound_address), &bound_length) < 0) {
    ::close(socket_);
    socket_ = -1;
    throw std::runtime_error(context_ + ": failed to query RTP socket port");
  }
  local_port_ = ntohs(bound_address.sin_port);
  if (local_port_ == std::numeric_limits<uint16_t>::max()) {
    ::close(socket_);
    socket_ = -1;
    throw std::runtime_error(context_ + ": RTP port 65535 has no adjacent RTCP port");
  }
  try {
    rtcp_endpoint_ = std::make_unique<RtcpUdpEndpoint>(
        static_cast<uint16_t>(local_port_ + 1));
  } catch (...) {
    ::close(socket_);
    socket_ = -1;
    throw;
  }
  receive_thread_ = std::thread(&RtpMessageReceiver::ReceiveLoop, this);
  rtcp_thread_ = std::thread(&RtpMessageReceiver::RtcpLoop, this);
}

RtpMessageReceiver::~RtpMessageReceiver() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    stopping_ = true;
  }
  queue_not_empty_.notify_all();
  queue_not_full_.notify_all();
  rtcp_stopping_ = true;
  if (socket_ >= 0) {
    ::shutdown(socket_, SHUT_RDWR);
    ::close(socket_);
    socket_ = -1;
  }
  if (receive_thread_.joinable()) {
    receive_thread_.join();
  }
  if (rtcp_thread_.joinable()) rtcp_thread_.join();
  if (rtcp_endpoint_) rtcp_endpoint_->Close();
}

bool RtpMessageReceiver::TrackSequence(
    uint16_t sequence, std::chrono::steady_clock::time_point arrival_time) {
  std::lock_guard<std::mutex> lock(sequence_mutex_);
  if (seen_sequences_.find(sequence) != seen_sequences_.end()) {
    ++duplicate_packets_;
    return true;
  }

  if (have_sequence_) {
    const uint16_t delta = static_cast<uint16_t>(sequence - highest_sequence_);
    if (delta < 0x8000u) {
      if (delta > 1u) {
        const uint16_t gap = static_cast<uint16_t>(delta - 1u);
        lost_packets_ += gap;
        const uint16_t tracked_gap =
            std::min<uint16_t>(gap, static_cast<uint16_t>(kMaxSequenceHistory));
        for (uint16_t offset = tracked_gap; offset > 0; --offset) {
          const uint16_t missing = static_cast<uint16_t>(sequence - offset);
          missing_sequences_.insert(missing);
          missing_sequence_order_.push_back(missing);
          missing_sequence_times_[missing] = arrival_time;
        }
        while (missing_sequence_order_.size() > kMaxSequenceHistory) {
          missing_sequences_.erase(missing_sequence_order_.front());
          missing_sequence_times_.erase(missing_sequence_order_.front());
          missing_sequence_order_.pop_front();
        }
      }
      highest_sequence_ = sequence;
    } else {
      ++reordered_packets_;
      if (missing_sequences_.erase(sequence) != 0 && lost_packets_ > 0) {
        --lost_packets_;
      }
      missing_sequence_times_.erase(sequence);
    }
  } else {
    have_sequence_ = true;
    highest_sequence_ = sequence;
  }

  seen_sequences_.insert(sequence);
  seen_sequence_order_.push_back(sequence);
  while (seen_sequence_order_.size() > kMaxSequenceHistory) {
    seen_sequences_.erase(seen_sequence_order_.front());
    seen_sequence_order_.pop_front();
  }
  return false;
}

void RtpMessageReceiver::SetReceiveError(std::exception_ptr error) {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  if (!stopping_ && receive_error_ == nullptr) {
    receive_error_ = std::move(error);
  }
  queue_not_empty_.notify_all();
  queue_not_full_.notify_all();
}

void RtpMessageReceiver::ReceiveLoop() {
  try {
    std::array<uint8_t, 65536> buffer{};
    for (;;) {
      {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (stopping_) return;
      }
      const ssize_t received = ::recv(socket_, buffer.data(), buffer.size(), 0);
      if (received <= 0) {
        if ((errno == EINTR) != 0) continue;
        {
          std::lock_guard<std::mutex> lock(queue_mutex_);
          if (stopping_) return;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
          throw std::runtime_error(
              context_ + ": RTP receive timed out (lost=" + std::to_string(lost_packets_.load()) +
              ", duplicate=" + std::to_string(duplicate_packets_.load()) +
              ", reordered=" + std::to_string(reordered_packets_.load()) + ")");
        }
        throw std::runtime_error(context_ + ": RTP receive failed: " + std::strerror(errno));
      }

      const std::vector<uint8_t> packet_bytes(buffer.begin(), buffer.begin() + received);
      const RtpPacket packet = DecodeRtpPacket(packet_bytes, payload_type_);
      if (!have_sequence_) {
        ssrc_.store(packet.ssrc);
      } else if (packet.ssrc != ssrc_.load()) {
        throw std::runtime_error(context_ + ": RTP SSRC changed");
      }
      if (TrackSequence(packet.sequence, std::chrono::steady_clock::now())) continue;
      UpdateRtcpReceiveStats(packet);

      RtpMlvcFragment fragment = DecodeRtpMlvcFragment(packet.payload);
      fragment.timestamp = packet.timestamp;
      fragment.ssrc = packet.ssrc;
      if (fragment.unit_type == RtpUnitType::kEfu) {
        const auto config_limit = std::find_if(
            config_frame_limits_.begin(), config_frame_limits_.end(),
            [&fragment](const auto& entry) { return entry.first == fragment.config_id; });
        if (config_limit != config_frame_limits_.end()) {
          const uint64_t max_unit_bytes = std::min<uint64_t>(
              kMlvcMediaUnitMaxBytes,
              static_cast<uint64_t>(config_limit->second) + kMlvcHeaderMaxBytes);
          if (fragment.unit_length > max_unit_bytes) {
            throw std::runtime_error(context_ + ": EFU exceeds its SCU frame-size limit");
          }
        }
      }
      const bool fragment_is_last =
          static_cast<uint64_t>(fragment.fragment_offset) + fragment.payload.size() ==
          fragment.unit_length;
      const bool expected_marker = fragment.unit_type == RtpUnitType::kEfu && fragment_is_last;
      if (packet.marker != expected_marker)
        throw std::runtime_error(context_ + ": RTP marker does not match MLVC unit boundary");
      std::vector<uint8_t> unit;
      if (!reassembler_.Push(fragment, &unit)) continue;
      const auto media_header = ParseMediaUnitHeader(unit);
      if (media_header.unit_type != fragment.unit_type ||
          media_header.unit_id != fragment.unit_id ||
          media_header.config_id != fragment.config_id ||
          media_header.unit_length != unit.size()) {
        throw std::runtime_error(context_ + ": RTP descriptor does not match media unit header");
      }
      try {
        ValidateMediaUnit(unit);
      } catch (const MlvcEfuCrcError&) {
        // A complete but damaged EFU is a lost frame, not a fatal RTP session
        // error. The frame-level receiver waits for a valid recovery point.
        continue;
      }
      if (media_header.unit_type == MlvcMediaUnitType::kScu) {
        const MlvcScu scu = ParseScu(unit);
        auto existing = std::find_if(
            config_frame_limits_.begin(), config_frame_limits_.end(),
            [&scu](const auto& entry) { return entry.first == scu.config_id; });
        if (existing != config_frame_limits_.end()) {
          existing->second = scu.max_frame_bytes;
        } else {
          config_frame_limits_.emplace_back(scu.config_id, scu.max_frame_bytes);
          constexpr std::size_t kMaxRememberedConfigLimits = 8;
          if (config_frame_limits_.size() > kMaxRememberedConfigLimits) {
            config_frame_limits_.pop_front();
          }
        }
      }
      if (unit.size() > kMaxQueuedUnitBytes)
        throw std::runtime_error(context_ + ": reassembled unit exceeds receive queue byte limit");

      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_not_full_.wait(lock, [this, &unit] {
        return stopping_ ||
               (unit_queue_.size() < kMaxQueuedUnits &&
                unit.size() <= kMaxQueuedUnitBytes - unit_queue_bytes_);
      });
      if (stopping_) return;
      unit_queue_bytes_ += unit.size();
      unit_queue_.push_back(std::move(unit));
      lock.unlock();
      queue_not_empty_.notify_one();
    }
  } catch (...) {
    SetReceiveError(std::current_exception());
  }
}

void RtpMessageReceiver::UpdateRtcpReceiveStats(const RtpPacket& packet) {
  std::lock_guard<std::mutex> lock(rtcp_stats_mutex_);
  uint64_t extended_sequence = packet.sequence;
  if (!rtcp_have_sequence_) {
    rtcp_have_sequence_ = true;
    rtcp_max_sequence_ = packet.sequence;
    rtcp_base_sequence_ = packet.sequence;
    rtcp_highest_sequence_ = packet.sequence;
  } else {
    const uint16_t delta = static_cast<uint16_t>(packet.sequence - rtcp_max_sequence_);
    if (delta != 0 && delta < 0x8000u) {
      if (packet.sequence < rtcp_max_sequence_) rtcp_sequence_cycles_ += 0x10000u;
      rtcp_max_sequence_ = packet.sequence;
      extended_sequence = rtcp_sequence_cycles_ + packet.sequence;
      rtcp_highest_sequence_ = extended_sequence;
    } else if (delta >= 0x8000u && packet.sequence > rtcp_max_sequence_ &&
               rtcp_sequence_cycles_ >= 0x10000u) {
      extended_sequence = rtcp_sequence_cycles_ - 0x10000u + packet.sequence;
    } else {
      extended_sequence = rtcp_sequence_cycles_ + packet.sequence;
    }
  }
  ++rtcp_received_unique_;
  if (extended_sequence > rtcp_highest_sequence_) rtcp_highest_sequence_ = extended_sequence;

  const auto now = std::chrono::steady_clock::now();
  const uint64_t elapsed_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
  const uint64_t elapsed_seconds = elapsed_ns / 1000000000u;
  const uint64_t subsecond_ns = elapsed_ns % 1000000000u;
  const uint32_t arrival_ticks = static_cast<uint32_t>(
      (elapsed_seconds * 90000u + subsecond_ns * 90000u / 1000000000u) & 0xffffffffu);
  const uint32_t transit = arrival_ticks - packet.timestamp;
  if (rtcp_have_transit_) {
    const int32_t transit_delta = static_cast<int32_t>(transit - rtcp_prior_transit_);
    const double difference = static_cast<double>(transit_delta < 0 ? -int64_t{transit_delta}
                                                                    : transit_delta);
    rtcp_jitter_ += (difference - rtcp_jitter_) / 16.0;
  } else {
    rtcp_have_transit_ = true;
  }
  rtcp_prior_transit_ = transit;
}

RtcpReportBlock RtpMessageReceiver::SnapshotReportBlock(uint32_t source_ssrc) {
  std::lock_guard<std::mutex> lock(rtcp_stats_mutex_);
  const uint64_t expected = rtcp_have_sequence_
                                ? rtcp_highest_sequence_ - rtcp_base_sequence_ + 1u
                                : 0;
  const int64_t cumulative_lost = static_cast<int64_t>(expected) -
                                  static_cast<int64_t>(rtcp_received_unique_);
  const uint64_t interval_expected = expected - rtcp_prior_expected_;
  const uint64_t interval_received = rtcp_received_unique_ - rtcp_prior_received_;
  const int64_t interval_lost = static_cast<int64_t>(interval_expected) -
                                static_cast<int64_t>(interval_received);
  const uint8_t fraction_lost = interval_expected == 0 || interval_lost <= 0
                                    ? 0
                                    : static_cast<uint8_t>(std::min<int64_t>(
                                          255, interval_lost * 256 /
                                                   static_cast<int64_t>(interval_expected)));
  rtcp_prior_expected_ = expected;
  rtcp_prior_received_ = rtcp_received_unique_;
  const int64_t clipped_lost = std::clamp<int64_t>(cumulative_lost, -8388608, 8388607);
  uint32_t delay_since_sr = 0;
  if (rtcp_last_sender_report_ != 0) {
    const auto elapsed = std::chrono::steady_clock::now() - rtcp_last_sender_report_at_;
    const long double dlsr = std::chrono::duration<long double>(elapsed).count() * 65536.0L;
    delay_since_sr = static_cast<uint32_t>(std::min<long double>(
        std::max<long double>(0, dlsr), std::numeric_limits<uint32_t>::max()));
  }
  return RtcpReportBlock{source_ssrc, fraction_lost, static_cast<int32_t>(clipped_lost),
                         static_cast<uint32_t>(rtcp_highest_sequence_),
                         static_cast<uint32_t>(rtcp_jitter_), rtcp_last_sender_report_,
                         delay_since_sr};
}

std::vector<RtcpNack> RtpMessageReceiver::SnapshotPendingNacks(
    uint32_t sender_ssrc, uint32_t media_ssrc,
    std::chrono::steady_clock::time_point now) {
  std::lock_guard<std::mutex> lock(sequence_mutex_);
  std::vector<uint16_t> due;
  due.reserve(256);
  for (const auto& [sequence, first_seen] : missing_sequence_times_) {
    if (now - first_seen >= std::chrono::milliseconds(20)) {
      due.push_back(sequence);
      if (due.size() == 256) break;
    }
  }
  std::vector<RtcpNack> messages;
  for (uint16_t sequence : due) {
    bool grouped = false;
    for (auto& message : messages) {
      const uint16_t distance = static_cast<uint16_t>(sequence - message.pid);
      if (distance >= 1 && distance <= 16) {
        message.blp |= static_cast<uint16_t>(1u << (distance - 1));
        grouped = true;
        break;
      }
    }
    if (!grouped) messages.push_back(RtcpNack{sender_ssrc, media_ssrc, sequence, 0});
  }
  return messages;
}

void RtpMessageReceiver::RtcpLoop() {
  const uint32_t receiver_ssrc = RandomSsrc();
  const std::string cname = CnameForSsrc(receiver_ssrc);
  bool peer_known = false;
  auto last_nack_sent = std::chrono::steady_clock::now();
  while (!rtcp_stopping_.load()) {
    std::vector<uint8_t> bytes;
    try {
      const bool received = rtcp_endpoint_->Receive(&bytes, std::chrono::milliseconds(50));
      if (received) {
        const RtcpCompoundContents compound = DecodeRtcpCompound(bytes);
        if (compound.sender_report.has_value()) {
          const RtcpSenderReport& sender_report = *compound.sender_report;
          const uint32_t expected_ssrc = ssrc_.load();
          if (expected_ssrc != 0 && sender_report.sender_ssrc == expected_ssrc) {
            rtcp_endpoint_->AcceptLastPeer();
            peer_known = true;
            ++rtcp_sender_reports_received_;
            {
              std::lock_guard<std::mutex> lock(rtcp_stats_mutex_);
              rtcp_last_sender_report_ = CompactNtp(sender_report);
              rtcp_last_sender_report_at_ = std::chrono::steady_clock::now();
            }
            RtcpReceiverReport rr;
            rr.sender_ssrc = receiver_ssrc;
            rr.reports.push_back(SnapshotReportBlock(sender_report.sender_ssrc));
            rtcp_endpoint_->SendToLastPeer(EncodeRtcpReceiverReport(rr, cname));
            ++rtcp_receiver_reports_sent_;
          }
        }
      }
      const auto now = std::chrono::steady_clock::now();
      if (peer_known && now - last_nack_sent >= std::chrono::milliseconds(100)) {
        const uint32_t media_ssrc = ssrc_.load();
        if (media_ssrc != 0) {
          const auto nacks = SnapshotPendingNacks(receiver_ssrc, media_ssrc, now);
          if (!nacks.empty()) {
            std::vector<std::vector<uint8_t>> extra;
            extra.reserve(nacks.size());
            for (const auto& nack : nacks) extra.push_back(EncodeRtcpNack(nack));
            RtcpReceiverReport rr;
            rr.sender_ssrc = receiver_ssrc;
            rr.reports.push_back(SnapshotReportBlock(media_ssrc));
            rtcp_endpoint_->SendToLastPeer(EncodeRtcpReceiverReport(rr, cname, extra));
            ++rtcp_receiver_reports_sent_;
            last_nack_sent = now;
          }
        }
      }
    } catch (...) {
      if (rtcp_stopping_.load()) return;
      // Invalid or unauthenticated feedback must not stop RTP reception. The
      // receiver currently acts only on a matching SR and ignores other RTCP.
    }
  }
}

bool RtpMessageReceiver::Receive(std::vector<uint8_t>* unit) {
  if (unit == nullptr) {
    throw std::invalid_argument(context_ + ": output unit must not be null");
  }
  std::unique_lock<std::mutex> lock(queue_mutex_);
  queue_not_empty_.wait(
      lock, [this] { return stopping_ || receive_error_ != nullptr || !unit_queue_.empty(); });
  if (receive_error_ != nullptr) {
    std::rethrow_exception(receive_error_);
  }
  if (unit_queue_.empty()) return false;
  unit_queue_bytes_ -= unit_queue_.front().size();
  *unit = std::move(unit_queue_.front());
  unit_queue_.pop_front();
  lock.unlock();
  queue_not_full_.notify_one();
  return true;
}

bool RtpMessageReceiver::ReceiveFor(std::vector<uint8_t>* unit,
                                    std::chrono::milliseconds timeout) {
  if (unit == nullptr) {
    throw std::invalid_argument(context_ + ": output unit must not be null");
  }
  if (timeout.count() < 0) {
    throw std::invalid_argument(context_ + ": receive timeout must be non-negative");
  }
  std::unique_lock<std::mutex> lock(queue_mutex_);
  const bool ready = queue_not_empty_.wait_for(lock, timeout, [this] {
    return stopping_ || receive_error_ != nullptr || !unit_queue_.empty();
  });
  if (!ready) return false;
  if (receive_error_ != nullptr) {
    std::rethrow_exception(receive_error_);
  }
  if (unit_queue_.empty()) return false;
  unit_queue_bytes_ -= unit_queue_.front().size();
  *unit = std::move(unit_queue_.front());
  unit_queue_.pop_front();
  lock.unlock();
  queue_not_full_.notify_one();
  return true;
}

}  // namespace mlvc::transport
