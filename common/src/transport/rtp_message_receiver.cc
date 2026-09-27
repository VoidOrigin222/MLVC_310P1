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
#include <stdexcept>
#include <utility>

namespace mlvc::transport {

namespace {
constexpr std::size_t kMaxSequenceHistory = 4096;
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
  receive_thread_ = std::thread(&RtpMessageReceiver::ReceiveLoop, this);
}

RtpMessageReceiver::~RtpMessageReceiver() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    stopping_ = true;
  }
  queue_not_empty_.notify_all();
  queue_not_full_.notify_all();
  if (socket_ >= 0) {
    ::shutdown(socket_, SHUT_RDWR);
    ::close(socket_);
    socket_ = -1;
  }
  if (receive_thread_.joinable()) {
    receive_thread_.join();
  }
}

bool RtpMessageReceiver::TrackSequence(uint16_t sequence) {
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
        }
        while (missing_sequence_order_.size() > kMaxSequenceHistory) {
          missing_sequences_.erase(missing_sequence_order_.front());
          missing_sequence_order_.pop_front();
        }
      }
      highest_sequence_ = sequence;
    } else {
      ++reordered_packets_;
      if (missing_sequences_.erase(sequence) != 0 && lost_packets_ > 0) {
        --lost_packets_;
      }
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
        ssrc_ = packet.ssrc;
      } else if (packet.ssrc != ssrc_) {
        throw std::runtime_error(context_ + ": RTP SSRC changed");
      }
      if (TrackSequence(packet.sequence)) continue;

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
      ValidateMediaUnit(unit);
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
