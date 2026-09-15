#include "mlvc/transport/rtp_message_receiver.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace mlvc::transport {

namespace {
constexpr std::size_t kMaxSequenceHistory = 4096;
}

RtpMessageReceiver::RtpMessageReceiver(uint16_t port, std::string context)
    : context_(std::move(context)) {
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

void RtpMessageReceiver::TrackSequence(uint16_t sequence) {
  if (seen_sequences_.find(sequence) != seen_sequences_.end()) {
    ++duplicate_packets_;
    return;
  }

  if (have_sequence_) {
    const uint16_t delta = static_cast<uint16_t>(sequence - highest_sequence_);
    if (delta < 0x8000u) {
      if (delta > 1u) {
        lost_packets_ += static_cast<uint64_t>(delta - 1u);
      }
      highest_sequence_ = sequence;
    } else {
      ++reordered_packets_;
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
      const RtpPacket packet = DecodeRtpPacket(packet_bytes);
      if (!have_sequence_) {
        ssrc_ = packet.ssrc;
      } else if (packet.ssrc != ssrc_) {
        throw std::runtime_error(context_ + ": RTP SSRC changed");
      }
      TrackSequence(packet.sequence);

      const RtpMlvcFragment fragment = DecodeRtpMlvcFragment(packet.payload);
      std::vector<uint8_t> unit;
      if (!reassembler_.Push(fragment, &unit)) continue;

      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_not_full_.wait(lock,
                           [this] { return stopping_ || unit_queue_.size() < kMaxQueuedUnits; });
      if (stopping_) return;
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
  *unit = std::move(unit_queue_.front());
  unit_queue_.pop_front();
  lock.unlock();
  queue_not_full_.notify_one();
  return true;
}

}  // namespace mlvc::transport
