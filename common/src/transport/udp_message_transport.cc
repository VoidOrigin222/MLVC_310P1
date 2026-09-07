#include "mlvc/transport/udp_message_transport.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <sys/time.h>

#include "mlvc/core/status.h"

namespace mlvc::transport {
namespace {
constexpr uint8_t kMagic0 = 0xeb;
constexpr uint8_t kMagic1 = 0x90;
constexpr uint8_t kTail0 = 0xcd;
constexpr uint8_t kTail1 = 0xde;
constexpr std::size_t kHeaderBytes = 14;
constexpr std::size_t kTailBytes = 2;
constexpr std::size_t kPayloadBytes = 1024;
constexpr std::size_t kMaxPacketBytes = kHeaderBytes + kPayloadBytes + kTailBytes;
constexpr std::size_t kMaxMessageBytes = 128u * 1024u * 1024u;
constexpr std::size_t kMaxQueuedMessages = 64;
constexpr int kReceiveTimeoutSeconds = 5;

uint16_t ReadU16(const uint8_t* data) {
  return static_cast<uint16_t>(data[0]) | (static_cast<uint16_t>(data[1]) << 8);
}

uint32_t ReadU32(const uint8_t* data) {
  return static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8) |
         (static_cast<uint32_t>(data[2]) << 16) | (static_cast<uint32_t>(data[3]) << 24);
}

void CheckPacket(const uint8_t* packet, std::size_t size, const std::string& context) {
  mlvc::Check(size >= kHeaderBytes + kTailBytes, "truncated " + context + " UDP packet");
  mlvc::Check(packet[0] == kMagic0 && packet[1] == kMagic1,
              "invalid " + context + " UDP magic");
  mlvc::Check(packet[size - 2] == kTail0 && packet[size - 1] == kTail1,
              "invalid " + context + " UDP tail");
  const uint16_t payload_size = ReadU16(packet + 6);
  mlvc::Check(size == kHeaderBytes + payload_size + kTailBytes,
              context + " UDP payload length mismatch");
}

struct Address {
  sockaddr_storage storage{};
  socklen_t length = 0;
};

Address ResolveAddress(const std::string& host, uint16_t port, const std::string& context) {
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_DGRAM;
  addrinfo* result = nullptr;
  const std::string service = std::to_string(port);
  mlvc::Check(getaddrinfo(host.c_str(), service.c_str(), &hints, &result) == 0 && result,
              "failed to resolve " + context + " UDP destination");
  Address address;
  std::memcpy(&address.storage, result->ai_addr, result->ai_addrlen);
  address.length = static_cast<socklen_t>(result->ai_addrlen);
  freeaddrinfo(result);
  return address;
}

}  // namespace

struct UdpMessageSender::Impl {
  struct PendingMessage {
    std::vector<uint8_t> message;
    uint8_t channel = 0;
  };

  explicit Impl(const std::string& host, uint16_t port, std::string context)
      : address(ResolveAddress(host, port, context)), error_context(std::move(context)) {
    socket = ::socket(AF_INET, SOCK_DGRAM, 0);
    mlvc::Check(socket >= 0, "failed to create " + error_context + " socket");
    send_thread = std::thread([this] { SendLoop(); });
  }

  ~Impl() {
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      stopping = true;
    }
    queue_cv.notify_all();
    if (send_thread.joinable()) {
      send_thread.join();
    }
    if (socket >= 0) {
      close(socket);
    }
  }

  void Send(const std::vector<uint8_t>& message, uint8_t channel) {
    mlvc::Check(!message.empty(), "cannot send an empty " + error_context + " message");
    mlvc::Check(message.size() <= kMaxMessageBytes, error_context + " message is too large");
    std::lock_guard<std::mutex> lock(queue_mutex);
    if (send_error != nullptr) {
      std::rethrow_exception(send_error);
    }
    mlvc::Check(!stopping, error_context + " is stopping");
    mlvc::Check(send_queue.size() < kMaxQueuedMessages,
                error_context + " send queue is full");
    send_queue.push_back(PendingMessage{message, channel});
    queue_cv.notify_one();
  }

  void Flush() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    queue_cv.wait(lock, [this] {
      return (send_queue.empty() && !sending) || send_error != nullptr;
    });
    if (send_error != nullptr) {
      std::rethrow_exception(send_error);
    }
  }

  void SendLoop() {
    for (;;) {
      PendingMessage pending;
      {
        std::unique_lock<std::mutex> lock(queue_mutex);
        queue_cv.wait(lock, [this] { return stopping || !send_queue.empty(); });
        if (send_queue.empty()) {
          return;
        }
        pending = std::move(send_queue.front());
        send_queue.pop_front();
        sending = true;
      }
      try {
        SendMessageNow(pending.message, pending.channel);
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

  void SendMessageNow(const std::vector<uint8_t>& message, uint8_t channel) {
    const std::size_t packet_count = (message.size() + kPayloadBytes - 1) / kPayloadBytes;
    mlvc::Check(packet_count <= std::numeric_limits<uint16_t>::max(),
                error_context + " message has too many fragments");
    mlvc::Check(message.size() <= std::numeric_limits<uint32_t>::max(),
                error_context + " message is too large");
    const uint32_t total_size = static_cast<uint32_t>(message.size());
    std::array<uint8_t, kMaxPacketBytes> packet{};
    for (std::size_t packet_index = 0; packet_index < packet_count; ++packet_index) {
      const std::size_t offset = packet_index * kPayloadBytes;
      const uint16_t payload_size = static_cast<uint16_t>(
          std::min(kPayloadBytes, message.size() - offset));
      packet[0] = kMagic0;
      packet[1] = kMagic1;
      packet[2] = static_cast<uint8_t>(packet_index);
      packet[3] = static_cast<uint8_t>(packet_index >> 8);
      packet[4] = static_cast<uint8_t>(packet_count);
      packet[5] = static_cast<uint8_t>(packet_count >> 8);
      packet[6] = static_cast<uint8_t>(payload_size);
      packet[7] = static_cast<uint8_t>(payload_size >> 8);
      packet[8] = static_cast<uint8_t>(total_size);
      packet[9] = static_cast<uint8_t>(total_size >> 8);
      packet[10] = static_cast<uint8_t>(total_size >> 16);
      packet[11] = static_cast<uint8_t>(total_size >> 24);
      packet[12] = channel;
      packet[13] = 0;
      std::memcpy(packet.data() + kHeaderBytes, message.data() + offset, payload_size);
      packet[kHeaderBytes + payload_size] = kTail0;
      packet[kHeaderBytes + payload_size + 1] = kTail1;
      const auto packet_size = kHeaderBytes + payload_size + kTailBytes;
      mlvc::Check(sendto(socket, packet.data(), packet_size, 0,
                         reinterpret_cast<const sockaddr*>(&address.storage), address.length) ==
                      static_cast<ssize_t>(packet_size),
                  "failed to send " + error_context + " UDP packet");
    }
  }

  int socket = -1;
  Address address;
  std::string error_context;
  std::mutex queue_mutex;
  std::condition_variable queue_cv;
  std::deque<PendingMessage> send_queue;
  std::exception_ptr send_error;
  bool stopping = false;
  bool sending = false;
  std::thread send_thread;
};

UdpMessageSender::UdpMessageSender(const std::string& host, uint16_t port,
                                   std::string error_context)
    : impl_(std::make_unique<Impl>(host, port, std::move(error_context))) {}

UdpMessageSender::~UdpMessageSender() = default;

void UdpMessageSender::Send(const std::vector<uint8_t>& message, uint8_t channel) {
  impl_->Send(message, channel);
}

void UdpMessageSender::Flush() { impl_->Flush(); }

struct UdpMessageReceiver::Impl {
  explicit Impl(uint16_t port, std::string context) : error_context(std::move(context)) {
    socket = ::socket(AF_INET, SOCK_DGRAM, 0);
    mlvc::Check(socket >= 0, "failed to create " + error_context + " socket");
    int reuse = 1;
    setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    const int receive_buffer = 8 * 1024 * 1024;
    setsockopt(socket, SOL_SOCKET, SO_RCVBUF, &receive_buffer, sizeof(receive_buffer));
    timeval receive_timeout{};
    receive_timeout.tv_sec = kReceiveTimeoutSeconds;
    setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &receive_timeout, sizeof(receive_timeout));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(port);
    mlvc::Check(bind(socket, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0,
                "failed to bind " + error_context + " port");
    receive_thread = std::thread([this] { ReceiveLoop(); });
  }

  ~Impl() {
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      stopping = true;
    }
    queue_cv.notify_all();
    if (socket >= 0) {
      shutdown(socket, SHUT_RDWR);
      close(socket);
    }
    if (receive_thread.joinable()) {
      receive_thread.join();
    }
  }

  std::vector<uint8_t> Receive() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    queue_cv.wait(lock, [this] {
      return stopping || receive_error != nullptr || !message_queue.empty();
    });
    if (receive_error != nullptr) {
      std::rethrow_exception(receive_error);
    }
    mlvc::Check(!message_queue.empty(), error_context + " stopped before receiving a message");
    std::vector<uint8_t> message = std::move(message_queue.front());
    message_queue.pop_front();
    return message;
  }

  void ReceiveLoop() {
    try {
      for (;;) {
        {
          std::lock_guard<std::mutex> lock(queue_mutex);
          if (stopping) {
            return;
          }
        }
        std::vector<uint8_t> message = ReceiveMessage();
        {
          std::lock_guard<std::mutex> lock(queue_mutex);
          if (stopping) {
            return;
          }
          mlvc::Check(message_queue.size() < kMaxQueuedMessages,
                      error_context + " receive queue is full");
          message_queue.push_back(std::move(message));
        }
        queue_cv.notify_one();
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(queue_mutex);
      if (!stopping) {
        receive_error = std::current_exception();
      }
      queue_cv.notify_all();
    }
  }

  std::vector<uint8_t> ReceiveMessage() {
    std::vector<uint8_t> assembled;
    std::vector<bool> received;
    uint16_t expected_packets = 0;
    uint32_t expected_size = 0;
    uint16_t received_count = 0;
    bool assembling = false;
    std::array<uint8_t, kMaxPacketBytes> packet{};
    for (;;) {
      const ssize_t bytes = recvfrom(socket, packet.data(), packet.size(), 0, nullptr, nullptr);
      if (bytes < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        if (assembling) {
          throw mlvc::Error(error_context +
                            " UDP message timed out; packet loss or reordering detected");
        }
        continue;
      }
      mlvc::Check(bytes >= 0, "failed to receive " + error_context + " UDP packet");
      CheckPacket(packet.data(), static_cast<std::size_t>(bytes), error_context);
      const uint16_t packet_index = ReadU16(packet.data() + 2);
      const uint16_t total_packets = ReadU16(packet.data() + 4);
      const uint16_t payload_size = ReadU16(packet.data() + 6);
      const uint32_t total_size = ReadU32(packet.data() + 8);
      mlvc::Check(total_packets != 0 && packet_index < total_packets,
                  "invalid " + error_context + " UDP fragment index");
      mlvc::Check(total_size != 0 && total_size <= kMaxMessageBytes,
                  "invalid " + error_context + " UDP message size");
      if (packet_index == 0) {
        expected_packets = total_packets;
        expected_size = total_size;
        assembled.assign(expected_size, 0);
        received.assign(expected_packets, false);
        received_count = 0;
        assembling = true;
      } else if (total_packets != expected_packets || total_size != expected_size ||
                 received.empty()) {
        continue;
      }
      if (total_packets != expected_packets || total_size != expected_size) {
        continue;
      }
      const std::size_t offset = static_cast<std::size_t>(packet_index) * kPayloadBytes;
      mlvc::Check(offset + payload_size <= assembled.size(),
                  error_context + " UDP fragment exceeds message");
      if (!received[packet_index]) {
        std::memcpy(assembled.data() + offset, packet.data() + kHeaderBytes, payload_size);
        received[packet_index] = true;
        ++received_count;
      }
      if (received_count == expected_packets) {
        assembling = false;
        return assembled;
      }
    }
  }

  int socket = -1;
  std::string error_context;
  std::mutex queue_mutex;
  std::condition_variable queue_cv;
  std::deque<std::vector<uint8_t>> message_queue;
  std::exception_ptr receive_error;
  bool stopping = false;
  std::thread receive_thread;
};

UdpMessageReceiver::UdpMessageReceiver(uint16_t port, std::string error_context)
    : impl_(std::make_unique<Impl>(port, std::move(error_context))) {}

UdpMessageReceiver::~UdpMessageReceiver() = default;

std::vector<uint8_t> UdpMessageReceiver::Receive() { return impl_->Receive(); }

}  // namespace mlvc::transport
