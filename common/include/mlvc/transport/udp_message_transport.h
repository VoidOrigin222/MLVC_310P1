#ifndef MLVC_TRANSPORT_UDP_MESSAGE_TRANSPORT_H_
#define MLVC_TRANSPORT_UDP_MESSAGE_TRANSPORT_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mlvc::transport {
struct UdpTransportStats {
  uint64_t wire_bytes = 0;
  uint64_t packets = 0;
  uint64_t max_burst_bytes = 0;
  uint64_t max_queue_delay_us = 0;
  uint64_t average_queue_delay_us = 0;
  uint64_t socket_block_us = 0;
};

struct UdpSendOptions {
  uint64_t pacing_rate_bps = 0;
  std::size_t max_burst_bytes = 4096;
  std::size_t max_queue_bytes = 4u * 1024u * 1024u;
  uint64_t max_queue_delay_ms = 1000;
  // Optional deterministic failure hook used by transport tests.
  std::function<void()> before_send;
};

// VideoTrans-compatible fragmented UDP transport. The transport carries opaque
// messages; MLVC-specific serialization belongs in mlvc/io.
class UdpMessageSender {
 public:
  UdpMessageSender(const std::string& host, uint16_t port, std::string error_context = "UDP sender",
                   UdpSendOptions options = {});
  ~UdpMessageSender();

  UdpMessageSender(const UdpMessageSender&) = delete;
  UdpMessageSender& operator=(const UdpMessageSender&) = delete;

  void Send(const std::vector<uint8_t>& message, uint8_t channel = 0);
  void Flush();
  UdpTransportStats Stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class UdpMessageReceiver {
 public:
  explicit UdpMessageReceiver(uint16_t port, std::string error_context = "UDP receiver");
  ~UdpMessageReceiver();

  UdpMessageReceiver(const UdpMessageReceiver&) = delete;
  UdpMessageReceiver& operator=(const UdpMessageReceiver&) = delete;

  std::vector<uint8_t> Receive();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mlvc::transport

#endif  // MLVC_TRANSPORT_UDP_MESSAGE_TRANSPORT_H_
