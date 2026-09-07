#ifndef MLVC_TRANSPORT_UDP_MESSAGE_TRANSPORT_H_
#define MLVC_TRANSPORT_UDP_MESSAGE_TRANSPORT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mlvc::transport {

// VideoTrans-compatible fragmented UDP transport. The transport carries opaque
// messages; MLVC-specific serialization belongs in mlvc/io.
class UdpMessageSender {
 public:
  UdpMessageSender(const std::string& host, uint16_t port,
                   std::string error_context = "UDP sender");
  ~UdpMessageSender();

  UdpMessageSender(const UdpMessageSender&) = delete;
  UdpMessageSender& operator=(const UdpMessageSender&) = delete;

  void Send(const std::vector<uint8_t>& message, uint8_t channel = 0);
  void Flush();

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
