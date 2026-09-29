#include <mlvc/codec/tensor_utils.h>
#include <mlvc/core/status.h>
#include <mlvc/io/video_io.h>

#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
int main() {
  std::cout << "rtsp publisher test skipped on Windows\n";
  return 0;
}
#elif defined(MLVC_HAS_LIBAVFORMAT)

#include <algorithm>
#include <arpa/inet.h>
#include <array>
#include <atomic>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

void Expect(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

class RtspTestServer {
 public:
  RtspTestServer() {
    listener_ = socket(AF_INET, SOCK_STREAM, 0);
    Expect(listener_ >= 0, "create RTSP test socket");
    int reuse = 1;
    setsockopt(listener_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    Expect(bind(listener_, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0,
           "bind RTSP test socket");
    socklen_t length = sizeof(address);
    Expect(getsockname(listener_, reinterpret_cast<sockaddr*>(&address), &length) == 0,
           "query RTSP test port");
    port_ = ntohs(address.sin_port);
    Expect(listen(listener_, 1) == 0, "listen on RTSP test socket");
    worker_ = std::thread(&RtspTestServer::Run, this);
  }

  RtspTestServer(const RtspTestServer&) = delete;
  RtspTestServer& operator=(const RtspTestServer&) = delete;

  ~RtspTestServer() {
    stopping_.store(true);
    if (client_ >= 0) shutdown(client_, SHUT_RDWR);
    if (listener_ >= 0) shutdown(listener_, SHUT_RDWR);
    if (worker_.joinable()) worker_.join();
    if (client_ >= 0) close(client_);
    if (listener_ >= 0) close(listener_);
  }

  uint16_t port() const { return port_; }
  int request_count() const { return request_count_.load(); }
  int rtp_packet_count() const { return rtp_packet_count_.load(); }

 private:
  static std::string HeaderValue(const std::string& request, const std::string& name) {
    const std::string prefix = name + ":";
    std::size_t begin = request.find(prefix);
    if (begin == std::string::npos) return {};
    begin += prefix.size();
    while (begin < request.size() && request[begin] == ' ') ++begin;
    const std::size_t end = request.find("\r\n", begin);
    return request.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
  }

  void Reply(const std::string& method, const std::string& cseq) {
    std::string response = "RTSP/1.0 200 OK\r\nCSeq: " + cseq + "\r\n";
    if (method == "OPTIONS") {
      response += "Public: OPTIONS, ANNOUNCE, SETUP, RECORD, TEARDOWN\r\n";
    } else if (method == "SETUP") {
      response += "Session: 12345678\r\n"
                  "Transport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n";
    } else if (method == "RECORD") {
      response += "Session: 12345678\r\nRange: npt=0.000-\r\n";
    } else if (method == "TEARDOWN") {
      response += "Session: 12345678\r\n";
    }
    response += "\r\n";
    const char* data = response.data();
    std::size_t remaining = response.size();
    while (remaining != 0) {
      const ssize_t written = send(client_, data, remaining, MSG_NOSIGNAL);
      if (written <= 0) return;
      data += written;
      remaining -= static_cast<std::size_t>(written);
    }
  }

  void Run() {
    sockaddr_in peer{};
    socklen_t length = sizeof(peer);
    client_ = accept(listener_, reinterpret_cast<sockaddr*>(&peer), &length);
    if (client_ < 0) return;
    std::vector<uint8_t> pending;
    std::array<uint8_t, 8192> buffer{};
    const std::string delimiter = "\r\n\r\n";
    while (!stopping_.load()) {
      const ssize_t received = recv(client_, buffer.data(), buffer.size(), 0);
      if (received <= 0) break;
      pending.insert(pending.end(), buffer.data(), buffer.data() + received);
      for (;;) {
        if (!pending.empty() && pending[0] == '$') {
          if (pending.size() < 4) break;
          const std::size_t packet_size =
              (static_cast<std::size_t>(pending[2]) << 8) | pending[3];
          if (pending.size() < 4 + packet_size) break;
          pending.erase(pending.begin(), pending.begin() + 4 + packet_size);
          ++rtp_packet_count_;
          continue;
        }
        const auto end = std::search(pending.begin(), pending.end(), delimiter.begin(),
                                     delimiter.end());
        if (end == pending.end()) break;
        const std::size_t header_size = static_cast<std::size_t>(end - pending.begin()) + 4;
        const std::string request(pending.begin(), pending.begin() + header_size);
        const std::string length_text = HeaderValue(request, "Content-Length");
        const std::size_t content_length = length_text.empty() ? 0 : std::stoul(length_text);
        if (pending.size() < header_size + content_length) break;
        const std::size_t first_space = request.find(' ');
        const std::string method = request.substr(0, first_space);
        Reply(method, HeaderValue(request, "CSeq"));
        ++request_count_;
        pending.erase(pending.begin(), pending.begin() + header_size + content_length);
        if (method == "TEARDOWN") return;
      }
    }
  }

  int listener_ = -1;
  int client_ = -1;
  uint16_t port_ = 0;
  std::thread worker_;
  std::atomic<bool> stopping_{false};
  std::atomic<int> request_count_{0};
  std::atomic<int> rtp_packet_count_{0};
};

std::vector<uint8_t> SampleH264() {
  return {0, 0, 0, 1, 0x67, 0x42, 0x00, 0x0a, 0xe8, 0x40, 0x28, 0x02,
          0xdd, 0x80, 0x80, 0x80, 0xa0, 0, 0, 0, 1, 0x68, 0xce, 0x31,
          0xb2, 0, 0, 0, 0, 1, 0x65, 0x88, 0x84, 0x00, 0x01};
}

}  // namespace

int main() {
  try {
    RtspTestServer server;
    mlvc::io::RtspVideoPublisher publisher(
        "rtsp://127.0.0.1:" + std::to_string(server.port()) + "/mlvc", 30.0, 2, 2,
        "ultrafast", 23, 2, "tcp", true);
    publisher.WriteH264Frame(SampleH264());
    publisher.Close();
    Expect(publisher.frame_count() == 1, "direct RTSP H.264 frame count mismatch");
    Expect(publisher.dropped_frames() == 0, "unexpected direct RTSP H.264 drop");
    Expect(server.request_count() >= 4, "RTSP handshake was incomplete");
    Expect(server.rtp_packet_count() > 0, "direct RTSP publisher wrote no RTP packets");

    RtspTestServer nv12_server;
    mlvc::io::RtspVideoPublisher nv12_publisher(
        "rtsp://127.0.0.1:" + std::to_string(nv12_server.port()) + "/mlvc", 30.0, 2, 2,
        "ultrafast", 23, 2, "tcp", false);
    nv12_publisher.WriteNv12Frame(std::vector<uint8_t>(6, 128));
    nv12_publisher.Close();
    Expect(nv12_publisher.frame_count() == 1, "direct RTSP NV12 frame count mismatch");
    Expect(nv12_server.rtp_packet_count() > 0, "direct RTSP encoder wrote no RTP packets");
    std::cout << "direct libavformat RTSP publisher test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "direct RTSP publisher test failed: " << error.what() << "\n";
    return 1;
  }
}

#else

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sys/stat.h>

namespace {
void Expect(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
}  // namespace

int main() {
  try {
    const std::filesystem::path root = "/tmp/mlvc_rtsp_publisher_test";
    std::filesystem::create_directories(root / "bin");
    const std::filesystem::path script = root / "bin" / "ffmpeg";
    const std::filesystem::path output = root / "stream.bin";
    const std::filesystem::path arguments = root / "arguments.txt";
    {
      std::ofstream file(script);
      file << "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$MLVC_RTSP_TEST_ARGUMENTS\"\n"
              "cat > \"$MLVC_RTSP_TEST_OUTPUT\"\n";
    }
    chmod(script.c_str(), 0755);
    setenv("MLVC_RTSP_TEST_OUTPUT", output.c_str(), 1);
    setenv("MLVC_RTSP_TEST_ARGUMENTS", arguments.c_str(), 1);
    const std::string path =
        std::string("/tmp/mlvc_rtsp_publisher_test/bin:") + std::getenv("PATH");
    setenv("PATH", path.c_str(), 1);

    mlvc::codec::TensorData frame = mlvc::codec::MakeFp16Tensor({1, 3, 2, 2}, 0.0F);
    auto* data = reinterpret_cast<uint16_t*>(frame.bytes.data());
    data[0] = mlvc::codec::FloatToHalfBits(0.5F);
    data[4] = mlvc::codec::FloatToHalfBits(0.5F);
    data[8] = mlvc::codec::FloatToHalfBits(0.5F);

    mlvc::io::RtspVideoPublisher publisher("rtsp://127.0.0.1:8554/mlvc", 30.0, 2, 2,
                                           "ultrafast", 0, 2);
    publisher.WriteTensorFrame(frame);
    publisher.Close();
    Expect(publisher.frame_count() == 1, "RTSP publisher frame count mismatch");
    Expect(publisher.dropped_frames() == 0, "unexpected RTSP publisher drop");
    Expect(std::filesystem::file_size(output) == 6, "RTSP publisher output size mismatch");
    std::ifstream argument_file(arguments);
    const std::string argument_text((std::istreambuf_iterator<char>(argument_file)),
                                    std::istreambuf_iterator<char>());
    Expect(argument_text.find("-rtsp_transport\nudp\n") != std::string::npos,
           "RTSP publisher did not select UDP transport");
    std::cout << "subprocess RTSP publisher test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_rtsp_video_publisher failed: " << error.what() << "\n";
    return 1;
  }
}
#endif
