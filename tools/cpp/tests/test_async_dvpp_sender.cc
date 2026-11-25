#include <ulbvc/codec/tensor_data.h>
#include <ulbvc/core/status.h>
#include <ulbvc/io/udp_frame_transport.h>
#include <ulbvc/runtime/acl_runtime.h>
#include <ulbvc/transport/udp_message_transport.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <vector>

namespace {

uint32_t ReadU32(const uint8_t* data) {
  return static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8) |
         (static_cast<uint32_t>(data[2]) << 16) | (static_cast<uint32_t>(data[3]) << 24);
}

void CheckVideoTransJpeg(const std::vector<uint8_t>& payload) {
  ulbvc::Check(payload.size() >= 12, "VideoTrans payload is too short");
  const uint32_t jpeg_size = ReadU32(payload.data());
  ulbvc::Check(payload.size() == static_cast<std::size_t>(jpeg_size) + 8,
              "VideoTrans JPEG size mismatch");
  ulbvc::Check(payload[4] == 0xff && payload[5] == 0xd8,
              "VideoTrans JPEG is missing SOI");
  ulbvc::Check(payload[4 + jpeg_size - 2] == 0xff && payload[4 + jpeg_size - 1] == 0xd9,
              "VideoTrans JPEG is missing EOI");
  ulbvc::Check(ReadU32(payload.data() + 4 + jpeg_size) == 0,
              "VideoTrans test expects no detections");
}

}  // namespace

int main() {
  try {
    constexpr uint16_t kPort = 45123;
    ulbvc::AclRuntime runtime(0);
    ulbvc::transport::UdpMessageReceiver receiver(kPort, "DVPP sender test");
    ulbvc::io::AsyncDvppVideoTransUdpSender sender("127.0.0.1", kPort, 128, 128, 1, 75,
                                                   runtime.context());
    sender.SendFrame(ulbvc::codec::MakeFp16Tensor({1, 3, 128, 128}, 0.2F), 0);
    sender.SendFrame(ulbvc::codec::MakeFp16Tensor({1, 3, 128, 128}, 0.5F), 1);
    sender.SendFrame(ulbvc::codec::MakeFp16Tensor({1, 3, 128, 128}, 0.8F), 2);
    sender.Close();
    for (int i = 0; i < 3; ++i) CheckVideoTransJpeg(receiver.Receive());
    const ulbvc::io::DvppForwardStats stats = sender.stats();
    ulbvc::Check(stats.enqueued == 3 && stats.encoded == 3 && stats.sent == 3,
                "DVPP sender did not process all frames");
    ulbvc::Check(stats.failed == 0 && stats.dropped == 0,
                "DVPP sender reported a failure or drop");
    std::cout << "async DVPP sender test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_async_dvpp_sender failed: " << error.what() << "\n";
    return 1;
  }
}
