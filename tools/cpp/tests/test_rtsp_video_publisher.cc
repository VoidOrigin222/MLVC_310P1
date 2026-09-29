#include <mlvc/codec/tensor_utils.h>
#include <mlvc/core/status.h>
#include <mlvc/io/video_io.h>

#include <cstdint>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
int main() {
  std::cout << "rtsp publisher test skipped on Windows\n";
  return 0;
}
#else
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

    mlvc::io::RtspVideoPublisher publisher("rtsp://127.0.0.1:8554/mlvc", 30.0, 2, 2, "ultrafast", 0,
                                           2);
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

    const std::filesystem::path h264_output = root / "stream.h264";
    setenv("MLVC_RTSP_TEST_OUTPUT", h264_output.c_str(), 1);
    mlvc::io::RtspVideoPublisher h264_publisher("rtsp://127.0.0.1:8554/mlvc", 30.0, 2, 2,
                                                "ultrafast", 0, 2, "udp", true);
    const std::vector<std::uint8_t> h264_frame{0, 0, 0, 1, 0x67, 0x42, 0, 0x0a,
                                                0, 0, 0, 1, 0x68, 0xce, 0x31, 0xb2};
    h264_publisher.WriteH264Frame(h264_frame);
    h264_publisher.Close();
    Expect(h264_publisher.frame_count() == 1, "RTSP H.264 publisher frame count mismatch");
    Expect(h264_publisher.dropped_frames() == 0, "unexpected RTSP H.264 publisher drop");
    Expect(std::filesystem::file_size(h264_output) == h264_frame.size(),
           "RTSP H.264 publisher output size mismatch");
    std::ifstream h264_argument_file(arguments);
    const std::string h264_argument_text((std::istreambuf_iterator<char>(h264_argument_file)),
                                         std::istreambuf_iterator<char>());
    Expect(h264_argument_text.find("-f\nh264\n") != std::string::npos,
           "RTSP H.264 publisher did not select H.264 input");
    Expect(h264_argument_text.find("-c:v\ncopy\n") != std::string::npos,
           "RTSP H.264 publisher did not copy the encoded stream");

    {
      std::ofstream file(script);
      file << "#!/bin/sh\nexit 0\n";
    }
    chmod(script.c_str(), 0755);
    mlvc::io::RtspVideoPublisher broken_pipe_publisher(
        "rtsp://127.0.0.1:8554/mlvc", 30.0, 2, 2, "ultrafast", 0, 2, "udp", true);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    broken_pipe_publisher.WriteH264Frame(std::vector<std::uint8_t>(1U << 20, 0x55));
    bool broken_pipe_reported = false;
    try {
      broken_pipe_publisher.Close();
    } catch (const std::exception&) {
      broken_pipe_reported = true;
    }
    Expect(broken_pipe_reported,
           "RTSP publisher should report an exited FFmpeg child as a local pipe error");
    std::cout << "rtsp video publisher test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_rtsp_video_publisher failed: " << error.what() << "\n";
    return 1;
  }
}
#endif
