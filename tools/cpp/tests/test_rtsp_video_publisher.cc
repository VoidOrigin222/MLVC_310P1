#include <mlvc/codec/tensor_utils.h>
#include <mlvc/core/status.h>
#include <mlvc/io/video_io.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

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
    std::cout << "rtsp video publisher test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_rtsp_video_publisher failed: " << error.what() << "\n";
    return 1;
  }
}
#endif
