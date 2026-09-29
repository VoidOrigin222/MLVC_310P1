#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/videodev2.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace {

void CheckIoctl(int fd, unsigned long request, void* arg, const char* name) {
  if (ioctl(fd, request, arg) < 0) {
    std::ostringstream message;
    message << name << " failed: " << std::strerror(errno);
    throw std::runtime_error(message.str());
  }
}

uint32_t Fourcc(const std::string& value) {
  if (value.size() != 4) throw std::runtime_error("pixfmt must contain four characters");
  return v4l2_fourcc(value[0], value[1], value[2], value[3]);
}

struct Buffer {
  void* address = MAP_FAILED;
  size_t length = 0;
};

}  // namespace

int main(int argc, char** argv) {
  const std::string device = argc > 1 ? argv[1] : "/dev/video0";
  const int width = argc > 2 ? std::stoi(argv[2]) : 640;
  const int height = argc > 3 ? std::stoi(argv[3]) : 480;
  const std::string pixfmt_text = argc > 4 ? argv[4] : "MJPG";
  const std::string output = argc > 5 ? argv[5] : "camera_frame.bin";

  int fd = open(device.c_str(), O_RDWR | O_NONBLOCK);
  if (fd < 0) {
    std::cerr << "open " << device << " failed: " << std::strerror(errno) << "\n";
    return 1;
  }

  std::vector<Buffer> buffers;
  try {
    v4l2_capability capability{};
    CheckIoctl(fd, VIDIOC_QUERYCAP, &capability, "VIDIOC_QUERYCAP");
    if (!(capability.capabilities & V4L2_CAP_VIDEO_CAPTURE) ||
        !(capability.capabilities & V4L2_CAP_STREAMING)) {
      throw std::runtime_error("device does not support video capture and streaming");
    }

    v4l2_format format{};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = static_cast<uint32_t>(width);
    format.fmt.pix.height = static_cast<uint32_t>(height);
    format.fmt.pix.pixelformat = Fourcc(pixfmt_text);
    format.fmt.pix.field = V4L2_FIELD_ANY;
    CheckIoctl(fd, VIDIOC_S_FMT, &format, "VIDIOC_S_FMT");
    std::cout << "device=" << device << " driver=" << capability.driver
              << " card=" << capability.card << "\n"
              << "format=" << static_cast<char>(format.fmt.pix.pixelformat & 0xff)
              << static_cast<char>((format.fmt.pix.pixelformat >> 8) & 0xff)
              << static_cast<char>((format.fmt.pix.pixelformat >> 16) & 0xff)
              << static_cast<char>((format.fmt.pix.pixelformat >> 24) & 0xff)
              << " width=" << format.fmt.pix.width << " height=" << format.fmt.pix.height
              << " bytesperline=" << format.fmt.pix.bytesperline
              << " sizeimage=" << format.fmt.pix.sizeimage << "\n";

    v4l2_requestbuffers request{};
    request.count = 4;
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_MMAP;
    CheckIoctl(fd, VIDIOC_REQBUFS, &request, "VIDIOC_REQBUFS");
    if (request.count == 0) throw std::runtime_error("driver returned zero buffers");
    buffers.resize(request.count);

    for (uint32_t i = 0; i < request.count; ++i) {
      v4l2_buffer buffer{};
      buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buffer.memory = V4L2_MEMORY_MMAP;
      buffer.index = i;
      CheckIoctl(fd, VIDIOC_QUERYBUF, &buffer, "VIDIOC_QUERYBUF");
      buffers[i].length = buffer.length;
      buffers[i].address = mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED,
                                 fd, buffer.m.offset);
      if (buffers[i].address == MAP_FAILED) throw std::runtime_error("mmap failed");
      CheckIoctl(fd, VIDIOC_QBUF, &buffer, "VIDIOC_QBUF");
    }

    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    CheckIoctl(fd, VIDIOC_STREAMON, &type, "VIDIOC_STREAMON");
    bool streaming = true;
    for (int attempt = 0; attempt < 100; ++attempt) {
      v4l2_buffer buffer{};
      buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buffer.memory = V4L2_MEMORY_MMAP;
      if (ioctl(fd, VIDIOC_DQBUF, &buffer) < 0) {
        if (errno == EAGAIN) {
          usleep(10000);
          continue;
        }
        throw std::runtime_error(std::string("VIDIOC_DQBUF failed: ") + std::strerror(errno));
      }
      std::ofstream file(output, std::ios::binary);
      file.write(static_cast<const char*>(buffers[buffer.index].address), buffer.bytesused);
      if (!file) throw std::runtime_error("failed to write output frame");
      std::cout << "captured_bytes=" << buffer.bytesused << " output=" << output << "\n";
      CheckIoctl(fd, VIDIOC_QBUF, &buffer, "VIDIOC_QBUF");
      CheckIoctl(fd, VIDIOC_STREAMOFF, &type, "VIDIOC_STREAMOFF");
      streaming = false;
      break;
    }
    if (streaming) throw std::runtime_error("timed out waiting for a camera frame");
    for (const Buffer& buffer : buffers) munmap(buffer.address, buffer.length);
    close(fd);
    return 0;
  } catch (const std::exception& error) {
    for (const Buffer& buffer : buffers) {
      if (buffer.address != MAP_FAILED) munmap(buffer.address, buffer.length);
    }
    close(fd);
    std::cerr << "camera_v4l2_probe: " << error.what() << "\n";
    return 1;
  }
}
