#include <mlvc/motion/dvpp_translation_estimator.h>
#include <mlvc/runtime/acl_runtime.h>

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
struct DeviceBuffer {
  void* value = nullptr;
  ~DeviceBuffer() { if (value) (void)aclrtFree(value); }
};
}  // namespace

int main(int argc, char** argv) {
  try {
    const int frames = argc > 1 ? std::stoi(argv[1]) : 6;
    const int gop = argc > 2 ? std::stoi(argv[2]) : 4;
    const int width = argc > 3 ? std::stoi(argv[3]) : 640;
    const int height = argc > 4 ? std::stoi(argv[4]) : 384;
    const int forced_i = argc > 5 ? std::stoi(argv[5]) : 2;
    if (argc > 6 || frames < 2 || gop < 1 || width < 128 || height < 128) {
      throw std::runtime_error("usage: check_dvpp_motion_proxy [frames=6] [gop=4] "
                               "[width=640] [height=384] [forced_i=2]");
    }
    mlvc::AclRuntime runtime(0);
    mlvc::io::DvppH264EncoderConfig config;
    config.layout = {width, height, width, height};
    config.gop = gop;
    config.bitrate = 8000000;
    DeviceBuffer device;
    const std::size_t bytes = mlvc::io::Nv12BufferSize(config.layout);
    mlvc::CheckAcl(aclrtMalloc(&device.value, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "allocate proxy NV12");
    mlvc::motion::DvppTranslationEstimator estimator(runtime.context(), config);
    std::vector<uint8_t> base(width * height);
    uint32_t random = 0x12345678;
    for (std::size_t index = 0; index < base.size(); ++index) {
      random ^= random << 13;
      random ^= random >> 17;
      random ^= random << 5;
      const int x = index % width;
      const int y = index / width;
      base[index] = static_cast<uint8_t>(128 + 30 * std::sin(x / 17.0) +
          35 * std::cos(y / 11.0) + 20 * std::sin((x + y) / 27.0) + random % 5);
    }
    std::vector<uint8_t> nv12(bytes, 128);
    for (int frame = 0; frame < frames; ++frame) {
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          const int sx = (x - (16 * frame) % width + width) % width;
          const int sy = (y + 8 * frame) % height;
          nv12[y * width + x] = base[sy * width + sx];
        }
      }
      mlvc::CheckAcl(aclrtMemcpy(device.value, bytes, nv12.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE),
                     "upload synthetic proxy frame");
      const auto shift = estimator.EstimateDevice(device.value, bytes, frame, nullptr, frame == forced_i);
      std::cout << "frame=" << frame << " tx=" << shift.tx << " ty=" << shift.ty
                << " kx=" << static_cast<int>(shift.kx) << " ky=" << static_cast<int>(shift.ky)
                << " vectors=" << shift.vector_count << std::endl;
      const bool reset = frame == forced_i || frame % gop == 0;
      if (shift.kx != (reset ? 0 : 2) || shift.ky != (reset ? 0 : -1) ||
          (!reset && shift.vector_count == 0)) {
        throw std::runtime_error("DVPP proxy translation sign/scale/GOP mismatch");
      }
    }
    std::cout << "dvpp_motion_proxy status=ok\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "dvpp_motion_proxy status=failed: " << error.what() << '\n';
    return 1;
  }
}
