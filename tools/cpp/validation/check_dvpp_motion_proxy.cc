#include <mlvc/motion/dvpp_translation_estimator.h>
#include <mlvc/runtime/acl_runtime.h>
#include "motion_mv_parity.h"

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
    const bool host_upload = argc > 6 ? std::stoi(argv[6]) != 0 : false;
    if (argc > 7 || frames < 2 || gop < 1 || width < 128 || height < 128) {
      throw std::runtime_error("usage: check_dvpp_motion_proxy [frames=6] [gop=4] "
                               "[width=640] [height=384] [forced_i=2] [host_upload=0]");
    }
    mlvc::AclRuntime runtime(0);
    mlvc::io::DvppH264EncoderConfig config;
    config.layout = {width, height, width, height};
    config.gop = gop;
    config.bitrate = 8000000;
    DeviceBuffer device;
    const std::size_t bytes = mlvc::io::Nv12BufferSize(config.layout);
    mlvc::CheckAcl(aclrtMalloc(&device.value, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "allocate proxy NV12");
    auto estimator = std::make_unique<mlvc::motion::DvppTranslationEstimator>(
        runtime.context(), config, true);
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
    for (int invalid = 0; invalid < 3; ++invalid) {
      bool rejected = false;
      try {
        (void)estimator->EstimateNv12Host(invalid == 0 ? nullptr : nv12.data(),
                                         invalid == 1 ? bytes - 1 : bytes,
                                         invalid == 2 ? 1 : 0);
      } catch (const std::runtime_error&) { rejected = true; }
      if (!rejected) throw std::runtime_error("host NV12 accepted invalid input/order");
    }
    for (int frame = 0; frame < frames; ++frame) {
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          const int sx = (x - (16 * frame) % width + width) % width;
          const int sy = (y + 8 * frame) % height;
          nv12[y * width + x] = base[sy * width + sx];
        }
      }
      mlvc::motion::Translation shift;
      if (host_upload) {
        shift = estimator->EstimateNv12Host(nv12.data(), bytes, frame, frame == forced_i);
      } else {
        mlvc::CheckAcl(aclrtMemcpy(device.value, bytes, nv12.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE),
                       "upload synthetic proxy frame");
        shift = estimator->EstimateDevice(device.value, bytes, frame, nullptr, frame == forced_i);
      }
      std::cout << "frame=" << frame << " tx=" << shift.tx << " ty=" << shift.ty
                << " kx=" << static_cast<int>(shift.kx) << " ky=" << static_cast<int>(shift.ky)
                << " vectors=" << shift.vector_count << std::endl;
      const bool reset = frame == forced_i || frame % gop == 0;
      if (shift.kx != (reset ? 0 : 2) || shift.ky != (reset ? 0 : -1) ||
          (!reset && shift.vector_count == 0)) {
        throw std::runtime_error("DVPP proxy translation sign/scale/GOP mismatch");
      }
    }
    // Release the first MPI channel/system owner before starting a same-packet
    // baseline/fast comparison. No duplicated VENC runs can perturb this parity.
    estimator.reset();
    config.persistent_channel = true;
    config.single_reference = true;
    config.zero_copy_input = false;
    config.channel = 1;
    mlvc::io::DvppH264Encoder proxy(runtime.context(), config);
    mlvc::validation::MotionMvParity parity(gop);
    int offset_x = 0, offset_y = 0;
    for (int frame = 0; frame < frames; ++frame) {
      // Repeated positive pan / static / negative pan segments, including
      // retained image-edge regions, IDRs and resets in the same histories.
      const int phase = frame % 30;
      const int dx = frame == 0 || (phase >= 10 && phase < 20) ? 0 : phase < 10 ? 16 : -16;
      const int dy = -dx / 2;
      offset_x = (offset_x + dx + width) % width;
      offset_y = (offset_y + dy + height) % height;
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          const int sx = (x - offset_x + width) % width;
          const int sy = (y - offset_y + height) % height;
          nv12[y * width + x] = base[sy * width + sx];
        }
      }
      const bool random_access = frame == forced_i;
      const auto packet = proxy.EncodeHostNv12(nv12.data(), bytes,
                                               random_access || frame % gop == 0);
      (void)parity.Inspect(packet.data(), packet.size(), frame, random_access);
    }
    if (parity.frames() != static_cast<uint64_t>(frames))
      throw std::runtime_error("not all DVPP packets were checked");
    if (gop > 1 && forced_i != 1) parity.CheckTruncatedPRejected();
    std::cout << "motion_mv_parity backend=dvpp frames=" << parity.frames()
              << " vectors=" << parity.vectors() << " field_hash=" << parity.field_hash()
              << " status=ok\n";
    std::cout << "dvpp_motion_proxy status=ok\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "dvpp_motion_proxy status=failed: " << error.what() << '\n';
    return 1;
  }
}
