#include <mlvc/motion/translation_estimator.h>
#include <mlvc/io/fp16_yuv444_to_nv12.h>

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <vector>

int main() {
  try {
    constexpr int width = 256;
    constexpr int height = 128;
    constexpr int gop = 4;
    mlvc::motion::TranslationEstimator estimator({width, height, 30, gop});
    const mlvc::io::Nv12Layout layout{width, height, width + 32, height + 16};
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
    std::vector<uint8_t> nv12(mlvc::io::Nv12BufferSize(layout), 128);
    for (int frame = 0; frame < 6; ++frame) {
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          const int sx = (x - 16 * frame + width) % width;
          const int sy = (y + 8 * frame) % height;
          nv12[y * layout.width_stride + x] = base[sy * width + sx];
        }
      }
      const bool forced_i = frame == 2;
      auto shift = estimator.EstimateNv12(nv12.data(), nv12.size(), layout, frame, forced_i);
      const bool reset = forced_i || frame % gop == 0;
      if (shift.kx != (reset ? 0 : 2) || shift.ky != (reset ? 0 : -1) ||
          (!reset && shift.vector_count == 0)) {
        throw std::runtime_error("synthetic translation sign/scale/GOP mismatch at frame " +
                                 std::to_string(frame) + " (" + std::to_string(shift.kx) + "," +
                                 std::to_string(shift.ky) + "), pixels=(" + std::to_string(shift.tx) +
                                 "," + std::to_string(shift.ty) + "), vectors=" +
                                 std::to_string(shift.vector_count));
      }
      std::cout << "frame=" << frame << " tx=" << shift.tx << " ty=" << shift.ty
                << " kx=" << static_cast<int>(shift.kx) << " ky=" << static_cast<int>(shift.ky)
                << " vectors=" << shift.vector_count << '\n';
    }
    bool rejected = false;
    try { estimator.EstimateNv12(nv12.data(), nv12.size(), layout, 8); }
    catch (const std::exception&) { rejected = true; }
    if (!rejected) throw std::runtime_error("nonconsecutive input frame was accepted");
    std::cout << "motion_proxy status=ok\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "motion_proxy status=failed: " << error.what() << '\n';
    return 1;
  }
}
