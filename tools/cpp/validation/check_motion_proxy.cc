#include <mlvc/motion/translation_estimator.h>
#include <mlvc/io/fp16_yuv444_to_nv12.h>
#include "motion_mv_parity.h"

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
void CheckProxy(const std::string& preset, int threads) {
    const int width = threads == 1 ? 256 : 512;
    const int height = threads == 1 ? 128 : 256;
    constexpr int gop = 4;
    mlvc::validation::MotionMvParity parity(gop);
    mlvc::motion::Translation observed;
    mlvc::motion::TranslationEstimator estimator({width, height, 30, gop, preset, threads},
        [&](const uint8_t* packet, std::size_t bytes, uint64_t index, bool random_access) {
          observed = parity.Inspect(packet, bytes, index, random_access);
        });
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
    constexpr int frames = 14;
    int offset_x = 0, offset_y = 0;
    for (int frame = 0; frame < frames; ++frame) {
      const int dx = frame == 0 || frame == 10 ? 0 : frame >= 11 ? -16 : 16;
      const int dy = frame == 0 || frame == 10 ? 0 : frame >= 11 ? 8 : -8;
      offset_x = (offset_x + dx + width) % width;
      offset_y = (offset_y + dy + height) % height;
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          const int sx = (x - offset_x + width) % width;
          const int sy = (y - offset_y + height) % height;
          nv12[y * layout.width_stride + x] = base[sy * width + sx];
        }
      }
      const bool forced_i = frame == 2;
      auto shift = estimator.EstimateNv12(nv12.data(), nv12.size(), layout, frame, forced_i);
      const bool reset = forced_i || frame % gop == 0;
      if (shift.kx != (reset ? 0 : dx / 8) || shift.ky != (reset ? 0 : dy / 8) ||
          (!reset && shift.vector_count == 0)) {
        throw std::runtime_error("synthetic translation sign/scale/GOP mismatch at frame " +
                                 std::to_string(frame) + " (" + std::to_string(shift.kx) + "," +
                                 std::to_string(shift.ky) + "), pixels=(" + std::to_string(shift.tx) +
                                 "," + std::to_string(shift.ty) + "), vectors=" +
                                 std::to_string(shift.vector_count));
      }
      if (shift.kx != observed.kx || shift.ky != observed.ky || shift.tx != observed.tx ||
          shift.ty != observed.ty || shift.vector_count != observed.vector_count) {
        throw std::runtime_error("production x264 extractor differs from observed baseline");
      }
      std::cout << "preset=" << preset << " threads=" << threads
                << " frame=" << frame << " tx=" << shift.tx << " ty=" << shift.ty
                << " kx=" << static_cast<int>(shift.kx) << " ky=" << static_cast<int>(shift.ky)
                << " vectors=" << shift.vector_count << '\n';
    }
    bool rejected = false;
    try { estimator.EstimateNv12(nv12.data(), nv12.size(), layout, frames + 1); }
    catch (const std::exception&) { rejected = true; }
    if (!rejected) throw std::runtime_error("nonconsecutive input frame was accepted");
    if (parity.frames() != frames) throw std::runtime_error("not all x264 packets were observed");
    parity.CheckTruncatedPRejected();
    std::cout << "motion_mv_parity backend=libx264 preset=" << preset << " threads=" << threads
              << " frames=" << parity.frames() << " vectors=" << parity.vectors()
              << " field_hash=" << parity.field_hash() << " status=ok\n";
}
}  // namespace

int main() {
  try {
    CheckProxy("medium", 1);
    CheckProxy("veryfast", 4);
    CheckProxy("superfast", 4);
    CheckProxy("ultrafast", 4);
    for (const auto& config : std::vector<mlvc::motion::TranslationEstimatorConfig>{
             {256, 128, 30, 4, "slow", 1}, {256, 128, 30, 4, "medium", 0},
             {256, 128, 30, 4, "medium", 17}}) {
      bool rejected = false;
      try { mlvc::motion::TranslationEstimator invalid(config); }
      catch (const std::exception&) { rejected = true; }
      if (!rejected) throw std::runtime_error("invalid preset/thread config was accepted");
    }
    std::cout << "motion_proxy status=ok\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "motion_proxy status=failed: " << error.what() << '\n';
    return 1;
  }
}
