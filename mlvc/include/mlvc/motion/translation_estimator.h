#ifndef MLVC_MOTION_TRANSLATION_ESTIMATOR_H_
#define MLVC_MOTION_TRANSLATION_ESTIMATOR_H_

#include <mlvc/motion/translation_math.h>

#include <memory>
#include <functional>
#include <string>

namespace mlvc::codec { struct TensorData; }
namespace mlvc::io { struct Nv12Layout; }

namespace mlvc::motion {

struct TranslationEstimatorConfig {
  int width = 0;
  int height = 0;
  int fps = 30;
  int gop = 96;
  std::string preset = "medium";
  int threads = 1;
  // MV-only decode optimization. Reconstructed H.264 pixels are not consumed.
  bool skip_loop_filter = false;
};

// Decode a complete Annex B access unit from any proxy encoder (including DVPP).
// Rejects B frames, unexpected keyframes, and multiple reference frames. The
// caller must keep the hardware encoder alive across frames to retain references.
class H264MotionExtractor {
 public:
  explicit H264MotionExtractor(int gop = 96, bool skip_loop_filter = false);
  ~H264MotionExtractor();
  H264MotionExtractor(const H264MotionExtractor&) = delete;
  H264MotionExtractor& operator=(const H264MotionExtractor&) = delete;
  Translation Decode(const uint8_t* packet, std::size_t bytes, uint64_t frame_index,
                     bool random_access = false, std::vector<MotionVector>* raw_vectors = nullptr);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Synchronous CPU reference proxy: libx264, configurable preset, CRF18, fixed GOP,
// no B frames, one reference frame, zerolatency tune. Lookahead is disabled.
// Presets: medium/veryfast/superfast/ultrafast. Threads: 1..16, slice threads
// only, so each input still produces its own access unit without frame delay.
// Conversion uses the existing MLVC normalized FP16 YUV444 -> NV12 conversion;
// pixel-identical Python comparison must use the same 8-bit input conversion.
class TranslationEstimator {
 public:
  // Optional synchronous observer for tests/replay. Bytes remain borrowed and
  // valid only during this callback; an empty observer adds no packet copy.
  using ProxyPacketObserver =
      std::function<void(const uint8_t*, std::size_t, uint64_t, bool)>;
  explicit TranslationEstimator(TranslationEstimatorConfig config,
                                ProxyPacketObserver observer = {});
  ~TranslationEstimator();
  TranslationEstimator(const TranslationEstimator&) = delete;
  TranslationEstimator& operator=(const TranslationEstimator&) = delete;
  Translation Estimate(const codec::TensorData& input, uint64_t frame_index,
                       bool random_access = false);
  Translation EstimateNv12(const uint8_t* data, std::size_t bytes,
                           const io::Nv12Layout& layout, uint64_t frame_index,
                           bool random_access = false);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mlvc::motion

#endif  // MLVC_MOTION_TRANSLATION_ESTIMATOR_H_
