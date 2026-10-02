#ifndef MLVC_MOTION_TRANSLATION_ESTIMATOR_H_
#define MLVC_MOTION_TRANSLATION_ESTIMATOR_H_

#include <mlvc/motion/translation_math.h>

#include <memory>

namespace mlvc::codec { struct TensorData; }
namespace mlvc::io { struct Nv12Layout; }

namespace mlvc::motion {

struct TranslationEstimatorConfig {
  int width = 0;
  int height = 0;
  int fps = 30;
  int gop = 96;
};

// Decode a complete Annex B access unit from any proxy encoder (including DVPP).
// Rejects B frames, unexpected keyframes, and multiple reference frames. The
// caller must keep the hardware encoder alive across frames to retain references.
class H264MotionExtractor {
 public:
  explicit H264MotionExtractor(int gop = 96);
  ~H264MotionExtractor();
  H264MotionExtractor(const H264MotionExtractor&) = delete;
  H264MotionExtractor& operator=(const H264MotionExtractor&) = delete;
  Translation Decode(const uint8_t* packet, std::size_t bytes, uint64_t frame_index,
                     bool random_access = false);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Synchronous CPU reference proxy: libx264, medium preset, CRF18, fixed GOP,
// no B frames, one reference frame, zerolatency tune. Lookahead is disabled.
// Conversion uses the existing MLVC normalized FP16 YUV444 -> NV12 conversion;
// pixel-identical Python comparison must use the same 8-bit input conversion.
class TranslationEstimator {
 public:
  explicit TranslationEstimator(TranslationEstimatorConfig config);
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
