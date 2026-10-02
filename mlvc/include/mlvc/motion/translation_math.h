#ifndef MLVC_MOTION_TRANSLATION_MATH_H_
#define MLVC_MOTION_TRANSLATION_MATH_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlvc::motion {

// Same fields and units as FFmpeg AVMotionVector, independent of FFmpeg headers.
struct MotionVector {
  int32_t source = -1;
  uint8_t w = 0;
  uint8_t h = 0;
  int32_t motion_x = 0;
  int32_t motion_y = 0;
  uint16_t motion_scale = 0;
};

struct Translation {
  int8_t kx = 0;
  int8_t ky = 0;
  double tx = 0;
  double ty = 0;
  std::size_t vector_count = 0;
};

// Half-cell ties round toward zero. Overflow and nonfinite inputs throw.
int8_t NearestFeatureCell(double displacement);
Translation EstimateTranslation(const std::vector<MotionVector>& vectors, bool gop_start);

}  // namespace mlvc::motion

#endif  // MLVC_MOTION_TRANSLATION_MATH_H_
