#include <mlvc/motion/translation_math.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mlvc::motion {
namespace {

double WeightedMedian(std::vector<std::pair<double, uint64_t>> values) {
  std::sort(values.begin(), values.end(), [](const auto& a, const auto& b) {
    return a.first < b.first;
  });
  uint64_t total = 0;
  for (const auto& value : values) {
    if (total > std::numeric_limits<uint64_t>::max() - value.second) {
      throw std::overflow_error("motion-vector area sum overflow");
    }
    total += value.second;
  }
  uint64_t cumulative = 0;
  // Integer arithmetic preserves searchsorted(..., total / 2, side='left').
  const uint64_t threshold = total / 2 + total % 2;
  for (const auto& value : values) {
    cumulative += value.second;
    if (cumulative >= threshold) return value.first;
  }
  throw std::runtime_error("weighted median has no positive-weight samples");
}

}  // namespace

int8_t NearestFeatureCell(double displacement) {
  if (!std::isfinite(displacement)) {
    throw std::invalid_argument("motion displacement must be finite");
  }
  const double quantized = std::copysign(
      std::floor(std::abs(displacement) / 8.0 + 0.5 - 1e-9), displacement);
  if (quantized < -128 || quantized > 127) {
    throw std::out_of_range("motion shift exceeds the signed 8-bit side information");
  }
  return static_cast<int8_t>(quantized);
}

Translation EstimateTranslation(const std::vector<MotionVector>& vectors, bool gop_start) {
  Translation result;
  if (gop_start) return result;
  std::vector<std::pair<double, uint64_t>> xs;
  std::vector<std::pair<double, uint64_t>> ys;
  xs.reserve(vectors.size());
  ys.reserve(vectors.size());
  for (const auto& vector : vectors) {
    if (vector.source >= 0) continue;
    if (vector.motion_scale == 0 || vector.w == 0 || vector.h == 0) {
      throw std::invalid_argument("invalid backward-reference motion vector");
    }
    const uint64_t area = static_cast<uint64_t>(vector.w) * vector.h;
    xs.emplace_back(-static_cast<double>(vector.motion_x) / vector.motion_scale, area);
    ys.emplace_back(-static_cast<double>(vector.motion_y) / vector.motion_scale, area);
  }
  result.vector_count = xs.size();
  // A P frame may contain only intra blocks; such a frame has no displacement evidence.
  if (xs.empty()) return result;
  result.tx = WeightedMedian(std::move(xs));
  result.ty = WeightedMedian(std::move(ys));
  result.kx = NearestFeatureCell(result.tx);
  result.ky = NearestFeatureCell(result.ty);
  return result;
}

}  // namespace mlvc::motion
