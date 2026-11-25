#ifndef ULBVC_DETECTION_GEOMETRY_H_
#define ULBVC_DETECTION_GEOMETRY_H_

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "ulbvc/transport/detection_metadata.h"

namespace ulbvc::detection {
struct Letterbox {
  int source_width, source_height, resized_width, resized_height, left, top;
  double gain;
};
inline Letterbox PlanLetterbox(int width, int height, int input_width, int input_height) {
  if (width <= 0 || height <= 0 || input_width <= 0 || input_height <= 0)
    throw std::invalid_argument("letterbox dimensions must be positive");
  const double gain = std::min(static_cast<double>(input_width) / width,
                               static_cast<double>(input_height) / height);
  const int resized_width = std::lround(width * gain);
  const int resized_height = std::lround(height * gain);
  return {width,
          height,
          resized_width,
          resized_height,
          (input_width - resized_width) / 2,
          (input_height - resized_height) / 2,
          gain};
}
inline bool RestoreBox(transport::DetectionBox* box, const Letterbox& plan) {
  const float x1 =
      std::clamp<float>((box->cx - box->width / 2 - plan.left) / plan.gain, 0, plan.source_width);
  const float y1 =
      std::clamp<float>((box->cy - box->height / 2 - plan.top) / plan.gain, 0, plan.source_height);
  const float x2 =
      std::clamp<float>((box->cx + box->width / 2 - plan.left) / plan.gain, 0, plan.source_width);
  const float y2 =
      std::clamp<float>((box->cy + box->height / 2 - plan.top) / plan.gain, 0, plan.source_height);
  box->cx = (x1 + x2) / 2;
  box->cy = (y1 + y2) / 2;
  box->width = x2 - x1;
  box->height = y2 - y1;
  return box->width > 0 && box->height > 0;
}
}  // namespace ulbvc::detection
#endif  // ULBVC_DETECTION_GEOMETRY_H_
