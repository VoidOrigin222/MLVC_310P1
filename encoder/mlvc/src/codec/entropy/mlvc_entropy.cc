#include "mlvc/codec/mlvc_entropy.h"

#include <mlvc/codec/detail/tensor/tensor_utils.h>

#include <algorithm>
#include <cmath>

#include "mlvc/core/status.h"
#include "mlvc/entropy/entropy_codec.h"
#include "mlvc/framework/profile_range.h"

namespace mlvc::codec {
namespace {

inline std::size_t PlaneSize(int height, int width) {
  return static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
}

}  // namespace

bool IsMlvcIFrame(int frame_index, int gop, int reset_interval) {
  (void)reset_interval;
  if (frame_index == 0) {
    return true;
  }
  if (gop > 0 && frame_index % gop == 0) {
    return true;
  }
  return false;
}

bool ShouldResetReferenceFeature(int frame_index, int gop, int reset_interval) {
  if (IsMlvcIFrame(frame_index, gop, reset_interval)) {
    return true;
  }
  if (reset_interval <= 0) {
    return false;
  }
  const int gop_cycle_index = gop > 0 ? frame_index % gop : frame_index;
  return (gop_cycle_index + 1) % reset_interval == 0;
}

int MlvcResetCycleIndex(int frame_index, int reset_interval) {
  if (reset_interval <= 1 || frame_index <= 0) {
    return frame_index;
  }
  return (frame_index - 1) % reset_interval + 1;
}

bool ShouldUseLtrFeatures(int cycle_index, int ltr_start_idx, int ltr_period) {
  if (ltr_period <= 0) {
    return false;
  }
  return (cycle_index > ltr_start_idx) && (cycle_index % ltr_period == 0);
}

bool ShouldSaveLtrFeatures(int cycle_index, int ltr_start_idx, int ltr_period) {
  if (ltr_period <= 0) {
    return false;
  }
  return (cycle_index == ltr_start_idx) ||
         ShouldUseLtrFeatures(cycle_index, ltr_start_idx, ltr_period);
}

MlvcFrameType MlvcFrameTypeForFrame(int frame_index, int gop, int reset_interval, int ltr_start_idx,
                                    int ltr_period) {
  if (IsMlvcIFrame(frame_index, gop, reset_interval)) {
    return MlvcFrameType::kIFrame;
  }
  const int cycle_index = gop > 0 ? frame_index % gop : frame_index;
  if (ShouldUseLtrFeatures(cycle_index, ltr_start_idx, ltr_period)) {
    return MlvcFrameType::kLtrRecovery;
  }
  return MlvcFrameType::kPFrame;
}

std::string_view MlvcFrameTypeName(MlvcFrameType frame_type) {
  switch (frame_type) {
    case MlvcFrameType::kIFrame:
      return "i";
    case MlvcFrameType::kPFrame:
      return "p";
    case MlvcFrameType::kLtrRecovery:
      return "ltr";
  }
  return "p";
}

std::string_view MlvcEntropyPrefix(bool is_i_frame) { return is_i_frame ? "i" : "p"; }

bool MlvcUseTwoParts(int width, int height) { return width * height > 1280 * 720; }

void BuildMlvcScaleIndexesFromZRaw(const TensorData& z_raw, int y_channels, int y_height,
                                   int y_width, int channel_repeat, std::vector<uint8_t>* scales_0,
                                   std::vector<uint8_t>* scales_1) {
  MLVC_PROFILE_RANGE_FUNCTION();
  Check(scales_0 != nullptr && scales_1 != nullptr, "scale outputs are required");
  Check(z_raw.dtype == mlvc::DataType::kFloat16, "z_raw must be float16");
  Check(z_raw.shape.dims().size() == 4, "z_raw must be rank-4");
  Check(y_channels > 0 && y_height > 0 && y_width > 0, "invalid y shape");
  Check(y_channels % 2 == 0, "y channels must be even");
  Check(channel_repeat > 0 && y_channels % channel_repeat == 0,
        "y channels must be divisible by channel repeat");

  const int z_channels = static_cast<int>(z_raw.shape.dim(1));
  const int z_height = static_cast<int>(z_raw.shape.dim(2));
  const int z_width = static_cast<int>(z_raw.shape.dim(3));
  const int base_channels = y_channels / channel_repeat;
  Check(base_channels <= z_channels, "z_raw channel count is too small for scale decoding");

  std::vector<float> z_values;
  Fp16ToFloat(z_raw, &z_values);

  const int spatial_repeat_h = (y_height + z_height - 1) / z_height;
  const int spatial_repeat_w = (y_width + z_width - 1) / z_width;
  const int spatial_repeat = std::max(spatial_repeat_h, spatial_repeat_w);
  const std::size_t plane = PlaneSize(y_height, y_width);

  const int half_channels = y_channels / 2;
  scales_0->assign(static_cast<std::size_t>(half_channels) * plane, 0);
  scales_1->assign(static_cast<std::size_t>(half_channels) * plane, 0);
  // Write both checkerboard streams directly.  Materializing all y channels
  // and then splitting them required a second full-resolution traversal.
  std::vector<int> source_y(static_cast<std::size_t>(y_height));
  std::vector<int> source_x(static_cast<std::size_t>(y_width));
  for (int y = 0; y < y_height; ++y) {
    source_y[static_cast<std::size_t>(y)] = std::min(y / spatial_repeat, z_height - 1);
  }
  for (int x = 0; x < y_width; ++x) {
    source_x[static_cast<std::size_t>(x)] = std::min(x / spatial_repeat, z_width - 1);
  }
  for (int channel = 0; channel < half_channels; ++channel) {
    const std::size_t dst_base = static_cast<std::size_t>(channel) * plane;
    const int source_channel_0 = channel / channel_repeat;
    const int source_channel_1 = (channel + half_channels) / channel_repeat;
    for (int y = 0; y < y_height; ++y) {
      const bool even_row = (y & 1) == 0;
      const std::size_t source_row_0 =
          (static_cast<std::size_t>(source_channel_0) * static_cast<std::size_t>(z_height) +
           static_cast<std::size_t>(source_y[static_cast<std::size_t>(y)])) *
          static_cast<std::size_t>(z_width);
      const std::size_t source_row_1 =
          (static_cast<std::size_t>(source_channel_1) * static_cast<std::size_t>(z_height) +
           static_cast<std::size_t>(source_y[static_cast<std::size_t>(y)])) *
          static_cast<std::size_t>(z_width);
      for (int x = 0; x < y_width; ++x) {
        const bool even_col = (x & 1) == 0;
        const bool mask0 = (even_row == even_col);
        const std::size_t offset = static_cast<std::size_t>(y) * static_cast<std::size_t>(y_width) +
                                   static_cast<std::size_t>(x);
        const std::size_t source_offset =
            static_cast<std::size_t>(source_x[static_cast<std::size_t>(x)]);
        const int scale_0 = std::clamp(
            static_cast<int>(std::abs(z_values[source_row_0 + source_offset])), 0, kScaleLevel - 1);
        const int scale_1 = std::clamp(
            static_cast<int>(std::abs(z_values[source_row_1 + source_offset])), 0, kScaleLevel - 1);
        (*scales_0)[dst_base + offset] = static_cast<uint8_t>(mask0 ? scale_0 : scale_1);
        (*scales_1)[dst_base + offset] = static_cast<uint8_t>(mask0 ? scale_1 : scale_0);
      }
    }
  }
}

void BuildMlvcCombinedYIndexes(const std::vector<int8_t>& symbols,
                               const std::vector<uint8_t>& scales, std::vector<int16_t>* indexes) {
  MLVC_PROFILE_RANGE_FUNCTION();
  Check(indexes != nullptr, "index output is required");
  Check(symbols.size() == scales.size(), "symbols/scales size mismatch");
  indexes->resize(symbols.size());
  for (std::size_t i = 0; i < symbols.size(); ++i) {
    const int32_t symbol = static_cast<int32_t>(symbols[i]);
    const int32_t scale = static_cast<int32_t>(scales[i]);
    (*indexes)[i] = static_cast<int16_t>(symbol * 256 + scale);
  }
}

}  // namespace mlvc::codec
