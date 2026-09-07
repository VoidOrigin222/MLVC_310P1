#include <mlvc/codec/detail/prior/prior_layout.h>

#include <mlvc/codec/tensor_data.h>
#include <mlvc/framework/profile_range.h>

#include <cstddef>
#include <vector>

#include "mlvc/core/status.h"

namespace mlvc::codec {

int Offset4(int channel, int y, int x, int /*channels*/, int height, int width) {
  return ((channel * height) + y) * width + x;
}

int MicroMask4(int mask_index, int chunk, int y, int x) {
  static constexpr int patterns[4][4][2][2] = {
      {{{1, 0}, {0, 0}}, {{0, 1}, {0, 0}}, {{0, 0}, {1, 0}}, {{0, 0}, {0, 1}}},
      {{{0, 0}, {0, 1}}, {{0, 0}, {1, 0}}, {{0, 1}, {0, 0}}, {{1, 0}, {0, 0}}},
      {{{0, 0}, {1, 0}}, {{0, 0}, {0, 1}}, {{1, 0}, {0, 0}}, {{0, 1}, {0, 0}}},
      {{{0, 1}, {0, 0}}, {{1, 0}, {0, 0}}, {{0, 0}, {0, 1}}, {{0, 0}, {1, 0}}},
  };
  return patterns[mask_index][chunk][y & 1][x & 1];
}

int MicroMask2(int mask_index, int chunk, int y, int x) {
  static constexpr int patterns[2][2][2][2] = {
      {{{1, 0}, {0, 1}}, {{0, 1}, {1, 0}}},
      {{{0, 1}, {1, 0}}, {{1, 0}, {0, 1}}},
  };
  return patterns[mask_index][chunk][y & 1][x & 1];
}

void SinglePart4x(const std::vector<float>& full, int mask_index, int channels, int height,
                  int width, std::vector<float>* part) {
  MLVC_PROFILE_RANGE_FUNCTION();
  const int part_channels = channels / 4;
  part->resize(static_cast<std::size_t>(part_channels * height * width));
  for (int c = 0; c < part_channels; ++c) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        float value = 0.0f;
        for (int chunk = 0; chunk < 4; ++chunk) {
          if (MicroMask4(mask_index, chunk, y, x) != 0) {
            value += full[Offset4(chunk * part_channels + c, y, x, channels, height, width)];
          }
        }
        (*part)[Offset4(c, y, x, part_channels, height, width)] = value;
      }
    }
  }
}

void SinglePart2x(const std::vector<float>& full, int mask_index, int channels, int height,
                  int width, std::vector<float>* part) {
  MLVC_PROFILE_RANGE_FUNCTION();
  const int part_channels = channels / 2;
  part->resize(static_cast<std::size_t>(part_channels * height * width));
  for (int c = 0; c < part_channels; ++c) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        float value = 0.0f;
        for (int chunk = 0; chunk < 2; ++chunk) {
          if (MicroMask2(mask_index, chunk, y, x) != 0) {
            value += full[Offset4(chunk * part_channels + c, y, x, channels, height, width)];
          }
        }
        (*part)[Offset4(c, y, x, part_channels, height, width)] = value;
      }
    }
  }
}

void RestoreY4x(const std::vector<int8_t>& y_symbols, const std::vector<float>& means,
                int mask_index, int channels, int height, int width, std::vector<float>* output) {
  MLVC_PROFILE_RANGE_FUNCTION();
  const int part_channels = channels / 4;
  for (int c = 0; c < part_channels; ++c) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const float symbol =
            static_cast<float>(y_symbols[Offset4(c, y, x, part_channels, height, width)]);
        for (int chunk = 0; chunk < 4; ++chunk) {
          if (MicroMask4(mask_index, chunk, y, x) != 0) {
            const int out_channel = chunk * part_channels + c;
            const int index = Offset4(out_channel, y, x, channels, height, width);
            (*output)[index] = RoundToFp16(symbol + means[index]);
          }
        }
      }
    }
  }
}

void RestoreY2x(const std::vector<int8_t>& y_symbols, const std::vector<float>& means,
                int mask_index, int channels, int height, int width, std::vector<float>* output) {
  MLVC_PROFILE_RANGE_FUNCTION();
  const int part_channels = channels / 2;
  for (int c = 0; c < part_channels; ++c) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const float symbol =
            static_cast<float>(y_symbols[Offset4(c, y, x, part_channels, height, width)]);
        for (int chunk = 0; chunk < 2; ++chunk) {
          if (MicroMask2(mask_index, chunk, y, x) != 0) {
            const int out_channel = chunk * part_channels + c;
            const int index = Offset4(out_channel, y, x, channels, height, width);
            (*output)[index] = RoundToFp16(symbol + means[index]);
          }
        }
      }
    }
  }
}

}  // namespace mlvc::codec
