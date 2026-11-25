#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "ulbvc/codec/tensor_data.h"
#include "ulbvc/core/status.h"
#include "ulbvc/io/bgr_to_fp16_yuv444.h"

int main() {
  try {
    std::mt19937 random(8104);
    float maximum_error = 0;
    for (int width : {1, 14, 16, 18, 32, 130, 1920}) {
      const int height = 24, padded_width = width + 4, padded_height = height + 8;
      const std::size_t stride = width * 3 + 16, plane = padded_width * padded_height;
      std::vector<uint8_t> bgr(stride * height);
      for (auto& value : bgr) value = random() & 255;
      std::vector<uint16_t> result(plane * 3);
      ulbvc::io::ConvertBgrToFp16Yuv444(bgr.data(), stride, width, height, padded_width,
                                        padded_height, result.data());
      for (int y = 0; y < padded_height; ++y) {
        for (int x = 0; x < padded_width; ++x) {
          const auto* pixel =
              bgr.data() + std::min(y, height - 1) * stride + std::min(x, width - 1) * 3;
          const float b = pixel[0] / 255.0f, g = pixel[1] / 255.0f, r = pixel[2] / 255.0f;
          const float yy = 0.2126f * r + 0.7152f * g + 0.0722f * b;
          const float expected[] = {yy, 0.5f * (b - yy) / (1 - 0.0722f) + 0.5f,
                                    0.5f * (r - yy) / (1 - 0.2126f) + 0.5f};
          for (int c = 0; c < 3; ++c) {
            const float actual =
                ulbvc::codec::HalfBitsToFloat(result[c * plane + y * padded_width + x]);
            const float reference =
                ulbvc::codec::HalfBitsToFloat(ulbvc::codec::FloatToHalfBits(expected[c]));
            maximum_error = std::max(maximum_error, std::abs(actual - reference));
            ulbvc::Check(std::isfinite(actual) && std::abs(actual - reference) <= 0.00048828125f,
                         "camera SIMD conversion exceeds one FP16 rounding step");
          }
        }
      }
    }
    std::cout << "camera conversion max absolute error=" << maximum_error
              << "; strides, vector tails and edge padding passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
