#include <algorithm>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <random>
#include <vector>

#include "ulbvc/codec/tensor_data.h"
#include "ulbvc/core/status.h"
#include "ulbvc/detection/nv12_preprocess.h"

int main() {
  try {
    std::mt19937 random(7021);
    // Exercise vector blocks, scalar tails, padded strides and letterbox borders.
    for (int width : {2, 14, 16, 18, 32, 130, 1280}) {
      const int height = 32, stride = width + 16;
      std::vector<uint8_t> nv12(stride * height * 3 / 2);
      for (auto& pixel : nv12) pixel = random() & 255;
      cv::Mat y(height, width, CV_8UC1, nv12.data(), stride);
      cv::Mat uv(height / 2, width / 2, CV_8UC2, nv12.data() + stride * height, stride);
      cv::Mat rgb;
      cv::cvtColorTwoPlane(y, uv, rgb, cv::COLOR_YUV2RGB_NV12);
      const int input_width = width + 4, input_height = height + 8;
      const std::size_t plane = input_width * input_height;
      const uint16_t padding = ulbvc::codec::FloatToHalfBits(114.0f / 255);
      std::vector<uint16_t> result(plane * 3, padding);
      ulbvc::detection::PackNv12ToRgbFp16(y.data, y.step, uv.data, uv.step, width, height,
                                          input_width, input_height, 2, 4, result.data());
      for (int channel = 0; channel < 3; ++channel) {
        for (int row = 0; row < input_height; ++row) {
          for (int column = 0; column < input_width; ++column) {
            uint16_t expected = padding;
            if (row >= 4 && row < height + 4 && column >= 2 && column < width + 2)
              expected = ulbvc::codec::FloatToHalfBits(
                  rgb.at<cv::Vec3b>(row - 4, column - 2)[channel] / 255.0f);
            ulbvc::Check(result[channel * plane + row * input_width + column] == expected,
                         "fused NV12/FP16 preprocessing differs from OpenCV reference");
          }
        }
      }
    }
    std::cout << "detection preprocessing matches RGB reference, including padding and tails\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
