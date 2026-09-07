#include <mlvc/codec/tensor_data.h>
#include <mlvc/io/fp16_yuv444_to_nv12.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using mlvc::codec::TensorData;
using mlvc::io::Nv12Layout;

TensorData MakeFrame(int padded_width, int padded_height, float y, float u, float v) {
  TensorData frame = mlvc::codec::MakeFp16Tensor(
      {1, 3, padded_height, padded_width}, 0.0F);
  auto* data = reinterpret_cast<uint16_t*>(frame.bytes.data());
  const std::size_t plane = static_cast<std::size_t>(padded_width * padded_height);
  const uint16_t y_half = static_cast<uint16_t>(mlvc::codec::FloatToHalfBits(y));
  const uint16_t u_half = static_cast<uint16_t>(mlvc::codec::FloatToHalfBits(u));
  const uint16_t v_half = static_cast<uint16_t>(mlvc::codec::FloatToHalfBits(v));
  std::fill(data, data + plane, y_half);
  std::fill(data + plane, data + 2 * plane, u_half);
  std::fill(data + 2 * plane, data + 3 * plane, v_half);
  return frame;
}

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void ExpectBytes(const std::vector<uint8_t>& actual,
                 const std::vector<uint8_t>& expected,
                 const std::string& label) {
  Expect(actual == expected, label + " byte mismatch");
}

void TestConstantBlackAndWhite() {
  const Nv12Layout layout{4, 2, 4, 2};
  std::vector<uint8_t> output;
  mlvc::io::ConvertFp16Yuv444ToNv12Scalar(MakeFrame(4, 2, 0.0F, 0.5F, 0.5F),
                                           layout, &output);
  ExpectBytes(output, {0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128}, "black");

  mlvc::io::ConvertFp16Yuv444ToNv12Scalar(MakeFrame(4, 2, 1.0F, 0.5F, 0.5F),
                                           layout, &output);
  ExpectBytes(output, {255, 255, 255, 255, 255, 255, 255, 255, 128, 128, 128, 128},
              "white");
}

void TestChromaAveragesTwoByTwo() {
  TensorData frame = MakeFrame(2, 2, 0.25F, 0.0F, 0.0F);
  auto* data = reinterpret_cast<uint16_t*>(frame.bytes.data());
  const std::size_t plane = 4;
  const float samples[4] = {0.0F, 0.5F, 0.5F, 1.0F};
  for (std::size_t i = 0; i < 4; ++i) {
    data[plane + i] = static_cast<uint16_t>(mlvc::codec::FloatToHalfBits(samples[i]));
    data[2 * plane + i] =
        static_cast<uint16_t>(mlvc::codec::FloatToHalfBits(1.0F - samples[i]));
  }
  std::vector<uint8_t> output;
  mlvc::io::ConvertFp16Yuv444ToNv12Scalar(frame, {2, 2, 2, 2}, &output);
  ExpectBytes(output, {64, 64, 64, 64, 128, 128}, "chroma average");
}

void TestVisibleCropAndStridePadding() {
  TensorData frame = MakeFrame(6, 4, 0.0F, 0.5F, 0.5F);
  auto* data = reinterpret_cast<uint16_t*>(frame.bytes.data());
  for (int y = 0; y < 2; ++y) {
    for (int x = 0; x < 4; ++x) {
      data[y * 6 + x] = static_cast<uint16_t>(mlvc::codec::FloatToHalfBits(1.0F));
    }
  }
  std::vector<uint8_t> output;
  mlvc::io::ConvertFp16Yuv444ToNv12Scalar(frame, {4, 2, 8, 4}, &output);
  Expect(output.size() == 48, "strided NV12 size");
  for (int y = 0; y < 2; ++y) {
    for (int x = 0; x < 4; ++x) Expect(output[y * 8 + x] == 255, "visible Y");
    for (int x = 4; x < 8; ++x) Expect(output[y * 8 + x] == 0, "Y row padding");
  }
  for (int y = 2; y < 4; ++y) {
    for (int x = 0; x < 8; ++x) Expect(output[y * 8 + x] == 0, "Y height padding");
  }
  const std::size_t uv_offset = 32;
  for (int x = 0; x < 4; ++x) Expect(output[uv_offset + x] == 128, "visible UV");
  for (int x = 4; x < 8; ++x) Expect(output[uv_offset + x] == 0, "UV row padding");
  for (std::size_t i = uv_offset + 8; i < output.size(); ++i) {
    Expect(output[i] == 0, "UV height padding");
  }
}

void TestRejectsOddVisibleDimensions() {
  bool rejected = false;
  try {
    std::vector<uint8_t> output;
    mlvc::io::ConvertFp16Yuv444ToNv12Scalar(MakeFrame(4, 2, 0.0F, 0.5F, 0.5F),
                                             {3, 2, 4, 2}, &output);
  } catch (const std::exception&) {
    rejected = true;
  }
  Expect(rejected, "odd visible width must be rejected");
}

void TestOptimizedMatchesScalar() {
#if defined(__aarch64__)
  Expect(mlvc::io::Fp16Yuv444ToNv12UsesNeon(), "AArch64 build must use the NEON converter");
#endif
  TensorData frame = MakeFrame(32, 20, 0.0F, 0.0F, 0.0F);
  auto* data = reinterpret_cast<uint16_t*>(frame.bytes.data());
  std::mt19937 random(12345);
  std::uniform_real_distribution<float> values(-0.25F, 1.25F);
  for (std::size_t i = 0; i < frame.Elements(); ++i) {
    data[i] = static_cast<uint16_t>(mlvc::codec::FloatToHalfBits(values(random)));
  }
  const Nv12Layout layout{32, 18, 48, 20};
  std::vector<uint8_t> scalar;
  std::vector<uint8_t> optimized;
  mlvc::io::ConvertFp16Yuv444ToNv12Scalar(frame, layout, &scalar);
  mlvc::io::ConvertFp16Yuv444ToNv12(frame, layout, &optimized);
  ExpectBytes(optimized, scalar, "optimized versus scalar");
}

}  // namespace

int main() {
  try {
    TestConstantBlackAndWhite();
    TestChromaAveragesTwoByTwo();
    TestVisibleCropAndStridePadding();
    TestRejectsOddVisibleDimensions();
    TestOptimizedMatchesScalar();
    std::cout << "fp16_yuv444_to_nv12 tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test_fp16_yuv444_to_nv12 failed: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
