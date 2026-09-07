#include <mlvc/codec/tensor_data.h>
#include <mlvc/core/buffer.h>
#include <mlvc/core/status.h>
#include <mlvc/io/fp16_yuv444_to_nv12.h>
#include <mlvc/runtime/fp16_yuv444_to_nv12_acl.h>
#include <mlvc/runtime/stage_runtime.h>

#include <acl/acl.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>

int main() {
  try {
    constexpr int kWidth = 16;
    constexpr int kHeight = 2;
    mlvc::codec::TensorData input =
        mlvc::codec::MakeTensor({1, 3, kHeight, kWidth}, mlvc::DataType::kFloat16);
    auto* values = reinterpret_cast<uint16_t*>(input.bytes.data());
    for (std::size_t i = 0; i < input.Elements(); ++i) {
      const float value =
          static_cast<float>(static_cast<int>(i % 24U) - 3) / 16.0F;
      values[i] = mlvc::codec::FloatToHalfBits(value);
    }

    const mlvc::io::Nv12Layout layout{kWidth, kHeight, 32, kHeight};
    std::vector<uint8_t> expected;
    mlvc::io::ConvertFp16Yuv444ToNv12Scalar(input, layout, &expected);

    mlvc::StageRuntime runtime(0);
    mlvc::AclBuffer input_device(input.bytes.size());
    mlvc::AclBuffer output_device(expected.size());
    mlvc::CheckAcl(aclrtMemcpy(input_device.data(), input_device.bytes(), input.bytes.data(),
                               input.bytes.size(), ACL_MEMCPY_HOST_TO_DEVICE),
                   "copy conversion test input H2D");
    mlvc::Check(mlvc::Fp16Yuv444ToNv12AclAvailable(),
                "FP16 YUV444-to-NV12 ACL operator is unavailable");
    mlvc::Fp16Yuv444ToNv12Acl(input_device.data(), input.shape, layout,
                              output_device.data(), runtime.stream());
    mlvc::CheckAcl(aclrtSynchronizeStream(runtime.stream()),
                   "synchronize conversion test stream");

    std::vector<uint8_t> actual(expected.size());
    mlvc::CheckAcl(aclrtMemcpy(actual.data(), actual.size(), output_device.data(),
                               output_device.bytes(), ACL_MEMCPY_DEVICE_TO_HOST),
                   "copy conversion test output D2H");
    for (std::size_t i = 0; i < actual.size(); ++i) {
      if (actual[i] != expected[i]) {
        std::ostringstream message;
        message << "device FP16 YUV444-to-NV12 mismatch at byte " << i
                << ": expected=" << static_cast<int>(expected[i])
                << ", actual=" << static_cast<int>(actual[i]) << "; expected bytes=";
        for (uint8_t value : expected) message << ' ' << static_cast<int>(value);
        message << "; actual bytes=";
        for (uint8_t value : actual) message << ' ' << static_cast<int>(value);
        mlvc::Check(false, message.str());
      }
    }

    constexpr int kFullWidth = 1920;
    constexpr int kFullVisibleHeight = 1080;
    constexpr int kFullInputHeight = 1088;
    mlvc::codec::TensorData full_input = mlvc::codec::MakeTensor(
        {1, 3, kFullInputHeight, kFullWidth}, mlvc::DataType::kFloat16);
    auto* full_values = reinterpret_cast<uint16_t*>(full_input.bytes.data());
    for (std::size_t i = 0; i < full_input.Elements(); ++i) {
      const float value =
          static_cast<float>(static_cast<int>(i % 1536U) - 256) / 1024.0F;
      full_values[i] = mlvc::codec::FloatToHalfBits(value);
    }
    const mlvc::io::Nv12Layout full_layout{
        kFullWidth, kFullVisibleHeight, kFullWidth, kFullVisibleHeight};
    std::vector<uint8_t> full_expected;
    mlvc::io::ConvertFp16Yuv444ToNv12Scalar(full_input, full_layout,
                                             &full_expected);
    mlvc::AclBuffer full_input_device(full_input.bytes.size());
    mlvc::AclBuffer full_output_device(full_expected.size());
    mlvc::CheckAcl(aclrtMemcpy(full_input_device.data(), full_input_device.bytes(),
                               full_input.bytes.data(), full_input.bytes.size(),
                               ACL_MEMCPY_HOST_TO_DEVICE),
                   "copy full conversion test input H2D");
    mlvc::Fp16Yuv444ToNv12Acl(full_input_device.data(), full_input.shape,
                              full_layout, full_output_device.data(),
                              runtime.stream());
    mlvc::CheckAcl(aclrtSynchronizeStream(runtime.stream()),
                   "synchronize full conversion test stream");
    std::vector<uint8_t> full_actual(full_expected.size());
    mlvc::CheckAcl(aclrtMemcpy(full_actual.data(), full_actual.size(),
                               full_output_device.data(), full_output_device.bytes(),
                               ACL_MEMCPY_DEVICE_TO_HOST),
                   "copy full conversion test output D2H");
    mlvc::Check(full_actual == full_expected,
                "1080p device FP16 YUV444-to-NV12 output differs from scalar reference");

    constexpr int kTimingIterations = 5;
    const auto timing_begin = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < kTimingIterations; ++iteration) {
      mlvc::Fp16Yuv444ToNv12Acl(full_input_device.data(), full_input.shape,
                                full_layout, full_output_device.data(),
                                runtime.stream());
      mlvc::CheckAcl(aclrtSynchronizeStream(runtime.stream()),
                     "synchronize timed full conversion");
    }
    const auto timing_end = std::chrono::steady_clock::now();
    const double average_ms =
        std::chrono::duration<double, std::milli>(timing_end - timing_begin).count() /
        kTimingIterations;

    aclrtEvent conversion_start = nullptr;
    aclrtEvent conversion_ready = nullptr;
    aclrtStream wait_stream = nullptr;
    constexpr uint32_t kTimedSyncEvent = ACL_EVENT_SYNC | ACL_EVENT_TIME_LINE;
    mlvc::CheckAcl(aclrtCreateEventWithFlag(&conversion_start, kTimedSyncEvent),
                   "create conversion start timeline event");
    mlvc::CheckAcl(aclrtCreateEventWithFlag(&conversion_ready, kTimedSyncEvent),
                   "create conversion ready timeline event");
    mlvc::CheckAcl(aclrtCreateStream(&wait_stream), "create conversion wait stream");
    mlvc::Fp16Yuv444ToNv12Acl(
        full_input_device.data(), full_input.shape, full_layout,
        full_output_device.data(), runtime.stream(), conversion_start,
        conversion_ready);
    mlvc::CheckAcl(aclrtSynchronizeEvent(conversion_ready),
                   "synchronize conversion ready timeline event");
    float event_elapsed_ms = 0.0F;
    mlvc::CheckAcl(aclrtEventElapsedTime(&event_elapsed_ms, conversion_start,
                                        conversion_ready),
                   "read conversion timeline before stream wait");
    mlvc::CheckAcl(aclrtStreamWaitEvent(wait_stream, conversion_ready),
                   "wait for conversion ready timeline event");
    mlvc::CheckAcl(aclrtSynchronizeStream(wait_stream),
                   "synchronize conversion wait stream");
    mlvc::CheckAcl(aclrtDestroyStream(wait_stream), "destroy conversion wait stream");
    mlvc::CheckAcl(aclrtDestroyEvent(conversion_ready),
                   "destroy conversion ready timeline event");
    mlvc::CheckAcl(aclrtDestroyEvent(conversion_start),
                   "destroy conversion start timeline event");
    std::cout << "device_fp16_yuv444_to_nv12_1080p_avg_ms=" << average_ms << '\n';
    std::cout << "device_fp16_yuv444_to_nv12_event_ms=" << event_elapsed_ms << '\n';
    std::cout << "device FP16 YUV444-to-NV12 test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_fp16_yuv444_to_nv12_acl failed: " << error.what() << "\n";
    return 1;
  }
}
