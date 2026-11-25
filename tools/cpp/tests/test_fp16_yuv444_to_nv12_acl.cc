#include <acl/acl.h>
#include <ulbvc/codec/tensor_data.h>
#include <ulbvc/core/buffer.h>
#include <ulbvc/core/status.h>
#include <ulbvc/io/fp16_yuv444_to_nv12.h>
#include <ulbvc/runtime/fp16_yuv444_to_nv12_acl.h>
#include <ulbvc/runtime/stage_runtime.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>

int main() {
  try {
    constexpr int kWidth = 16;
    constexpr int kHeight = 2;
    ulbvc::codec::TensorData input =
        ulbvc::codec::MakeTensor({1, 3, kHeight, kWidth}, ulbvc::DataType::kFloat16);
    auto* values = reinterpret_cast<uint16_t*>(input.bytes.data());
    for (std::size_t i = 0; i < input.Elements(); ++i) {
      const float value = static_cast<float>(static_cast<int>(i % 24U) - 3) / 16.0F;
      values[i] = ulbvc::codec::FloatToHalfBits(value);
    }

    const ulbvc::io::Nv12Layout layout{kWidth, kHeight, 32, kHeight};
    std::vector<uint8_t> expected;
    ulbvc::io::ConvertFp16Yuv444ToNv12Scalar(input, layout, &expected);

    ulbvc::StageRuntime runtime(0);
    ulbvc::AclBuffer input_device(input.bytes.size());
    ulbvc::AclBuffer output_device(expected.size());
    ulbvc::CheckAcl(aclrtMemcpy(input_device.data(), input_device.bytes(), input.bytes.data(),
                               input.bytes.size(), ACL_MEMCPY_HOST_TO_DEVICE),
                   "copy conversion test input H2D");
    ulbvc::Check(ulbvc::Fp16Yuv444ToNv12AclAvailable(),
                "FP16 YUV444-to-NV12 ACL operator is unavailable");
    ulbvc::Fp16Yuv444ToNv12Acl(input_device.data(), input.shape, layout, output_device.data(),
                              runtime.stream());
    ulbvc::CheckAcl(aclrtSynchronizeStream(runtime.stream()), "synchronize conversion test stream");

    std::vector<uint8_t> actual(expected.size());
    ulbvc::CheckAcl(aclrtMemcpy(actual.data(), actual.size(), output_device.data(),
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
        ulbvc::Check(false, message.str());
      }
    }

    constexpr int kFullWidth = 1920;
    constexpr int kFullVisibleHeight = 1080;
    constexpr int kFullInputHeight = 1088;
    ulbvc::codec::TensorData full_input =
        ulbvc::codec::MakeTensor({1, 3, kFullInputHeight, kFullWidth}, ulbvc::DataType::kFloat16);
    auto* full_values = reinterpret_cast<uint16_t*>(full_input.bytes.data());
    for (std::size_t i = 0; i < full_input.Elements(); ++i) {
      const float value = static_cast<float>(static_cast<int>(i % 1536U) - 256) / 1024.0F;
      full_values[i] = ulbvc::codec::FloatToHalfBits(value);
    }
    const ulbvc::io::Nv12Layout full_layout{kFullWidth, kFullVisibleHeight, kFullWidth,
                                           kFullVisibleHeight};
    std::vector<uint8_t> full_expected;
    ulbvc::io::ConvertFp16Yuv444ToNv12Scalar(full_input, full_layout, &full_expected);
    ulbvc::AclBuffer full_input_device(full_input.bytes.size());
    ulbvc::AclBuffer full_output_device(full_expected.size());
    ulbvc::CheckAcl(
        aclrtMemcpy(full_input_device.data(), full_input_device.bytes(), full_input.bytes.data(),
                    full_input.bytes.size(), ACL_MEMCPY_HOST_TO_DEVICE),
        "copy full conversion test input H2D");
    ulbvc::Fp16Yuv444ToNv12Acl(full_input_device.data(), full_input.shape, full_layout,
                              full_output_device.data(), runtime.stream());
    ulbvc::CheckAcl(aclrtSynchronizeStream(runtime.stream()),
                   "synchronize full conversion test stream");
    std::vector<uint8_t> full_actual(full_expected.size());
    ulbvc::CheckAcl(aclrtMemcpy(full_actual.data(), full_actual.size(), full_output_device.data(),
                               full_output_device.bytes(), ACL_MEMCPY_DEVICE_TO_HOST),
                   "copy full conversion test output D2H");
    ulbvc::Check(full_actual == full_expected,
                "1080p device FP16 YUV444-to-NV12 output differs from scalar reference");

    constexpr int kTimingIterations = 5;
    const auto timing_begin = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < kTimingIterations; ++iteration) {
      ulbvc::Fp16Yuv444ToNv12Acl(full_input_device.data(), full_input.shape, full_layout,
                                full_output_device.data(), runtime.stream());
      ulbvc::CheckAcl(aclrtSynchronizeStream(runtime.stream()), "synchronize timed full conversion");
    }
    const auto timing_end = std::chrono::steady_clock::now();
    const double average_ms =
        std::chrono::duration<double, std::milli>(timing_end - timing_begin).count() /
        kTimingIterations;

    aclrtEvent conversion_start = nullptr;
    aclrtEvent conversion_ready = nullptr;
    aclrtStream wait_stream = nullptr;
    constexpr uint32_t kTimedSyncEvent = ACL_EVENT_SYNC | ACL_EVENT_TIME_LINE;
    ulbvc::CheckAcl(aclrtCreateEventWithFlag(&conversion_start, kTimedSyncEvent),
                   "create conversion start timeline event");
    ulbvc::CheckAcl(aclrtCreateEventWithFlag(&conversion_ready, kTimedSyncEvent),
                   "create conversion ready timeline event");
    ulbvc::CheckAcl(aclrtCreateStream(&wait_stream), "create conversion wait stream");
    ulbvc::Fp16Yuv444ToNv12Acl(full_input_device.data(), full_input.shape, full_layout,
                              full_output_device.data(), runtime.stream(), conversion_start,
                              conversion_ready);
    ulbvc::CheckAcl(aclrtSynchronizeEvent(conversion_ready),
                   "synchronize conversion ready timeline event");
    float event_elapsed_ms = 0.0F;
    ulbvc::CheckAcl(aclrtEventElapsedTime(&event_elapsed_ms, conversion_start, conversion_ready),
                   "read conversion timeline before stream wait");
    ulbvc::CheckAcl(aclrtStreamWaitEvent(wait_stream, conversion_ready),
                   "wait for conversion ready timeline event");
    ulbvc::CheckAcl(aclrtSynchronizeStream(wait_stream), "synchronize conversion wait stream");
    ulbvc::CheckAcl(aclrtDestroyStream(wait_stream), "destroy conversion wait stream");
    ulbvc::CheckAcl(aclrtDestroyEvent(conversion_ready), "destroy conversion ready timeline event");
    ulbvc::CheckAcl(aclrtDestroyEvent(conversion_start), "destroy conversion start timeline event");
    std::cout << "device_fp16_yuv444_to_nv12_1080p_avg_ms=" << average_ms << '\n';
    std::cout << "device_fp16_yuv444_to_nv12_event_ms=" << event_elapsed_ms << '\n';
    std::cout << "device FP16 YUV444-to-NV12 test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_fp16_yuv444_to_nv12_acl failed: " << error.what() << "\n";
    return 1;
  }
}
