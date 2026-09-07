#include <mlvc/codec/tensor_data.h>
#include <mlvc/core/status.h>
#include <mlvc/io/dvpp_jpeg_encoder.h>
#include <mlvc/io/fp16_yuv444_to_nv12.h>
#include <mlvc/runtime/acl_runtime.h>

#include <acl/ops/acl_dvpp.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Options {
  std::filesystem::path input_fp16;
  std::filesystem::path output_jpeg;
  int width = 1920;
  int height = 1080;
  int padded_width = 1920;
  int padded_height = 1088;
  int device = 0;
  int quality = 75;
  int warmup = 5;
  int frames = 50;
};

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    mlvc::Check(i + 1 < argc, "missing value for " + arg);
    const std::string value = argv[++i];
    if (arg == "--input-fp16") options.input_fp16 = value;
    else if (arg == "--output-jpeg") options.output_jpeg = value;
    else if (arg == "--width") options.width = std::stoi(value);
    else if (arg == "--height") options.height = std::stoi(value);
    else if (arg == "--padded-width") options.padded_width = std::stoi(value);
    else if (arg == "--padded-height") options.padded_height = std::stoi(value);
    else if (arg == "--device") options.device = std::stoi(value);
    else if (arg == "--quality") options.quality = std::stoi(value);
    else if (arg == "--warmup") options.warmup = std::stoi(value);
    else if (arg == "--frames") options.frames = std::stoi(value);
    else throw mlvc::Error("unknown option: " + arg);
  }
  mlvc::Check(!options.input_fp16.empty(), "--input-fp16 is required");
  mlvc::Check(!options.output_jpeg.empty(), "--output-jpeg is required");
  mlvc::Check(options.quality >= 1 && options.quality <= 100, "quality must be in [1, 100]");
  mlvc::Check(options.warmup >= 0 && options.frames > 0, "invalid warmup or frame count");
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    const mlvc::io::Nv12Layout layout{options.width, options.height, options.width, options.height};
    const mlvc::codec::TensorData input = mlvc::codec::ReadTensorFile(
        options.input_fp16, {1, 3, options.padded_height, options.padded_width},
        mlvc::DataType::kFloat16);
    std::vector<uint8_t> nv12;
    mlvc::io::ConvertFp16Yuv444ToNv12(input, layout, &nv12);

    mlvc::AclRuntime runtime(options.device);
    mlvc::io::DvppJpegEncoder encoder(runtime.context(), layout,
                                      static_cast<uint32_t>(options.quality));
    void* nv12_device = nullptr;
    aclrtEvent nv12_ready = nullptr;
    mlvc::CheckAcl(acldvppMalloc(&nv12_device, nv12.size()),
                   "acldvppMalloc benchmark device input");
    mlvc::CheckAcl(aclrtMemcpy(nv12_device, nv12.size(), nv12.data(), nv12.size(),
                               ACL_MEMCPY_HOST_TO_DEVICE),
                   "copy benchmark NV12 H2D");
    mlvc::CheckAcl(aclrtCreateEvent(&nv12_ready), "aclrtCreateEvent benchmark NV12");
    mlvc::CheckAcl(aclrtRecordEvent(nv12_ready, runtime.stream()),
                   "aclrtRecordEvent benchmark NV12");
    mlvc::io::DvppJpegTiming device_timing;
    const std::vector<uint8_t> device_jpeg =
        encoder.EncodeDevice(nv12_device, nv12.size(), nv12_ready, &device_timing);
    mlvc::Check(device_timing.h2d_ms == 0.0,
                "device-input DVPP encode must not report H2D time");
    mlvc::Check(device_jpeg.size() >= 4 && device_jpeg[0] == 0xff &&
                    device_jpeg[1] == 0xd8 &&
                    device_jpeg[device_jpeg.size() - 2] == 0xff &&
                    device_jpeg.back() == 0xd9,
                "device-input DVPP output is not a valid JPEG");
    mlvc::CheckAcl(aclrtDestroyEvent(nv12_ready),
                   "aclrtDestroyEvent benchmark NV12");
    mlvc::CheckAcl(acldvppFree(nv12_device), "acldvppFree benchmark device input");
    for (int i = 0; i < options.warmup; ++i) encoder.Encode(nv12);

    double conversion_ms = 0.0;
    mlvc::io::DvppJpegTiming totals;
    std::vector<uint8_t> jpeg;
    const auto total_begin = std::chrono::steady_clock::now();
    for (int i = 0; i < options.frames; ++i) {
      const auto conversion_begin = std::chrono::steady_clock::now();
      mlvc::io::ConvertFp16Yuv444ToNv12(input, layout, &nv12);
      const auto conversion_end = std::chrono::steady_clock::now();
      conversion_ms +=
          std::chrono::duration<double, std::milli>(conversion_end - conversion_begin).count();
      mlvc::io::DvppJpegTiming timing;
      jpeg = encoder.Encode(nv12, &timing);
      totals.h2d_ms += timing.h2d_ms;
      totals.encode_ms += timing.encode_ms;
      totals.d2h_ms += timing.d2h_ms;
    }
    const auto total_end = std::chrono::steady_clock::now();
    mlvc::Check(jpeg.size() >= 4 && jpeg[0] == 0xff && jpeg[1] == 0xd8 &&
                    jpeg[jpeg.size() - 2] == 0xff && jpeg.back() == 0xd9,
                "DVPP output does not have JPEG SOI/EOI markers");
    std::ofstream output(options.output_jpeg, std::ios::binary);
    output.write(reinterpret_cast<const char*>(jpeg.data()),
                 static_cast<std::streamsize>(jpeg.size()));
    output.close();
    mlvc::Check(output.good(), "failed to write output JPEG");
    uint32_t decoded_width = 0;
    uint32_t decoded_height = 0;
    int32_t components = 0;
    mlvc::CheckAcl(acldvppJpegGetImageInfo(jpeg.data(), static_cast<uint32_t>(jpeg.size()),
                                           &decoded_width, &decoded_height, &components),
                   "acldvppJpegGetImageInfo");
    mlvc::Check(decoded_width == static_cast<uint32_t>(options.width) &&
                    decoded_height == static_cast<uint32_t>(options.height) && components == 3,
                "DVPP JPEG metadata mismatch");

    const double frames = static_cast<double>(options.frames);
    const double total_ms =
        std::chrono::duration<double, std::milli>(total_end - total_begin).count();
    std::cout << "jpeg_valid=1\n";
    std::cout << "decoded_size=" << decoded_width << "x" << decoded_height << "\n";
    std::cout << "jpeg_bytes=" << jpeg.size() << "\n";
    std::cout << "neon=" << (mlvc::io::Fp16Yuv444ToNv12UsesNeon() ? 1 : 0) << "\n";
    std::cout << "frames=" << options.frames << "\n";
    std::cout << "device_input_h2d_ms=" << device_timing.h2d_ms << "\n";
    std::cout << "device_input_dvpp_encode_ms=" << device_timing.encode_ms << "\n";
    std::cout << "device_input_d2h_ms=" << device_timing.d2h_ms << "\n";
    std::cout << "conversion_avg_ms=" << conversion_ms / frames << "\n";
    std::cout << "h2d_avg_ms=" << totals.h2d_ms / frames << "\n";
    std::cout << "dvpp_encode_avg_ms=" << totals.encode_ms / frames << "\n";
    std::cout << "d2h_avg_ms=" << totals.d2h_ms / frames << "\n";
    std::cout << "total_avg_ms=" << total_ms / frames << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "benchmark_dvpp_jpeg failed: " << error.what() << "\n";
    return 1;
  }
}
