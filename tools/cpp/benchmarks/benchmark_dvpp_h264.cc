#include <mlvc/codec/tensor_data.h>
#include <mlvc/core/status.h>
#include <mlvc/io/dvpp_h264_encoder.h>
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
  std::filesystem::path output_h264;
  int width = 1920;
  int height = 1080;
  int padded_width = 1920;
  int padded_height = 1088;
  int device = 0;
  int frames = 100;
};

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    mlvc::Check(i + 1 < argc, "missing value for " + arg);
    const std::string value = argv[++i];
    if (arg == "--input-fp16") options.input_fp16 = value;
    else if (arg == "--output-h264") options.output_h264 = value;
    else if (arg == "--width") options.width = std::stoi(value);
    else if (arg == "--height") options.height = std::stoi(value);
    else if (arg == "--padded-width") options.padded_width = std::stoi(value);
    else if (arg == "--padded-height") options.padded_height = std::stoi(value);
    else if (arg == "--device") options.device = std::stoi(value);
    else if (arg == "--frames") options.frames = std::stoi(value);
    else throw mlvc::Error("unknown option: " + arg);
  }
  mlvc::Check(!options.input_fp16.empty(), "--input-fp16 is required");
  mlvc::Check(!options.output_h264.empty(), "--output-h264 is required");
  mlvc::Check(options.frames > 0, "--frames must be positive");
  return options;
}

bool HasAnnexBStartCode(const std::vector<uint8_t>& bytes) {
  return bytes.size() >= 4 && bytes[0] == 0 && bytes[1] == 0 && bytes[2] == 0 && bytes[3] == 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    const mlvc::io::Nv12Layout layout{options.width, options.height, options.padded_width,
                                      options.padded_height};
    const mlvc::codec::TensorData input = mlvc::codec::ReadTensorFile(
        options.input_fp16, {1, 3, options.padded_height, options.padded_width},
        mlvc::DataType::kFloat16);
    std::vector<uint8_t> nv12;
    mlvc::io::ConvertFp16Yuv444ToNv12(input, layout, &nv12);

    mlvc::AclRuntime runtime(options.device);
    mlvc::io::DvppH264EncoderConfig config;
    config.layout = layout;
    config.fps = 30;
    config.gop = 96;
    config.bitrate = 8'000'000;
    mlvc::io::DvppH264Encoder encoder(runtime.context(), config);

    void* nv12_device = nullptr;
    mlvc::CheckAcl(acldvppMalloc(&nv12_device, nv12.size()), "acldvppMalloc H.264 input");
    mlvc::CheckAcl(aclrtMemcpy(nv12_device, nv12.size(), nv12.data(), nv12.size(),
                               ACL_MEMCPY_HOST_TO_DEVICE),
                   "aclrtMemcpy H.264 input");
    aclrtEvent ready = nullptr;
    mlvc::CheckAcl(aclrtCreateEvent(&ready), "aclrtCreateEvent H.264 input");
    mlvc::CheckAcl(aclrtRecordEvent(ready, runtime.stream()), "aclrtRecordEvent H.264 input");
    std::vector<uint8_t> h264;
    const auto encode_start = std::chrono::steady_clock::now();
    for (int frame = 0; frame < options.frames; ++frame) {
      std::vector<uint8_t> frame_h264 =
          encoder.EncodeDevice(nv12_device, nv12.size(), ready, true);
      h264.insert(h264.end(), frame_h264.begin(), frame_h264.end());
    }
    const double encode_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - encode_start).count();
    mlvc::CheckAcl(aclrtDestroyEvent(ready), "aclrtDestroyEvent H.264 input");
    mlvc::CheckAcl(acldvppFree(nv12_device), "acldvppFree H.264 input");
    mlvc::Check(!h264.empty() && HasAnnexBStartCode(h264),
                "DVPP VENC output is not an Annex-B H.264 stream");

    std::ofstream output(options.output_h264, std::ios::binary);
    output.write(reinterpret_cast<const char*>(h264.data()),
                 static_cast<std::streamsize>(h264.size()));
    output.close();
    mlvc::Check(output.good(), "failed to write H.264 output");
    std::cout << "h264_valid=1\n";
    std::cout << "venc_frames=" << options.frames << "\n";
    std::cout << "venc_fps=" << options.frames / encode_seconds << "\n";
    std::cout << "h264_bytes=" << h264.size() << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "benchmark_dvpp_h264 failed: " << error.what() << "\n";
    return 1;
  }
}
