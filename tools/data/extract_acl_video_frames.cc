#include <mlvc/codec/tensor_data.h>
#include <mlvc/core/status.h>
#include <mlvc/io/video_io.h>
#include <mlvc/runtime/model_manifest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

namespace {

constexpr std::string_view kDefaultManifest = "../mlvc1080p/manifest.json";

const mlvc::TensorSpec& FindTensorSpec(const mlvc::ModelRecord& record, std::string_view name) {
  for (const mlvc::TensorSpec& spec : record.inputs) {
    if (spec.name == name) {
      return spec;
    }
  }
  for (const mlvc::TensorSpec& spec : record.outputs) {
    if (spec.name == name) {
      return spec;
    }
  }
  throw mlvc::Error("missing tensor spec: " + record.name + "." + std::string(name));
}

void PrintUsage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " --video <input.mp4> --output-frame-dir <dir>"
               " [--manifest <manifest.json>] [--limit -1]\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path manifest_path = std::filesystem::path(kDefaultManifest);
  std::filesystem::path input_video_path;
  std::filesystem::path output_frame_dir;
  int limit = -1;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--manifest" && i + 1 < argc) {
      manifest_path = argv[++i];
    } else if (arg == "--video" && i + 1 < argc) {
      input_video_path = argv[++i];
    } else if (arg == "--output-frame-dir" && i + 1 < argc) {
      output_frame_dir = argv[++i];
    } else if (arg == "--limit" && i + 1 < argc) {
      limit = std::stoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    } else {
      PrintUsage(argv[0]);
      return 2;
    }
  }

  if (input_video_path.empty() || output_frame_dir.empty()) {
    PrintUsage(argv[0]);
    return 2;
  }

  try {
    mlvc::ModelManifest manifest = mlvc::ModelManifest::Load(manifest_path);
    const mlvc::TensorSpec& frame_spec = FindTensorSpec(manifest.GetModel("MLVCEncoder"), "x");
    mlvc::io::VideoFrameReader video_reader(input_video_path);
    const mlvc::io::VideoInfo& video_info = video_reader.info();
    mlvc::Check(video_info.width <= static_cast<int>(frame_spec.shape[3]) &&
                    video_info.height <= static_cast<int>(frame_spec.shape[2]),
                "input video exceeds model frame tensor shape");
    std::filesystem::create_directories(output_frame_dir);

    mlvc::codec::TensorData frame = mlvc::codec::MakeTensorLike(frame_spec);
    int frame_index = 0;
    const auto begin = std::chrono::steady_clock::now();
    while (limit < 0 || frame_index < limit) {
      if (!video_reader.ReadFrameAsTensor(frame_spec, &frame)) {
        break;
      }
      const std::filesystem::path frame_path =
          output_frame_dir / ("frame_" + std::to_string(frame_index) + ".fp16");
      mlvc::codec::WriteTensorFile(frame_path, frame);
      ++frame_index;
    }
    const auto end = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    const double fps =
        frame_index > 0 ? (1000.0 * static_cast<double>(frame_index) / total_ms) : 0.0;
    {
      std::ofstream info(output_frame_dir / "source_info.txt");
      mlvc::Check(info.good(), "failed to open source info output");
      info << "width=" << video_info.width << "\n";
      info << "height=" << video_info.height << "\n";
      info << "fps=" << video_info.fps << "\n";
    }
    std::cout << "frames=" << frame_index << "\n";
    std::cout << "total_ms=" << total_ms << "\n";
    std::cout << "fps=" << fps << "\n";
    std::cout << "output_frame_dir=" << output_frame_dir.string() << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "extract_acl_video_frames failed: " << error.what() << "\n";
    return 1;
  }
}
