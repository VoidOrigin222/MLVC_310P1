#include <mlvc/application/input/frame_input_queue.h>
#include <mlvc/codec/mlvc_entropy.h>
#include <mlvc/codec/mlvc_rate_control.h>
#include <mlvc/codec/tensor_utils.h>
#include <mlvc/core/status.h>
#include <mlvc/entropy/entropy_codec.h>
#include <mlvc/entropy/mlvc_official_entropy.h>
#include <mlvc/entropy/sidecar.h>
#include <mlvc/framework/entropy_worker.h>
#include <mlvc/io/mlvc_bitstream.h>
#include <mlvc/io/video_io.h>
#include <mlvc/runtime/stage_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

namespace {

constexpr std::string_view kDefaultManifest = "../mlvc1080p/manifest.json";
constexpr int kDefaultQIndex = 21;
constexpr int kDefaultGop = 32;
constexpr int kDefaultResetInterval = 16;
constexpr int kDefaultLtrStartIdx = 8;
constexpr int kDefaultLtrPeriod = 64;
constexpr int kDefaultLtrQpShift = 8;
constexpr std::array<int, 8> kMlvcQIndexShift = {8, 0, 4, 0, 4, 0, 4, 0};

void PrintUsage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " (--video <input.mp4> | --input-frame-dir <dir>)"
               " [--manifest <manifest.json>] [--device 0] [--fps 30]"
               " [--warmup 16] [--frames -1] [--q-index 21]"
               " [--gop 32] [--reset-interval 16]"
               " [--ltr-start-idx 8] [--ltr-period 64] [--ltr-qp-shift 8]"
               " [--target-bitrate-bps 0]"
               " [--entropy-pipeline]"
               " --output-bitstream <stream.bin>\n";
}

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

mlvc::codec::TensorData MakeInt32Tensor(const mlvc::TensorSpec& spec, int32_t value) {
  mlvc::codec::TensorData tensor = mlvc::codec::MakeTensorLike(spec);
  mlvc::Check(spec.dtype == mlvc::DataType::kInt32, "q-index tensor must be int32");
  mlvc::Check(spec.shape.size() == 1 && spec.shape[0] == 1, "q-index tensor must be shape [1]");
  std::memcpy(tensor.bytes.data(), &value, sizeof(value));
  return tensor;
}

void SetInt32TensorValue(mlvc::codec::TensorData* tensor, int32_t value) {
  std::memcpy(tensor->bytes.data(), &value, sizeof(value));
}

void ResetTensor(mlvc::codec::TensorData* tensor) {
  std::fill(tensor->bytes.begin(), tensor->bytes.end(), 0);
}

void DumpFeatureStats(const mlvc::codec::TensorData& tensor,
                      const std::filesystem::path& output_path) {
  mlvc::Check(tensor.dtype == mlvc::DataType::kFloat16,
              "feature dump requires an FP16 tensor");
  mlvc::codec::WriteTensorFile(output_path, tensor);

  const auto* bits = reinterpret_cast<const uint16_t*>(tensor.bytes.data());
  const std::size_t elements = tensor.bytes.size() / sizeof(uint16_t);
  float min_value = std::numeric_limits<float>::infinity();
  float max_value = -std::numeric_limits<float>::infinity();
  float max_abs = 0.0F;
  double sum = 0.0;
  std::size_t finite_count = 0;
  std::size_t nan_count = 0;
  std::size_t inf_count = 0;
  std::size_t zero_count = 0;
  for (std::size_t i = 0; i < elements; ++i) {
    const float value = mlvc::codec::HalfBitsToFloat(bits[i]);
    if (std::isnan(value)) {
      ++nan_count;
      continue;
    }
    if (std::isinf(value)) {
      ++inf_count;
      continue;
    }
    min_value = std::min(min_value, value);
    max_value = std::max(max_value, value);
    max_abs = std::max(max_abs, std::abs(value));
    sum += static_cast<double>(value);
    ++finite_count;
    if (value == 0.0F) {
      ++zero_count;
    }
  }

  std::cout << std::setprecision(17);
  std::cout << "feature_debug_path=" << output_path.string() << "\n";
  std::cout << "feature_elements=" << elements << "\n";
  std::cout << "feature_bytes=" << tensor.bytes.size() << "\n";
  std::cout << "feature_min=" << min_value << "\n";
  std::cout << "feature_max=" << max_value << "\n";
  std::cout << "feature_mean="
            << (finite_count > 0 ? sum / static_cast<double>(finite_count) : 0.0) << "\n";
  std::cout << "feature_max_abs=" << max_abs << "\n";
  std::cout << "feature_nan=" << nan_count << "\n";
  std::cout << "feature_inf=" << inf_count << "\n";
  std::cout << "feature_zero=" << zero_count << "\n";
  std::cout.flush();
}

struct SourceInfo {
  int width = 0;
  int height = 0;
  double fps = 30.0;
};

std::optional<SourceInfo> ReadSourceInfo(const std::filesystem::path& dir) {
  const std::filesystem::path path = dir / "source_info.txt";
  if (!std::filesystem::exists(path)) {
    return std::nullopt;
  }
  std::ifstream input(path);
  if (!input.good()) {
    return std::nullopt;
  }
  SourceInfo info;
  std::string line;
  while (std::getline(input, line)) {
    const std::size_t eq = line.find('=');
    if (eq == std::string::npos) {
      continue;
    }
    const std::string key = line.substr(0, eq);
    const std::string value = line.substr(eq + 1);
    if (key == "width") {
      info.width = std::stoi(value);
    } else if (key == "height") {
      info.height = std::stoi(value);
    } else if (key == "fps") {
      info.fps = std::stod(value);
    }
  }
  if (info.width <= 0 || info.height <= 0) {
    return std::nullopt;
  }
  return info;
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path manifest_path = std::filesystem::path(kDefaultManifest);
  std::filesystem::path input_path;
  std::filesystem::path input_frame_dir;
  std::filesystem::path output_bitstream_path;
  int device = 0;
  double fallback_fps = 30.0;
  int warmup = 16;
  int frames = -1;
  int q_index = kDefaultQIndex;
  int gop = kDefaultGop;
  int reset_interval = kDefaultResetInterval;
  int ltr_start_idx = kDefaultLtrStartIdx;
  int ltr_period = kDefaultLtrPeriod;
  int ltr_qp_shift = kDefaultLtrQpShift;
  double target_bitrate_bps = 0.0;
  bool entropy_pipeline = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--manifest" && i + 1 < argc) {
      manifest_path = argv[++i];
    } else if (arg == "--video" && i + 1 < argc) {
      input_path = argv[++i];
    } else if (arg == "--input-frame-dir" && i + 1 < argc) {
      input_frame_dir = argv[++i];
    } else if (arg == "--output-bitstream" && i + 1 < argc) {
      output_bitstream_path = argv[++i];
    } else if (arg == "--device" && i + 1 < argc) {
      device = std::stoi(argv[++i]);
    } else if (arg == "--fps" && i + 1 < argc) {
      fallback_fps = std::stod(argv[++i]);
    } else if (arg == "--warmup" && i + 1 < argc) {
      warmup = std::stoi(argv[++i]);
    } else if (arg == "--frames" && i + 1 < argc) {
      frames = std::stoi(argv[++i]);
    } else if (arg == "--q-index" && i + 1 < argc) {
      q_index = std::stoi(argv[++i]);
    } else if (arg == "--gop" && i + 1 < argc) {
      gop = std::stoi(argv[++i]);
    } else if (arg == "--reset-interval" && i + 1 < argc) {
      reset_interval = std::stoi(argv[++i]);
    } else if (arg == "--ltr-start-idx" && i + 1 < argc) {
      ltr_start_idx = std::stoi(argv[++i]);
    } else if (arg == "--ltr-period" && i + 1 < argc) {
      ltr_period = std::stoi(argv[++i]);
    } else if (arg == "--ltr-qp-shift" && i + 1 < argc) {
      ltr_qp_shift = std::stoi(argv[++i]);
    } else if (arg == "--target-bitrate-bps" && i + 1 < argc) {
      target_bitrate_bps = std::stod(argv[++i]);
    } else if (arg == "--entropy-pipeline") {
      entropy_pipeline = true;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    } else {
      PrintUsage(argv[0]);
      return 2;
    }
  }

  if ((input_path.empty() && input_frame_dir.empty()) || output_bitstream_path.empty() ||
      warmup < 0 || frames == 0 || gop < 0 || reset_interval < 0) {
    PrintUsage(argv[0]);
    return 2;
  }
  mlvc::Check(ltr_start_idx >= 0, "ltr_start_idx must be non-negative");
  mlvc::Check(ltr_period >= 0, "ltr_period must be non-negative");
  mlvc::Check(input_path.empty() || input_frame_dir.empty(),
              "--video and --input-frame-dir are mutually exclusive");
  mlvc::Check(!entropy_pipeline || target_bitrate_bps <= 0.0,
              "--entropy-pipeline currently requires --target-bitrate-bps 0");

  try {
    mlvc::StageRuntime runtime(device);
    mlvc::ModelManifest manifest = mlvc::ModelManifest::Load(manifest_path);
    const std::filesystem::path model_directory = manifest.directory();
    mlvc::Check(manifest.HasModel("MLVCEncoder") && manifest.HasModel("MLVCDecoder"),
                "e1d1 manifest must contain MLVCEncoder and MLVCDecoder");
    mlvc::StageModelSet models(&runtime, std::move(manifest));
    mlvc::StageModel& encoder = models.GetStage(std::string("MLVCEncoder"));

    const mlvc::TensorSpec& frame_spec = FindTensorSpec(encoder.record(), "x");
    const mlvc::TensorSpec& ref_feature_spec = FindTensorSpec(encoder.record(), "ref_feature");
    const mlvc::TensorSpec& q_index_spec = FindTensorSpec(encoder.record(), "q_index_shifted");
    const mlvc::TensorSpec& encoder_feature_spec = FindTensorSpec(encoder.record(), "feature");
    const mlvc::TensorSpec& z_raw_spec = FindTensorSpec(encoder.record(), "z_raw");
    const mlvc::TensorSpec& y_raw_0_spec = FindTensorSpec(encoder.record(), "y_raw_0");
    const mlvc::TensorSpec& y_raw_1_spec = FindTensorSpec(encoder.record(), "y_raw_1");

    mlvc::Check(encoder.record().inputs.size() == 3, "MLVCEncoder input count mismatch");
    mlvc::Check(encoder.record().outputs.size() == 4, "MLVCEncoder output count mismatch");

    std::optional<mlvc::io::VideoFrameReader> video_reader;
    std::optional<mlvc::io::VideoInfo> video_info;
    int source_width = static_cast<int>(frame_spec.shape[3]);
    int source_height = static_cast<int>(frame_spec.shape[2]);
    double source_fps = fallback_fps;
    if (!input_path.empty()) {
      video_reader.emplace(input_path);
      video_info = video_reader->info();
      mlvc::Check(video_info->width <= static_cast<int>(frame_spec.shape[3]) &&
                      video_info->height <= static_cast<int>(frame_spec.shape[2]),
                  "input video exceeds model frame tensor shape");
      source_width = video_info->width;
      source_height = video_info->height;
      source_fps = video_info->fps;
    } else {
      const std::optional<SourceInfo> source_info = ReadSourceInfo(input_frame_dir);
      if (source_info.has_value()) {
        source_width = source_info->width;
        source_height = source_info->height;
        source_fps = source_info->fps;
      }
    }

    int measured_frames = frames;
    if (video_info.has_value() && video_info->frame_count > 0) {
      const int available_frames = std::max(video_info->frame_count - warmup, 0);
      if (measured_frames < 0) {
        measured_frames = available_frames;
      } else {
        measured_frames = std::min(measured_frames, available_frames);
      }
    }

    int frames_to_attempt = std::numeric_limits<int>::max();
    if (video_info.has_value() && video_info->frame_count > 0) {
      frames_to_attempt = warmup + std::max(measured_frames, 0);
    } else if (frames >= 0) {
      frames_to_attempt = warmup + frames;
    }

    const int y_channels = static_cast<int>(y_raw_0_spec.shape[1] + y_raw_1_spec.shape[1]);
    const int y_height = static_cast<int>(y_raw_0_spec.shape[2]);
    const int y_width = static_cast<int>(y_raw_0_spec.shape[3]);
    const int z_height = static_cast<int>(z_raw_spec.shape[2]);
    const int z_width = static_cast<int>(z_raw_spec.shape[3]);
    const int z_channel = static_cast<int>(z_raw_spec.shape[1]);
    mlvc::MlvcOfficialEntropyEncoder official_entropy(model_directory);
    mlvc::codec::MlvcRateControlOptions rate_control_options;
    rate_control_options.width = source_width;
    rate_control_options.height = source_height;
    rate_control_options.fps = source_fps;
    rate_control_options.target_bitrate_bps = target_bitrate_bps;
    rate_control_options.default_q_index = q_index;
    rate_control_options.min_q_index = 0;
    rate_control_options.max_q_index = 63;
    mlvc::codec::MlvcRateController rate_control(rate_control_options);

    mlvc::CodecGraphExecutor graph_executor(2);
    mlvc::app::FrameInputArena frame_prepare_arena(frame_spec, 3);
    mlvc::app::AsyncFrameInputQueue prepare_queue(
        frames_to_attempt, frames, input_path, input_frame_dir, frame_spec, source_width,
        source_height, &frame_prepare_arena, nullptr, &graph_executor);

    mlvc::codec::TensorData ref_feature = mlvc::codec::MakeTensorLike(ref_feature_spec);
    mlvc::codec::TensorData ltr_feature = mlvc::codec::MakeTensorLike(ref_feature_spec);
    mlvc::codec::TensorData q_index_tensor = MakeInt32Tensor(q_index_spec, q_index);
    mlvc::codec::TensorData encoder_feature = mlvc::codec::MakeTensorLike(encoder_feature_spec);
    mlvc::codec::TensorData z_raw = mlvc::codec::MakeTensorLike(z_raw_spec);
    mlvc::codec::TensorData y_raw_0 = mlvc::codec::MakeTensorLike(y_raw_0_spec);
    mlvc::codec::TensorData y_raw_1 = mlvc::codec::MakeTensorLike(y_raw_1_spec);
    std::vector<int8_t> z_symbol_buffer;
    std::vector<int8_t> y_symbols_0;
    std::vector<int8_t> y_symbols_1;
    std::vector<uint8_t> scales_0;
    std::vector<uint8_t> scales_1;
    std::vector<uint8_t> payload;
    bool has_ltr_feature = false;

    double feature_copy_ms = 0.0;
    double encoder_ms = 0.0;
    double entropy_encode_ms = 0.0;
    double bitstream_write_ms = 0.0;
    double rate_control_ms = 0.0;
    int processed = 0;
    int i_frames = 0;
    int p_frames = 0;
    int ltr_frames = 0;

    mlvc::io::OfficialMlvcBitstreamWriter bitstream_writer(output_bitstream_path);
    mlvc::EntropyWorker entropy_worker;
    std::deque<std::future<void>> pending_entropy_jobs;
    constexpr std::size_t kMaxPendingEntropyJobs = 2;

    auto encode_and_write = [&](int frame_index, mlvc::codec::MlvcFrameType frame_type,
                                int frame_q_index, const mlvc::codec::TensorData& job_z_raw,
                                const mlvc::codec::TensorData& job_y_raw_0,
                                const mlvc::codec::TensorData& job_y_raw_1, bool measure) {
      std::vector<int8_t> job_z_symbol_buffer;
      std::vector<int8_t> job_y_symbols_0;
      std::vector<int8_t> job_y_symbols_1;
      std::vector<uint8_t> job_scales_0;
      std::vector<uint8_t> job_scales_1;
      std::vector<uint8_t> job_payload;
      const auto entropy_begin = std::chrono::steady_clock::now();
      mlvc::codec::TensorToInt8Symbols(job_z_raw, &job_z_symbol_buffer);
      mlvc::codec::BuildMlvcScaleIndexesFromZRaw(job_z_raw, y_channels, y_height, y_width,
                                                 mlvc::codec::kMlvcChannelRepeat, &job_scales_0,
                                                 &job_scales_1);
      mlvc::codec::TensorToInt8Symbols(job_y_raw_0, &job_y_symbols_0);
      mlvc::codec::TensorToInt8Symbols(job_y_raw_1, &job_y_symbols_1);
      official_entropy.Encode(job_z_symbol_buffer, job_y_symbols_0, job_y_symbols_1, job_scales_0,
                              job_scales_1, frame_q_index, z_channel, z_height, z_width,
                              &job_payload);
      const auto entropy_end = std::chrono::steady_clock::now();
      if (measure) {
        entropy_encode_ms +=
            std::chrono::duration<double, std::milli>(entropy_end - entropy_begin).count();
      }

      const auto write_begin = std::chrono::steady_clock::now();
      bitstream_writer.WriteFrame(frame_q_index, job_payload);
      const auto write_end = std::chrono::steady_clock::now();
      if (measure) {
        bitstream_write_ms +=
            std::chrono::duration<double, std::milli>(write_end - write_begin).count();
      }
      if (!entropy_pipeline) {
        const auto rate_control_update_begin = std::chrono::steady_clock::now();
        rate_control.Update((frame_index + 1) / source_fps, frame_type, frame_q_index,
                            job_payload.size(), mlvc::io::kOfficialMlvcFrameOverheadBytes);
        const auto rate_control_update_end = std::chrono::steady_clock::now();
        if (measure) {
          rate_control_ms += std::chrono::duration<double, std::milli>(rate_control_update_end -
                                                                       rate_control_update_begin)
                                 .count();
        }
      }
    };

    auto drain_entropy_jobs = [&] {
      while (!pending_entropy_jobs.empty()) {
        pending_entropy_jobs.front().get();
        pending_entropy_jobs.pop_front();
      }
    };

    auto run_frame = [&](int frame_index, const mlvc::codec::TensorData& current_frame,
                         bool measure) {
      const int cycle_index = mlvc::codec::MlvcResetCycleIndex(frame_index, reset_interval);
      const mlvc::codec::MlvcFrameType frame_type = mlvc::codec::MlvcFrameTypeForFrame(
          frame_index, gop, reset_interval, ltr_start_idx, ltr_period);
      const bool save_ltr_feature =
          mlvc::codec::ShouldSaveLtrFeatures(cycle_index, ltr_start_idx, ltr_period);
      if (frame_type == mlvc::codec::MlvcFrameType::kIFrame) {
        ResetTensor(&ref_feature);
      } else if (frame_type == mlvc::codec::MlvcFrameType::kLtrRecovery) {
        mlvc::Check(has_ltr_feature, "LTR frame requested before LTR feature is initialized");
        mlvc::codec::CloneTensorInto(ltr_feature, &ref_feature);
      }

      int frame_q_index = q_index;
      const auto rate_control_begin = std::chrono::steady_clock::now();
      if (rate_control.enabled()) {
        frame_q_index = rate_control.SolveQIndex((frame_index + 1) / source_fps, frame_type);
      }
      if (frame_type == mlvc::codec::MlvcFrameType::kLtrRecovery) {
        frame_q_index += ltr_qp_shift;
      }
      frame_q_index = std::clamp(frame_q_index, 0, 63);
      const int q_index_shifted = frame_q_index + kMlvcQIndexShift[frame_index % 8];
      SetInt32TensorValue(&q_index_tensor, q_index_shifted);
      const auto rate_control_end = std::chrono::steady_clock::now();
      if (measure) {
        rate_control_ms +=
            std::chrono::duration<double, std::milli>(rate_control_end - rate_control_begin)
                .count();
      }

      const std::array<mlvc::NamedTensorView, 3> encoder_inputs = {
          mlvc::NamedTensorView{"x", current_frame.View()},
          mlvc::NamedTensorView{"ref_feature", ref_feature.View()},
          mlvc::NamedTensorView{"q_index_shifted", q_index_tensor.View()},
      };
      const std::array<mlvc::NamedTensorView, 4> encoder_output_views = {
          mlvc::NamedTensorView{"feature", encoder_feature.View()},
          mlvc::NamedTensorView{"z_raw", z_raw.View()},
          mlvc::NamedTensorView{"y_raw_0", y_raw_0.View()},
          mlvc::NamedTensorView{"y_raw_1", y_raw_1.View()},
      };

      const auto encoder_begin = std::chrono::steady_clock::now();
      encoder.RunNamed(encoder_inputs.data(), encoder_inputs.size(), encoder_output_views.data(),
                       encoder_output_views.size());
      const auto encoder_end = std::chrono::steady_clock::now();
      if (frame_index == 0) {
        const char* feature_dump_path = std::getenv("MLVC_DUMP_FEATURE0");
        if (feature_dump_path != nullptr && feature_dump_path[0] != '\0') {
          DumpFeatureStats(encoder_feature, feature_dump_path);
        }
      }
      if (measure) {
        encoder_ms +=
            std::chrono::duration<double, std::milli>(encoder_end - encoder_begin).count();
      }

      const auto copy_begin = std::chrono::steady_clock::now();
      mlvc::codec::CloneTensorInto(encoder_feature, &ref_feature);
      const auto copy_end = std::chrono::steady_clock::now();
      if (measure) {
        feature_copy_ms += std::chrono::duration<double, std::milli>(copy_end - copy_begin).count();
      }
      if (save_ltr_feature) {
        mlvc::codec::CloneTensorInto(ref_feature, &ltr_feature);
        has_ltr_feature = true;
      }

      if (entropy_pipeline) {
        mlvc::codec::TensorData job_z_raw = mlvc::codec::CloneTensor(z_raw);
        mlvc::codec::TensorData job_y_raw_0 = mlvc::codec::CloneTensor(y_raw_0);
        mlvc::codec::TensorData job_y_raw_1 = mlvc::codec::CloneTensor(y_raw_1);
        pending_entropy_jobs.push_back(entropy_worker.SubmitVoid(
            [&, frame_index, frame_type, frame_q_index, measure, job_z_raw = std::move(job_z_raw),
             job_y_raw_0 = std::move(job_y_raw_0), job_y_raw_1 = std::move(job_y_raw_1)]() {
              encode_and_write(frame_index, frame_type, frame_q_index, job_z_raw, job_y_raw_0,
                               job_y_raw_1, measure);
            }));
        if (pending_entropy_jobs.size() > kMaxPendingEntropyJobs) {
          pending_entropy_jobs.front().get();
          pending_entropy_jobs.pop_front();
        }
      } else {
        encode_and_write(frame_index, frame_type, frame_q_index, z_raw, y_raw_0, y_raw_1, measure);
      }
      if (frame_type == mlvc::codec::MlvcFrameType::kIFrame) {
        ++i_frames;
      } else if (frame_type == mlvc::codec::MlvcFrameType::kLtrRecovery) {
        ++ltr_frames;
      } else {
        ++p_frames;
      }
    };

    for (int i = 0; i < warmup; ++i) {
      mlvc::app::InputFrame prepared_frame = prepare_queue.Pop();
      if (prepared_frame.eof) {
        throw mlvc::Error("input ended before warmup frames completed");
      }
      mlvc::Check(prepared_frame.frame != nullptr, "prepared frame slot is empty");
      mlvc::Check(prepared_frame.frame_index == i,
                  "async frame prepare queue returned out-of-order frame");
      run_frame(i, *prepared_frame.frame, false);
      prepare_queue.Release(&prepared_frame);
    }
    drain_entropy_jobs();

    const auto begin = std::chrono::steady_clock::now();
    while (measured_frames < 0 || processed < measured_frames) {
      mlvc::app::InputFrame prepared_frame = prepare_queue.Pop();
      if (prepared_frame.eof) {
        if (measured_frames >= 0) {
          throw mlvc::Error("input ended before requested frame count");
        }
        break;
      }
      const int frame_index = warmup + processed;
      mlvc::Check(prepared_frame.frame != nullptr, "prepared frame slot is empty");
      mlvc::Check(prepared_frame.frame_index == frame_index,
                  "async frame prepare queue returned out-of-order frame");
      run_frame(frame_index, *prepared_frame.frame, true);
      prepare_queue.Release(&prepared_frame);
      ++processed;
    }
    drain_entropy_jobs();
    const auto end = std::chrono::steady_clock::now();
    const mlvc::app::AsyncFrameInputQueue::Stats prepare_stats = prepare_queue.stats();
    bitstream_writer.Close();
    const uint64_t bitstream_bytes = std::filesystem::exists(output_bitstream_path)
                                         ? std::filesystem::file_size(output_bitstream_path)
                                         : 0;

    const double total_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    const double fps = processed > 0 ? (1000.0 * static_cast<double>(processed) / total_ms) : 0.0;
    std::cout << "stage=MLVCEncoder\n";
    std::cout << "pipeline=" << (entropy_pipeline ? "input+npu-rans" : "input-only") << "\n";
    std::cout << "entropy_pipeline=" << (entropy_pipeline ? 1 : 0) << "\n";
    std::cout << "entropy_pipeline_max_pending=" << kMaxPendingEntropyJobs << "\n";
    if (input_frame_dir.empty()) {
      std::cout << "input_mode=video\n";
      std::cout << "input_video=" << input_path.string() << "\n";
    } else {
      std::cout << "input_mode=frame_dir\n";
      std::cout << "input_frame_dir=" << input_frame_dir.string() << "\n";
    }
    std::cout << "warmup=" << warmup << "\n";
    std::cout << "gop=" << gop << "\n";
    std::cout << "reset_interval=" << reset_interval << "\n";
    std::cout << "ltr_start_idx=" << ltr_start_idx << "\n";
    std::cout << "ltr_period=" << ltr_period << "\n";
    std::cout << "ltr_qp_shift=" << ltr_qp_shift << "\n";
    std::cout << "target_bitrate_bps=" << target_bitrate_bps << "\n";
    std::cout << "rate_control_enabled=" << (rate_control.enabled() ? 1 : 0) << "\n";
    std::cout << "frames=" << processed << "\n";
    std::cout << "total_ms=" << total_ms << "\n";
    std::cout << "avg_ms=" << (processed > 0 ? total_ms / static_cast<double>(processed) : 0.0)
              << "\n";
    std::cout << "fps=" << fps << "\n";
    std::cout << "breakdown:\n";
    std::cout << "  queue_consumer_wait_ms=" << prepare_stats.consumer_wait_ms << " avg="
              << (prepare_stats.consumer_wait_count > 0
                      ? prepare_stats.consumer_wait_ms /
                            static_cast<double>(prepare_stats.consumer_wait_count)
                      : 0.0)
              << "\n";
    std::cout << "  queue_producer_wait_ms=" << prepare_stats.producer_wait_ms << " avg="
              << (prepare_stats.producer_wait_count > 0
                      ? prepare_stats.producer_wait_ms /
                            static_cast<double>(prepare_stats.producer_wait_count)
                      : 0.0)
              << "\n";
    std::cout << "  prepare_ms=" << prepare_stats.prepare_ms << " avg="
              << (prepare_stats.prepared_frames > 0
                      ? prepare_stats.prepare_ms /
                            static_cast<double>(prepare_stats.prepared_frames)
                      : 0.0)
              << "\n";
    std::cout << "  prepare_video_read_ms=" << prepare_stats.video_read_ms << " avg="
              << (prepare_stats.prepared_frames > 0
                      ? prepare_stats.video_read_ms /
                            static_cast<double>(prepare_stats.prepared_frames)
                      : 0.0)
              << "\n";
    std::cout << "  prepare_fp16_read_ms=" << prepare_stats.fp16_read_ms << " avg="
              << (prepare_stats.prepared_frames > 0
                      ? prepare_stats.fp16_read_ms /
                            static_cast<double>(prepare_stats.prepared_frames)
                      : 0.0)
              << "\n";
    std::cout << "  prepare_synthetic_ms=" << prepare_stats.synthetic_read_ms << " avg="
              << (prepare_stats.prepared_frames > 0
                      ? prepare_stats.synthetic_read_ms /
                            static_cast<double>(prepare_stats.prepared_frames)
                      : 0.0)
              << "\n";
    std::cout << "  encoder_ms=" << encoder_ms
              << " avg=" << (processed > 0 ? encoder_ms / static_cast<double>(processed) : 0.0)
              << "\n";
    std::cout << "  feature_copy_ms=" << feature_copy_ms
              << " avg=" << (processed > 0 ? feature_copy_ms / static_cast<double>(processed) : 0.0)
              << "\n";
    std::cout << "  rate_control_ms=" << rate_control_ms
              << " avg=" << (processed > 0 ? rate_control_ms / static_cast<double>(processed) : 0.0)
              << "\n";
    std::cout << "  entropy_encode_ms=" << entropy_encode_ms << " avg="
              << (processed > 0 ? entropy_encode_ms / static_cast<double>(processed) : 0.0) << "\n";
    std::cout << "  bitstream_write_ms=" << bitstream_write_ms << " avg="
              << (processed > 0 ? bitstream_write_ms / static_cast<double>(processed) : 0.0)
              << "\n";
    std::cout << "output_bitstream=" << output_bitstream_path.string() << "\n";
    std::cout << "bitstream_bytes=" << bitstream_bytes << "\n";
    std::cout << "frame_counts=i:" << i_frames << ",p:" << p_frames << ",ltr:" << ltr_frames
              << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "benchmark_acl_encoder failed: " << error.what() << "\n";
    return 1;
  }
}
