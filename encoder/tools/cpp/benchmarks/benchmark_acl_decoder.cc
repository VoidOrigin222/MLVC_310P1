#include <mlvc/codec/mlvc_entropy.h>
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
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace {

constexpr std::string_view kDefaultManifest = "../mlvc1080p/manifest.json";
constexpr int kDefaultQIndex = 21;
constexpr int kDefaultGop = 32;
constexpr int kDefaultResetInterval = 16;
constexpr int kDefaultLtrStartIdx = 8;
constexpr int kDefaultLtrPeriod = 64;
constexpr int kDefaultLtrQpShift = 8;
constexpr std::array<int, 8> kMlvcQIndexShift = {8, 0, 4, 0, 4, 0, 4, 0};

template <typename T>
class BlockingQueue {
 public:
  void Push(T value) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closed_) {
        throw mlvc::Error("push on closed queue");
      }
      queue_.push_back(std::move(value));
    }
    condition_.notify_one();
  }

  bool Pop(T* value) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return closed_ || !queue_.empty(); });
    if (queue_.empty()) {
      return false;
    }
    *value = std::move(queue_.front());
    queue_.pop_front();
    return true;
  }

  void Close() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    condition_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<T> queue_;
  bool closed_ = false;
};

struct OutputFrame {
  int frame_index = 0;
  mlvc::codec::TensorData tensor;
};

struct DecodedEntropyFrame {
  int frame_index = 0;
  mlvc::codec::MlvcFrameType frame_type = mlvc::codec::MlvcFrameType::kPFrame;
  int q_index = 0;
  mlvc::codec::TensorData z_raw;
  mlvc::codec::TensorData y_raw_0;
  mlvc::codec::TensorData y_raw_1;
};

void PrintUsage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " --input-bitstream <stream.bin> [--output-video <output>]"
               " [--output-format mp4|png] [--manifest <manifest.json>] [--device 0]"
               " [--fps 30] [--warmup 16] [--frames -1] [--q-index 21]"
               " [--gop 32] [--reset-interval 16]"
               " [--ltr-start-idx 8] [--ltr-period 64] [--ltr-qp-shift 8]"
               " [--entropy-pipeline]\n";
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

void ResetTensor(mlvc::codec::TensorData* tensor) {
  std::fill(tensor->bytes.begin(), tensor->bytes.end(), 0);
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path manifest_path = std::filesystem::path(kDefaultManifest);
  std::filesystem::path input_bitstream_path;
  std::filesystem::path output_video_path;
  std::string output_format;
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
  bool entropy_pipeline = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--manifest" && i + 1 < argc) {
      manifest_path = argv[++i];
    } else if (arg == "--input-bitstream" && i + 1 < argc) {
      input_bitstream_path = argv[++i];
    } else if (arg == "--output-video" && i + 1 < argc) {
      output_video_path = argv[++i];
    } else if (arg == "--output-format" && i + 1 < argc) {
      output_format = argv[++i];
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

  if (input_bitstream_path.empty() || warmup < 0 || frames == 0 || gop < 0 || reset_interval < 0) {
    PrintUsage(argv[0]);
    return 2;
  }
  mlvc::Check(ltr_start_idx >= 0, "ltr_start_idx must be non-negative");
  mlvc::Check(ltr_period >= 0, "ltr_period must be non-negative");

  try {
    mlvc::StageRuntime runtime(device);
    mlvc::ModelManifest manifest = mlvc::ModelManifest::Load(manifest_path);
    const std::filesystem::path model_directory = manifest.directory();
    mlvc::Check(manifest.HasModel("MLVCEncoder") && manifest.HasModel("MLVCDecoder"),
                "e1d1 manifest must contain MLVCEncoder and MLVCDecoder");
    mlvc::StageModelSet models(&runtime, std::move(manifest));
    mlvc::StageModel& decoder = models.GetStage(std::string("MLVCDecoder"));

    const mlvc::TensorSpec& ref_feature_spec = FindTensorSpec(decoder.record(), "ref_feature");
    const mlvc::TensorSpec& q_index_spec = FindTensorSpec(decoder.record(), "q_index_shifted");
    const mlvc::TensorSpec& z_raw_spec = FindTensorSpec(decoder.record(), "z_raw");
    const mlvc::TensorSpec& y_raw_0_spec = FindTensorSpec(decoder.record(), "y_raw_0");
    const mlvc::TensorSpec& y_raw_1_spec = FindTensorSpec(decoder.record(), "y_raw_1");
    const mlvc::TensorSpec& x_hat_spec = FindTensorSpec(decoder.record(), "x_hat");
    const mlvc::TensorSpec& decoded_feature_spec = FindTensorSpec(decoder.record(), "feature");

    mlvc::Check(decoder.record().inputs.size() == 5, "MLVCDecoder input count mismatch");
    mlvc::Check(decoder.record().outputs.size() == 2, "MLVCDecoder output count mismatch");

    mlvc::io::OfficialMlvcBitstreamReader bitstream_reader(input_bitstream_path);
    const int source_width = static_cast<int>(x_hat_spec.shape[3]);
    const int model_height = static_cast<int>(x_hat_spec.shape[2]);
    const int source_height = model_height == 1088 ? 1080 : model_height;
    const double source_fps = fallback_fps;
    const int effective_gop = gop;
    const int effective_reset_interval = reset_interval;
    const int effective_ltr_start_idx = ltr_start_idx;
    const int effective_ltr_period = ltr_period;
    const int effective_ltr_qp_shift = ltr_qp_shift;
    const int y_channels = static_cast<int>(y_raw_0_spec.shape[1] + y_raw_1_spec.shape[1]);
    const int y_height = static_cast<int>(y_raw_0_spec.shape[2]);
    const int y_width = static_cast<int>(y_raw_0_spec.shape[3]);
    const int z_height = static_cast<int>(z_raw_spec.shape[2]);
    const int z_width = static_cast<int>(z_raw_spec.shape[3]);
    const int z_channel = static_cast<int>(z_raw_spec.shape[1]);
    mlvc::MlvcOfficialEntropyDecoder official_entropy(model_directory);

    mlvc::codec::TensorData ref_feature = mlvc::codec::MakeTensorLike(ref_feature_spec);
    mlvc::codec::TensorData ltr_feature = mlvc::codec::MakeTensorLike(ref_feature_spec);
    mlvc::codec::TensorData z_raw = mlvc::codec::MakeTensorLike(z_raw_spec);
    mlvc::codec::TensorData y_raw_0 = mlvc::codec::MakeTensorLike(y_raw_0_spec);
    mlvc::codec::TensorData y_raw_1 = mlvc::codec::MakeTensorLike(y_raw_1_spec);
    mlvc::codec::TensorData x_hat = mlvc::codec::MakeTensorLike(x_hat_spec);
    mlvc::codec::TensorData decoded_feature = mlvc::codec::MakeTensorLike(decoded_feature_spec);
    std::vector<uint8_t> payload;
    std::vector<uint8_t> scales_0;
    std::vector<uint8_t> scales_1;
    std::vector<int8_t> z_symbols_buffer;
    std::vector<int8_t> y_symbols_0;
    std::vector<int8_t> y_symbols_1;
    bool has_ltr_feature = false;

    double bitstream_read_ms = 0.0;
    double entropy_decode_ms = 0.0;
    double decoder_ms = 0.0;
    double feature_copy_ms = 0.0;
    double output_push_ms = 0.0;
    double writer_thread_ms = 0.0;
    int next_frame_index = 0;
    std::optional<mlvc::io::DecodedVideoWriter> writer;
    std::optional<std::thread> writer_thread;
    BlockingQueue<OutputFrame> output_queue;
    std::exception_ptr writer_error;
    if (!output_video_path.empty()) {
      writer.emplace(output_video_path, output_format, source_fps, source_width, source_height, 23,
                     "", "medium");
      writer_thread.emplace([&] {
        try {
          OutputFrame output;
          while (output_queue.Pop(&output)) {
            const auto write_begin = std::chrono::steady_clock::now();
            writer->WriteTensorFrame(output.tensor, output.frame_index);
            const auto write_end = std::chrono::steady_clock::now();
            writer_thread_ms +=
                std::chrono::duration<double, std::milli>(write_end - write_begin).count();
          }
          writer->Close();
        } catch (...) {
          writer_error = std::current_exception();
          try {
            writer->Close();
          } catch (...) {
          }
        }
      });
    }

    mlvc::EntropyWorker entropy_worker;
    auto decode_entropy_frame = [&](bool measure) -> std::optional<DecodedEntropyFrame> {
      DecodedEntropyFrame decoded;
      std::vector<uint8_t> job_payload;
      const auto read_begin = std::chrono::steady_clock::now();
      const bool has_frame =
          bitstream_reader.ReadFrame(&decoded.frame_index, &decoded.q_index, &job_payload);
      const auto read_end = std::chrono::steady_clock::now();
      if (!has_frame) {
        return std::nullopt;
      }
      if (decoded.q_index < 0) {
        decoded.q_index = q_index;
      }
      mlvc::Check(decoded.frame_index == next_frame_index,
                  "mlvc bitstream frame index out of order");
      ++next_frame_index;
      decoded.frame_type = mlvc::codec::MlvcFrameTypeForFrame(
          decoded.frame_index, effective_gop, effective_reset_interval, effective_ltr_start_idx,
          effective_ltr_period);
      if (measure) {
        bitstream_read_ms +=
            std::chrono::duration<double, std::milli>(read_end - read_begin).count();
      }

      decoded.z_raw = mlvc::codec::MakeTensorLike(z_raw_spec);
      decoded.y_raw_0 = mlvc::codec::MakeTensorLike(y_raw_0_spec);
      decoded.y_raw_1 = mlvc::codec::MakeTensorLike(y_raw_1_spec);
      std::vector<uint8_t> job_scales_0;
      std::vector<uint8_t> job_scales_1;
      std::vector<int8_t> job_y_symbols_0;
      std::vector<int8_t> job_y_symbols_1;
      const auto entropy_begin = std::chrono::steady_clock::now();
      official_entropy.SetStream(job_payload);
      const std::vector<int8_t> job_z_symbol_buffer =
          official_entropy.DecodeZ(decoded.q_index, z_channel, z_height, z_width);
      mlvc::codec::FloatToFp16TensorFromInt8(job_z_symbol_buffer,
                                             mlvc::TensorShape(z_raw_spec.shape), &decoded.z_raw);
      mlvc::codec::BuildMlvcScaleIndexesFromZRaw(decoded.z_raw, y_channels, y_height, y_width,
                                                 mlvc::codec::kMlvcChannelRepeat, &job_scales_0,
                                                 &job_scales_1);
      job_y_symbols_0 = official_entropy.DecodeY(job_scales_0, false);
      job_y_symbols_1 = official_entropy.DecodeY(job_scales_1, true);
      mlvc::codec::FloatToFp16TensorFromInt8(job_y_symbols_0, mlvc::TensorShape(y_raw_0_spec.shape),
                                             &decoded.y_raw_0);
      mlvc::codec::FloatToFp16TensorFromInt8(job_y_symbols_1, mlvc::TensorShape(y_raw_1_spec.shape),
                                             &decoded.y_raw_1);
      const auto entropy_end = std::chrono::steady_clock::now();
      if (measure) {
        entropy_decode_ms +=
            std::chrono::duration<double, std::milli>(entropy_end - entropy_begin).count();
      }
      return decoded;
    };

    auto run_decoded_frame = [&](const DecodedEntropyFrame& decoded, bool measure) {
      const int cycle_index =
          mlvc::codec::MlvcResetCycleIndex(decoded.frame_index, effective_reset_interval);
      const bool save_ltr_feature = mlvc::codec::ShouldSaveLtrFeatures(
          cycle_index, effective_ltr_start_idx, effective_ltr_period);
      if (decoded.frame_type == mlvc::codec::MlvcFrameType::kIFrame) {
        ResetTensor(&ref_feature);
      } else if (decoded.frame_type == mlvc::codec::MlvcFrameType::kLtrRecovery) {
        mlvc::Check(has_ltr_feature, "LTR frame requested before LTR feature is initialized");
        mlvc::codec::CloneTensorInto(ltr_feature, &ref_feature);
      }

      const int q_index_shifted = decoded.q_index + kMlvcQIndexShift[decoded.frame_index % 8];
      mlvc::codec::TensorData q_index_tensor = MakeInt32Tensor(q_index_spec, q_index_shifted);
      const std::array<mlvc::NamedTensorView, 5> decoder_inputs = {
          mlvc::NamedTensorView{"z_raw", decoded.z_raw.View()},
          mlvc::NamedTensorView{"y_raw_0", decoded.y_raw_0.View()},
          mlvc::NamedTensorView{"y_raw_1", decoded.y_raw_1.View()},
          mlvc::NamedTensorView{"ref_feature", ref_feature.View()},
          mlvc::NamedTensorView{"q_index_shifted", q_index_tensor.View()},
      };
      const std::array<mlvc::NamedTensorView, 2> decoder_output_views = {
          mlvc::NamedTensorView{"x_hat", x_hat.View()},
          mlvc::NamedTensorView{"feature", decoded_feature.View()},
      };
      const auto decoder_begin = std::chrono::steady_clock::now();
      decoder.RunNamed(decoder_inputs.data(), decoder_inputs.size(), decoder_output_views.data(),
                       decoder_output_views.size());
      const auto decoder_end = std::chrono::steady_clock::now();
      if (measure) {
        decoder_ms +=
            std::chrono::duration<double, std::milli>(decoder_end - decoder_begin).count();
      }

      const auto copy_begin = std::chrono::steady_clock::now();
      mlvc::codec::CloneTensorInto(decoded_feature, &ref_feature);
      const auto copy_end = std::chrono::steady_clock::now();
      if (measure) {
        feature_copy_ms += std::chrono::duration<double, std::milli>(copy_end - copy_begin).count();
      }
      if (save_ltr_feature) {
        mlvc::codec::CloneTensorInto(ref_feature, &ltr_feature);
        has_ltr_feature = true;
      }
      if (writer.has_value()) {
        const auto push_begin = std::chrono::steady_clock::now();
        output_queue.Push(OutputFrame{decoded.frame_index, mlvc::codec::CloneTensor(x_hat)});
        const auto push_end = std::chrono::steady_clock::now();
        if (measure) {
          output_push_ms +=
              std::chrono::duration<double, std::milli>(push_end - push_begin).count();
        }
      }
    };

    auto run_frame = [&](bool measure) {
      int frame_index = 0;
      int frame_q_index = 0;
      const auto read_begin = std::chrono::steady_clock::now();
      const bool has_frame = bitstream_reader.ReadFrame(&frame_index, &frame_q_index, &payload);
      const auto read_end = std::chrono::steady_clock::now();
      if (!has_frame) {
        return false;
      }
      if (frame_q_index < 0) {
        frame_q_index = q_index;
      }
      mlvc::Check(frame_index == next_frame_index, "mlvc bitstream frame index out of order");
      ++next_frame_index;
      const mlvc::codec::MlvcFrameType frame_type =
          mlvc::codec::MlvcFrameTypeForFrame(frame_index, effective_gop, effective_reset_interval,
                                             effective_ltr_start_idx, effective_ltr_period);
      if (measure) {
        bitstream_read_ms +=
            std::chrono::duration<double, std::milli>(read_end - read_begin).count();
      }

      const int cycle_index =
          mlvc::codec::MlvcResetCycleIndex(frame_index, effective_reset_interval);
      const bool save_ltr_feature = mlvc::codec::ShouldSaveLtrFeatures(
          cycle_index, effective_ltr_start_idx, effective_ltr_period);
      if (frame_type == mlvc::codec::MlvcFrameType::kIFrame) {
        ResetTensor(&ref_feature);
      } else if (frame_type == mlvc::codec::MlvcFrameType::kLtrRecovery) {
        mlvc::Check(has_ltr_feature, "LTR frame requested before LTR feature is initialized");
        mlvc::codec::CloneTensorInto(ltr_feature, &ref_feature);
      }

      official_entropy.SetStream(payload);
      const auto entropy_begin = std::chrono::steady_clock::now();
      const std::vector<int8_t> z_symbol_buffer =
          official_entropy.DecodeZ(frame_q_index, z_channel, z_height, z_width);
      mlvc::codec::FloatToFp16TensorFromInt8(z_symbol_buffer, mlvc::TensorShape(z_raw_spec.shape),
                                             &z_raw);
      mlvc::codec::BuildMlvcScaleIndexesFromZRaw(z_raw, y_channels, y_height, y_width,
                                                 mlvc::codec::kMlvcChannelRepeat, &scales_0,
                                                 &scales_1);
      y_symbols_0 = official_entropy.DecodeY(scales_0, false);
      y_symbols_1 = official_entropy.DecodeY(scales_1, true);
      mlvc::codec::FloatToFp16TensorFromInt8(y_symbols_0, mlvc::TensorShape(y_raw_0_spec.shape),
                                             &y_raw_0);
      mlvc::codec::FloatToFp16TensorFromInt8(y_symbols_1, mlvc::TensorShape(y_raw_1_spec.shape),
                                             &y_raw_1);
      const auto entropy_end = std::chrono::steady_clock::now();
      if (measure) {
        entropy_decode_ms +=
            std::chrono::duration<double, std::milli>(entropy_end - entropy_begin).count();
      }

      const int q_index_shifted = frame_q_index + kMlvcQIndexShift[frame_index % 8];
      mlvc::codec::TensorData q_index_tensor = MakeInt32Tensor(q_index_spec, q_index_shifted);
      const std::array<mlvc::NamedTensorView, 5> decoder_inputs = {
          mlvc::NamedTensorView{"z_raw", z_raw.View()},
          mlvc::NamedTensorView{"y_raw_0", y_raw_0.View()},
          mlvc::NamedTensorView{"y_raw_1", y_raw_1.View()},
          mlvc::NamedTensorView{"ref_feature", ref_feature.View()},
          mlvc::NamedTensorView{"q_index_shifted", q_index_tensor.View()},
      };
      const std::array<mlvc::NamedTensorView, 2> decoder_output_views = {
          mlvc::NamedTensorView{"x_hat", x_hat.View()},
          mlvc::NamedTensorView{"feature", decoded_feature.View()},
      };

      const auto decoder_begin = std::chrono::steady_clock::now();
      decoder.RunNamed(decoder_inputs.data(), decoder_inputs.size(), decoder_output_views.data(),
                       decoder_output_views.size());
      const auto decoder_end = std::chrono::steady_clock::now();
      if (measure) {
        decoder_ms +=
            std::chrono::duration<double, std::milli>(decoder_end - decoder_begin).count();
      }

      const auto copy_begin = std::chrono::steady_clock::now();
      mlvc::codec::CloneTensorInto(decoded_feature, &ref_feature);
      const auto copy_end = std::chrono::steady_clock::now();
      if (measure) {
        feature_copy_ms += std::chrono::duration<double, std::milli>(copy_end - copy_begin).count();
      }
      if (save_ltr_feature) {
        mlvc::codec::CloneTensorInto(ref_feature, &ltr_feature);
        has_ltr_feature = true;
      }

      if (writer.has_value()) {
        const auto push_begin = std::chrono::steady_clock::now();
        output_queue.Push(OutputFrame{frame_index, mlvc::codec::CloneTensor(x_hat)});
        const auto push_end = std::chrono::steady_clock::now();
        if (measure) {
          output_push_ms +=
              std::chrono::duration<double, std::milli>(push_end - push_begin).count();
        }
      }
      return true;
    };

    int processed = 0;
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    if (!entropy_pipeline) {
      for (int i = 0; i < warmup; ++i) {
        if (!run_frame(false)) {
          throw mlvc::Error("input ended before warmup frames completed");
        }
      }
      begin = std::chrono::steady_clock::now();
      while (frames < 0 || processed < frames) {
        if (!run_frame(true)) {
          if (frames >= 0) {
            throw mlvc::Error("input ended before requested frame count");
          }
          break;
        }
        ++processed;
      }
      end = std::chrono::steady_clock::now();
    } else {
      auto submit_decode = [&](bool measure) {
        return entropy_worker.SubmitValue<std::optional<DecodedEntropyFrame>>(
            [&, measure] { return decode_entropy_frame(measure); });
      };
      std::optional<std::future<std::optional<DecodedEntropyFrame>>> next;
      if (warmup > 0) {
        next.emplace(submit_decode(false));
      }
      for (int i = 0; i < warmup; ++i) {
        std::optional<DecodedEntropyFrame> decoded = next->get();
        if (!decoded.has_value()) {
          throw mlvc::Error("input ended before warmup frames completed");
        }
        if (i + 1 < warmup) {
          next.emplace(submit_decode(false));
        }
        run_decoded_frame(*decoded, false);
      }

      begin = std::chrono::steady_clock::now();
      next.emplace(submit_decode(true));
      while (frames < 0 || processed < frames) {
        std::optional<DecodedEntropyFrame> decoded = next->get();
        if (!decoded.has_value()) {
          if (frames >= 0) {
            throw mlvc::Error("input ended before requested frame count");
          }
          break;
        }
        next.emplace(submit_decode(true));
        run_decoded_frame(*decoded, true);
        ++processed;
      }
      end = std::chrono::steady_clock::now();
    }

    output_queue.Close();
    if (writer_thread.has_value()) {
      writer_thread->join();
      if (writer_error) {
        std::rethrow_exception(writer_error);
      }
    }
    const double total_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    const double fps = processed > 0 ? (1000.0 * static_cast<double>(processed) / total_ms) : 0.0;
    std::cout << "stage=MLVCDecoder\n";
    std::cout << "pipeline=" << (entropy_pipeline ? "npu-rans" : "single-stage") << "\n";
    std::cout << "entropy_pipeline=" << (entropy_pipeline ? 1 : 0) << "\n";
    std::cout << "input_bitstream=" << input_bitstream_path.string() << "\n";
    std::cout << "input_bitstream_bytes="
              << (std::filesystem::exists(input_bitstream_path)
                      ? std::filesystem::file_size(input_bitstream_path)
                      : 0)
              << "\n";
    std::cout << "warmup=" << warmup << "\n";
    std::cout << "gop=" << effective_gop << "\n";
    std::cout << "reset_interval=" << effective_reset_interval << "\n";
    std::cout << "ltr_start_idx=" << effective_ltr_start_idx << "\n";
    std::cout << "ltr_period=" << effective_ltr_period << "\n";
    std::cout << "ltr_qp_shift=" << effective_ltr_qp_shift << "\n";
    std::cout << "frames=" << processed << "\n";
    std::cout << "total_ms=" << total_ms << "\n";
    std::cout << "avg_ms=" << (processed > 0 ? total_ms / static_cast<double>(processed) : 0.0)
              << "\n";
    std::cout << "fps=" << fps << "\n";
    std::cout << "breakdown:\n";
    std::cout << "  bitstream_read_ms=" << bitstream_read_ms << " avg="
              << (processed > 0 ? bitstream_read_ms / static_cast<double>(processed) : 0.0) << "\n";
    std::cout << "  entropy_decode_ms=" << entropy_decode_ms << " avg="
              << (processed > 0 ? entropy_decode_ms / static_cast<double>(processed) : 0.0) << "\n";
    std::cout << "  decoder_ms=" << decoder_ms
              << " avg=" << (processed > 0 ? decoder_ms / static_cast<double>(processed) : 0.0)
              << "\n";
    std::cout << "  feature_copy_ms=" << feature_copy_ms
              << " avg=" << (processed > 0 ? feature_copy_ms / static_cast<double>(processed) : 0.0)
              << "\n";
    if (writer.has_value()) {
      std::cout << "  output_push_ms=" << output_push_ms << " avg="
                << (processed > 0 ? output_push_ms / static_cast<double>(processed) : 0.0) << "\n";
    }
    if (writer.has_value()) {
      std::cout << "  writer_thread_ms=" << writer_thread_ms << " avg="
                << (processed > 0 ? writer_thread_ms / static_cast<double>(processed) : 0.0)
                << "\n";
      if (mlvc::io::GuessDecodeOutputFormat(output_video_path, output_format) == "png") {
        std::cout << "output_frame_dir=" << output_video_path.string() << "\n";
      } else {
        std::cout << "output_video=" << output_video_path.string() << "\n";
      }
    } else {
      std::cout << "output_mode=none\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "benchmark_acl_decoder failed: " << error.what() << "\n";
    return 1;
  }
}
