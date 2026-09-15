#include <mlvc/application/input/frame_input_queue.h>
#include <mlvc/core/status.h>
#include <mlvc/io/video_io.h>
#include <mlvc/runtime/stage_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
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

void PrintUsage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " (--video <input.mp4> | --input-frame-dir <dir>)"
               " [--manifest <manifest.json>] [--device 0] [--fps 30]"
               " [--warmup 16] [--frames -1] [--q-index 21]"
               " [--output-video <output.mp4>]\n";
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

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path manifest_path = std::filesystem::path(kDefaultManifest);
  std::filesystem::path input_path;
  std::filesystem::path input_frame_dir;
  std::filesystem::path output_video_path;
  int device = 0;
  double fallback_fps = 30.0;
  int warmup = 16;
  int frames = -1;
  int q_index = kDefaultQIndex;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--manifest" && i + 1 < argc) {
      manifest_path = argv[++i];
    } else if (arg == "--video" && i + 1 < argc) {
      input_path = argv[++i];
    } else if (arg == "--input-frame-dir" && i + 1 < argc) {
      input_frame_dir = argv[++i];
    } else if (arg == "--output-video" && i + 1 < argc) {
      output_video_path = argv[++i];
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
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    } else {
      PrintUsage(argv[0]);
      return 2;
    }
  }

  if ((input_path.empty() && input_frame_dir.empty()) || warmup < 0 || frames == 0) {
    PrintUsage(argv[0]);
    return 2;
  }
  mlvc::Check(input_path.empty() || input_frame_dir.empty(),
              "--video and --input-frame-dir are mutually exclusive");

  try {
    mlvc::StageRuntime runtime(device);
    mlvc::ModelManifest manifest = mlvc::ModelManifest::Load(manifest_path);
    mlvc::StageModelSet models(&runtime, std::move(manifest));
    mlvc::StageModel& encoder = models.GetStage(std::string("MLVCEncoder"));
    mlvc::StageModel& decoder = models.GetStage(std::string("MLVCDecoder"));

    const mlvc::TensorSpec& frame_spec = FindTensorSpec(encoder.record(), "x");
    const mlvc::TensorSpec& ref_feature_spec = FindTensorSpec(encoder.record(), "ref_feature");
    const mlvc::TensorSpec& q_index_spec = FindTensorSpec(encoder.record(), "q_index_shifted");
    const mlvc::TensorSpec& encoder_feature_spec = FindTensorSpec(encoder.record(), "feature");
    const mlvc::TensorSpec& z_raw_spec = FindTensorSpec(encoder.record(), "z_raw");
    const mlvc::TensorSpec& y_raw_0_spec = FindTensorSpec(encoder.record(), "y_raw_0");
    const mlvc::TensorSpec& y_raw_1_spec = FindTensorSpec(encoder.record(), "y_raw_1");
    const mlvc::TensorSpec& x_hat_spec = FindTensorSpec(decoder.record(), "x_hat");
    const mlvc::TensorSpec& decoder_feature_spec = FindTensorSpec(decoder.record(), "feature");

    mlvc::Check(encoder.record().inputs.size() == 3, "MLVCEncoder input count mismatch");
    mlvc::Check(encoder.record().outputs.size() == 4, "MLVCEncoder output count mismatch");
    mlvc::Check(decoder.record().inputs.size() == 5, "MLVCDecoder input count mismatch");
    mlvc::Check(decoder.record().outputs.size() == 2, "MLVCDecoder output count mismatch");

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

    mlvc::CodecGraphExecutor graph_executor(2);
    mlvc::app::FrameInputArena frame_prepare_arena(frame_spec, 3);
    mlvc::app::AsyncFrameInputQueue prepare_queue(
        frames_to_attempt, frames, input_path, input_frame_dir, frame_spec, source_width,
        source_height, &frame_prepare_arena, nullptr, &graph_executor);

    mlvc::codec::TensorData ref_feature = mlvc::codec::MakeTensorLike(ref_feature_spec);
    mlvc::codec::TensorData q_index_tensor = MakeInt32Tensor(q_index_spec, q_index);
    mlvc::codec::TensorData encoder_feature = mlvc::codec::MakeTensorLike(encoder_feature_spec);
    mlvc::codec::TensorData z_raw = mlvc::codec::MakeTensorLike(z_raw_spec);
    mlvc::codec::TensorData y_raw_0 = mlvc::codec::MakeTensorLike(y_raw_0_spec);
    mlvc::codec::TensorData y_raw_1 = mlvc::codec::MakeTensorLike(y_raw_1_spec);
    mlvc::codec::TensorData x_hat = mlvc::codec::MakeTensorLike(x_hat_spec);
    mlvc::codec::TensorData decoded_feature = mlvc::codec::MakeTensorLike(decoder_feature_spec);

    double async_wait_ms = 0.0;
    double encoder_ms = 0.0;
    double decoder_ms = 0.0;
    double feature_copy_ms = 0.0;
    double output_push_ms = 0.0;
    double writer_ms = 0.0;
    BlockingQueue<OutputFrame> output_queue;
    std::optional<mlvc::io::DecodedVideoWriter> writer;
    std::optional<std::thread> writer_thread;
    std::exception_ptr writer_error;
    if (!output_video_path.empty()) {
      writer.emplace(output_video_path, "", source_fps, source_width, source_height, 23, "",
                     "medium");
      writer_thread.emplace([&] {
        try {
          OutputFrame output;
          while (output_queue.Pop(&output)) {
            const auto write_begin = std::chrono::steady_clock::now();
            writer->WriteTensorFrame(output.tensor, output.frame_index);
            const auto write_end = std::chrono::steady_clock::now();
            writer_ms += std::chrono::duration<double, std::milli>(write_end - write_begin).count();
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

    auto process_frame = [&](int frame_index, const mlvc::codec::TensorData& current_frame) {
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

      const auto encoder_begin = std::chrono::steady_clock::now();
      encoder.RunNamed(encoder_inputs.data(), encoder_inputs.size(), encoder_output_views.data(),
                       encoder_output_views.size());
      const auto encoder_end = std::chrono::steady_clock::now();
      encoder_ms += std::chrono::duration<double, std::milli>(encoder_end - encoder_begin).count();

      const auto decoder_begin = std::chrono::steady_clock::now();
      decoder.RunNamed(decoder_inputs.data(), decoder_inputs.size(), decoder_output_views.data(),
                       decoder_output_views.size());
      const auto decoder_end = std::chrono::steady_clock::now();
      decoder_ms += std::chrono::duration<double, std::milli>(decoder_end - decoder_begin).count();

      const auto copy_begin = std::chrono::steady_clock::now();
      mlvc::codec::CloneTensorInto(decoded_feature, &ref_feature);
      const auto copy_end = std::chrono::steady_clock::now();
      feature_copy_ms += std::chrono::duration<double, std::milli>(copy_end - copy_begin).count();
      if (writer.has_value()) {
        const auto push_begin = std::chrono::steady_clock::now();
        output_queue.Push(OutputFrame{frame_index, mlvc::codec::CloneTensor(x_hat)});
        const auto push_end = std::chrono::steady_clock::now();
        output_push_ms += std::chrono::duration<double, std::milli>(push_end - push_begin).count();
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
      process_frame(i, *prepared_frame.frame);
      prepare_queue.Release(&prepared_frame);
    }

    const auto begin = std::chrono::steady_clock::now();
    int processed = 0;
    while (measured_frames < 0 || processed < measured_frames) {
      const auto pop_begin = std::chrono::steady_clock::now();
      mlvc::app::InputFrame prepared_frame = prepare_queue.Pop();
      const auto pop_end = std::chrono::steady_clock::now();
      async_wait_ms += std::chrono::duration<double, std::milli>(pop_end - pop_begin).count();
      if (prepared_frame.eof) {
        if (measured_frames >= 0) {
          throw mlvc::Error("input ended before requested frame count");
        }
        break;
      }
      mlvc::Check(prepared_frame.frame != nullptr, "prepared frame slot is empty");
      const int frame_index = warmup + processed;
      mlvc::Check(prepared_frame.frame_index == frame_index,
                  "async frame prepare queue returned out-of-order frame");
      process_frame(frame_index, *prepared_frame.frame);
      prepare_queue.Release(&prepared_frame);
      ++processed;
    }

    const mlvc::app::AsyncFrameInputQueue::Stats prepare_stats = prepare_queue.stats();
    output_queue.Close();
    if (writer_thread.has_value()) {
      writer_thread->join();
      if (writer_error) {
        std::rethrow_exception(writer_error);
      }
    }
    const auto end = std::chrono::steady_clock::now();

    const double total_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    const double fps = processed > 0 ? (1000.0 * static_cast<double>(processed) / total_ms) : 0.0;
    std::cout << "stage=MLVCEncoder+MLVCDecoder\n";
    std::cout << "pipeline=async\n";
    if (input_frame_dir.empty()) {
      std::cout << "input_mode=video\n";
      std::cout << "input_video=" << input_path.string() << "\n";
    } else {
      std::cout << "input_mode=frame_dir\n";
      std::cout << "input_frame_dir=" << input_frame_dir.string() << "\n";
    }
    std::cout << "warmup=" << warmup << "\n";
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
    std::cout << "  async_wait_ms=" << async_wait_ms
              << " avg=" << (processed > 0 ? async_wait_ms / static_cast<double>(processed) : 0.0)
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
    std::cout << "  decoder_ms=" << decoder_ms
              << " avg=" << (processed > 0 ? decoder_ms / static_cast<double>(processed) : 0.0)
              << "\n";
    std::cout << "  feature_copy_ms=" << feature_copy_ms
              << " avg=" << (processed > 0 ? feature_copy_ms / static_cast<double>(processed) : 0.0)
              << "\n";
    std::cout << "  output_push_ms=" << output_push_ms
              << " avg=" << (processed > 0 ? output_push_ms / static_cast<double>(processed) : 0.0)
              << "\n";
    std::cout << "  writer_ms=" << writer_ms
              << " avg=" << (processed > 0 ? writer_ms / static_cast<double>(processed) : 0.0)
              << "\n";
    if (!output_video_path.empty()) {
      std::cout << "output_video=" << output_video_path.string() << "\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "benchmark_acl_video failed: " << error.what() << "\n";
    return 1;
  }
}
