#include <mlvc/application/input/frame_input_queue.h>
#include <mlvc/application/cli/runtime_environment.h>
#include <mlvc/codec/mlvc_entropy.h>
#include <mlvc/codec/tensor_data.h>
#include <mlvc/codec/tensor_utils.h>
#include <mlvc/core/status.h>
#include <mlvc/entropy/mlvc_official_entropy.h>
#include <mlvc/entropy/sidecar.h>
#include <mlvc/io/mlvc_bitstream.h>
#include <mlvc/io/video_io.h>
#include <mlvc/runtime/stage_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::string_view kDefaultManifest = "../mlvc1080p/manifest.json";
constexpr std::array<int, 8> kFrameIndexMap = {0, 1, 0, 2, 0, 2, 0, 2};

struct Options {
  std::filesystem::path manifest_path = std::filesystem::path(kDefaultManifest);
  std::filesystem::path input_frame_dir;
  std::filesystem::path output_bitstream_path;
  std::filesystem::path output_frame_dir;
  int device = 0;
  int frames = -1;
  int q_index = 8;
  int gop = 96;
  int reset_interval = 32;
  int ltr_start_idx = 0;
  int ltr_period = 0;
  double fallback_fps = 30.0;
  std::filesystem::path debug_dump_dir;
  int debug_dump_frames = 0;
};

struct SourceInfo {
  int width = 0;
  int height = 0;
  double fps = 30.0;
};

struct Timings {
  double input_wait_ms = 0.0;
  double reference_prepare_ms = 0.0;
  double q_prepare_ms = 0.0;
  double encoder_om_ms = 0.0;
  double encoder_feature_copy_ms = 0.0;
  double encode_symbol_convert_ms = 0.0;
  double encode_scale_decode_ms = 0.0;
  double rans_encode_ms = 0.0;
  double bitstream_write_ms = 0.0;
  double rans_decode_z_ms = 0.0;
  double decode_z_restore_ms = 0.0;
  double decode_scale_decode_ms = 0.0;
  double rans_decode_y_ms = 0.0;
  double decode_y_restore_ms = 0.0;
  double decoder_om_ms = 0.0;
  double decoder_feature_copy_ms = 0.0;
  double output_write_ms = 0.0;
};

class Timer {
 public:
  explicit Timer(double* accumulator) : accumulator_(accumulator), begin_(Clock::now()) {}
  ~Timer() {
    *accumulator_ += std::chrono::duration<double, std::milli>(Clock::now() - begin_).count();
  }

 private:
  double* accumulator_;
  Clock::time_point begin_;
};

void PrintUsage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " --input-frame-dir <dir> [--manifest <manifest.json>] [--device 0]"
               " [--frames -1] [--q-index 8] [--gop 96] [--reset-interval 32]"
               " [--ltr-start-idx 0] [--ltr-period 0] [--fps 30]"
               " [--output-bitstream <official.mlvc>] [--output-frame-dir <png-dir>]"
               " [--debug-dump-dir <dir>] [--debug-dump-frames <count>]\n";
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const char* name) -> std::string {
      mlvc::Check(i + 1 < argc, std::string("missing value for ") + name);
      return argv[++i];
    };
    if (arg == "--manifest") {
      options.manifest_path = require_value("--manifest");
    } else if (arg == "--input-frame-dir") {
      options.input_frame_dir = require_value("--input-frame-dir");
    } else if (arg == "--output-bitstream") {
      options.output_bitstream_path = require_value("--output-bitstream");
    } else if (arg == "--output-frame-dir") {
      options.output_frame_dir = require_value("--output-frame-dir");
    } else if (arg == "--device") {
      options.device = std::stoi(require_value("--device"));
    } else if (arg == "--frames") {
      options.frames = std::stoi(require_value("--frames"));
    } else if (arg == "--q-index") {
      options.q_index = std::stoi(require_value("--q-index"));
    } else if (arg == "--gop") {
      options.gop = std::stoi(require_value("--gop"));
    } else if (arg == "--reset-interval") {
      options.reset_interval = std::stoi(require_value("--reset-interval"));
    } else if (arg == "--ltr-start-idx") {
      options.ltr_start_idx = std::stoi(require_value("--ltr-start-idx"));
    } else if (arg == "--ltr-period") {
      options.ltr_period = std::stoi(require_value("--ltr-period"));
    } else if (arg == "--fps") {
      options.fallback_fps = std::stod(require_value("--fps"));
    } else if (arg == "--debug-dump-dir") {
      options.debug_dump_dir = require_value("--debug-dump-dir");
    } else if (arg == "--debug-dump-frames") {
      options.debug_dump_frames = std::stoi(require_value("--debug-dump-frames"));
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw mlvc::Error("unknown argument: " + arg);
    }
  }

  mlvc::Check(!options.input_frame_dir.empty(), "--input-frame-dir is required");
  mlvc::Check(std::filesystem::is_directory(options.input_frame_dir),
              "input frame directory does not exist: " + options.input_frame_dir.string());
  mlvc::Check(options.frames != 0 && options.frames >= -1, "--frames must be -1 or positive");
  mlvc::Check(options.q_index >= 0 && options.q_index <= 63, "--q-index must be in [0, 63]");
  mlvc::Check(options.gop > 0, "--gop must be positive");
  mlvc::Check(options.reset_interval > 0, "--reset-interval must be positive");
  mlvc::Check(options.ltr_start_idx >= 0, "--ltr-start-idx must be non-negative");
  mlvc::Check(options.ltr_period >= 0, "--ltr-period must be non-negative");
  mlvc::Check(options.fallback_fps > 0.0, "--fps must be positive");
  mlvc::Check(options.debug_dump_frames >= 0, "--debug-dump-frames must be non-negative");
  mlvc::Check(options.debug_dump_frames == 0 || !options.debug_dump_dir.empty(),
              "--debug-dump-frames requires --debug-dump-dir");
  return options;
}

SourceInfo ReadSourceInfo(const std::filesystem::path& input_frame_dir, double fallback_fps) {
  SourceInfo info;
  info.fps = fallback_fps;
  std::ifstream input(input_frame_dir / "source_info.txt");
  mlvc::Check(input.good(), "input frame directory requires source_info.txt");
  std::string line;
  while (std::getline(input, line)) {
    const std::size_t separator = line.find('=');
    if (separator == std::string::npos) {
      continue;
    }
    const std::string key = line.substr(0, separator);
    const std::string value = line.substr(separator + 1);
    if (key == "width") {
      info.width = std::stoi(value);
    } else if (key == "height") {
      info.height = std::stoi(value);
    } else if (key == "fps") {
      info.fps = std::stod(value);
    }
  }
  mlvc::Check(info.width > 0 && info.height > 0 && info.fps > 0.0, "invalid source_info.txt");
  return info;
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
  mlvc::Check(spec.dtype == mlvc::DataType::kInt32 && spec.shape == std::vector<int64_t>{1},
              "q-index tensor must be int32[1]");
  std::memcpy(tensor.bytes.data(), &value, sizeof(value));
  return tensor;
}

void SetInt32TensorValue(mlvc::codec::TensorData* tensor, int32_t value) {
  std::memcpy(tensor->bytes.data(), &value, sizeof(value));
}

void ResetTensor(mlvc::codec::TensorData* tensor) {
  std::fill(tensor->bytes.begin(), tensor->bytes.end(), 0);
}

bool MarkAsLtr(int local_frame_index, int ltr_start_idx, int ltr_period) {
  return ltr_period > 0 &&
         (local_frame_index == ltr_start_idx ||
          (local_frame_index > ltr_start_idx && local_frame_index % ltr_period == 0));
}

double SumTimedModules(const Timings& timings) {
  return timings.input_wait_ms + timings.reference_prepare_ms + timings.q_prepare_ms +
         timings.encoder_om_ms + timings.encoder_feature_copy_ms +
         timings.encode_symbol_convert_ms + timings.encode_scale_decode_ms +
         timings.rans_encode_ms + timings.bitstream_write_ms + timings.rans_decode_z_ms +
         timings.decode_z_restore_ms + timings.decode_scale_decode_ms + timings.rans_decode_y_ms +
         timings.decode_y_restore_ms + timings.decoder_om_ms + timings.decoder_feature_copy_ms +
         timings.output_write_ms;
}

void PrintTiming(std::string_view name, double total_ms, int frames) {
  const double average_ms = frames > 0 ? total_ms / static_cast<double>(frames) : 0.0;
  std::cout << "  " << name << "_ms=" << total_ms << " avg=" << average_ms << "\n";
}

void DumpTensor(const std::filesystem::path& directory, std::string_view name,
                const mlvc::codec::TensorData& tensor) {
  std::filesystem::create_directories(directory);
  std::ofstream output(directory / (std::string(name) + ".bin"), std::ios::binary);
  mlvc::Check(output.good(), "failed to create debug tensor dump: " + std::string(name));
  output.write(reinterpret_cast<const char*>(tensor.bytes.data()),
               static_cast<std::streamsize>(tensor.bytes.size()));
  mlvc::Check(output.good(), "failed to write debug tensor dump: " + std::string(name));
}

std::filesystem::path DebugFrameDirectory(const Options& options, int frame_index) {
  std::ostringstream name;
  name << "frame_" << std::setfill('0') << std::setw(4) << frame_index;
  return options.debug_dump_dir / name.str();
}

int Run(const Options& options) {
  mlvc::StageRuntime runtime(options.device);
  mlvc::ModelManifest manifest = mlvc::ModelManifest::Load(options.manifest_path);
  const std::filesystem::path model_directory = manifest.directory();
  const std::filesystem::path sidecar_path = manifest.sidecar().file.is_absolute()
                                                 ? manifest.sidecar().file
                                                 : manifest.directory() / manifest.sidecar().file;
  mlvc::StageModelSet models(&runtime, std::move(manifest));
  mlvc::StageModel& encoder = models.GetStage(std::string_view("MLVCEncoder"));
  mlvc::StageModel& decoder = models.GetStage(std::string_view("MLVCDecoder"));
  const mlvc::RuntimeSidecar sidecar = mlvc::RuntimeSidecar::Load(sidecar_path);

  const mlvc::TensorSpec& frame_spec = FindTensorSpec(encoder.record(), "x");
  const mlvc::TensorSpec& encoder_ref_spec = FindTensorSpec(encoder.record(), "ref_feature");
  const mlvc::TensorSpec& q_spec = FindTensorSpec(encoder.record(), "q_index_shifted");
  const mlvc::TensorSpec& encoder_feature_spec = FindTensorSpec(encoder.record(), "feature");
  const mlvc::TensorSpec& z_spec = FindTensorSpec(encoder.record(), "z_raw");
  const mlvc::TensorSpec& y0_spec = FindTensorSpec(encoder.record(), "y_raw_0");
  const mlvc::TensorSpec& y1_spec = FindTensorSpec(encoder.record(), "y_raw_1");
  const mlvc::TensorSpec& decoder_ref_spec = FindTensorSpec(decoder.record(), "ref_feature");
  const mlvc::TensorSpec& x_hat_spec = FindTensorSpec(decoder.record(), "x_hat");
  const mlvc::TensorSpec& decoder_feature_spec = FindTensorSpec(decoder.record(), "feature");
  mlvc::Check(encoder_ref_spec.shape == decoder_ref_spec.shape,
              "encoder and decoder reference feature shapes differ");

  const SourceInfo source_info = ReadSourceInfo(options.input_frame_dir, options.fallback_fps);
  mlvc::Check(
      source_info.width <= frame_spec.shape.at(3) && source_info.height <= frame_spec.shape.at(2),
      "source dimensions exceed model dimensions");

  const int frames_to_attempt =
      options.frames > 0 ? options.frames : std::numeric_limits<int>::max();
  mlvc::CodecGraphExecutor graph_executor(2);
  mlvc::app::FrameInputArena input_arena(frame_spec, 3);
  mlvc::app::AsyncFrameInputQueue input_queue(
      frames_to_attempt, options.frames, {}, options.input_frame_dir, frame_spec, source_info.width,
      source_info.height, &input_arena, nullptr, &graph_executor);

  mlvc::MlvcOfficialEntropyEncoder entropy_encoder(model_directory);
  mlvc::MlvcOfficialEntropyDecoder entropy_decoder(model_directory);
  std::optional<mlvc::io::OfficialMlvcBitstreamWriter> bitstream_writer;
  if (!options.output_bitstream_path.empty()) {
    bitstream_writer.emplace(options.output_bitstream_path);
  }
  std::optional<mlvc::io::DecodedVideoWriter> output_writer;
  if (!options.output_frame_dir.empty()) {
    output_writer.emplace(options.output_frame_dir, "png", source_info.fps, source_info.width,
                          source_info.height, 23, "", "medium");
  }

  mlvc::codec::TensorData encoder_ref = mlvc::codec::MakeTensorLike(encoder_ref_spec);
  mlvc::codec::TensorData decoder_ref = mlvc::codec::MakeTensorLike(decoder_ref_spec);
  mlvc::codec::TensorData encoder_ltr = mlvc::codec::MakeTensorLike(encoder_ref_spec);
  mlvc::codec::TensorData decoder_ltr = mlvc::codec::MakeTensorLike(decoder_ref_spec);
  mlvc::codec::TensorData encoder_feature = mlvc::codec::MakeTensorLike(encoder_feature_spec);
  mlvc::codec::TensorData encoder_z = mlvc::codec::MakeTensorLike(z_spec);
  mlvc::codec::TensorData encoder_y0 = mlvc::codec::MakeTensorLike(y0_spec);
  mlvc::codec::TensorData encoder_y1 = mlvc::codec::MakeTensorLike(y1_spec);
  mlvc::codec::TensorData decoder_z = mlvc::codec::MakeTensorLike(z_spec);
  mlvc::codec::TensorData decoder_y0 = mlvc::codec::MakeTensorLike(y0_spec);
  mlvc::codec::TensorData decoder_y1 = mlvc::codec::MakeTensorLike(y1_spec);
  mlvc::codec::TensorData x_hat = mlvc::codec::MakeTensorLike(x_hat_spec);
  mlvc::codec::TensorData decoder_feature = mlvc::codec::MakeTensorLike(decoder_feature_spec);
  mlvc::codec::TensorData q_tensor = MakeInt32Tensor(q_spec, options.q_index);

  const int y_channels = static_cast<int>(y0_spec.shape.at(1) + y1_spec.shape.at(1));
  const int y_height = static_cast<int>(y0_spec.shape.at(2));
  const int y_width = static_cast<int>(y0_spec.shape.at(3));
  const int z_channels = static_cast<int>(z_spec.shape.at(1));
  const int z_height = static_cast<int>(z_spec.shape.at(2));
  const int z_width = static_cast<int>(z_spec.shape.at(3));

  std::vector<int8_t> encoder_z_symbols;
  std::vector<int8_t> encoder_y0_symbols;
  std::vector<int8_t> encoder_y1_symbols;
  std::vector<int8_t> decoder_z_symbols;
  std::vector<int8_t> decoder_y0_symbols;
  std::vector<int8_t> decoder_y1_symbols;
  std::vector<uint8_t> encoder_scales0;
  std::vector<uint8_t> encoder_scales1;
  std::vector<uint8_t> decoder_scales0;
  std::vector<uint8_t> decoder_scales1;
  std::vector<uint8_t> payload;
  bool has_encoder_ltr = false;
  bool has_decoder_ltr = false;
  Timings timings;
  uint64_t payload_bytes = 0;
  int i_frames = 0;
  int p_frames = 0;
  int ltr_frames = 0;
  int processed = 0;

  const auto benchmark_begin = Clock::now();
  for (;;) {
    mlvc::app::InputFrame input_frame;
    {
      Timer timer(&timings.input_wait_ms);
      input_frame = input_queue.Pop();
    }
    if (input_frame.eof) {
      break;
    }
    mlvc::Check(input_frame.frame != nullptr, "input queue returned an empty frame");
    mlvc::Check(input_frame.frame_index == processed, "input frame order mismatch");
    const int frame_index = input_frame.frame_index;
    const bool dump_frame =
        options.debug_dump_frames > 0 && frame_index < options.debug_dump_frames;
    const std::filesystem::path debug_frame_dir = DebugFrameDirectory(options, frame_index);
    const int local_frame_index = frame_index % options.gop;
    const bool is_i_frame = frame_index == 0 || local_frame_index == 0;
    const bool feature_reset = (local_frame_index + 1) % options.reset_interval == 0;
    const bool mark_as_ltr =
        MarkAsLtr(local_frame_index, options.ltr_start_idx, options.ltr_period);
    const bool use_ltr_recovery = !is_i_frame && mark_as_ltr && has_encoder_ltr && has_decoder_ltr;

    {
      Timer timer(&timings.reference_prepare_ms);
      if (is_i_frame) {
        ResetTensor(&encoder_ref);
        ResetTensor(&decoder_ref);
        ResetTensor(&encoder_ltr);
        ResetTensor(&decoder_ltr);
        has_encoder_ltr = false;
        has_decoder_ltr = false;
      } else if (feature_reset) {
        ResetTensor(&encoder_ref);
        ResetTensor(&decoder_ref);
      } else if (use_ltr_recovery) {
        mlvc::codec::CloneTensorInto(encoder_ltr, &encoder_ref);
        mlvc::codec::CloneTensorInto(decoder_ltr, &decoder_ref);
      }
    }

    {
      Timer timer(&timings.q_prepare_ms);
      const int frame_adaptation_index = kFrameIndexMap[(local_frame_index + 1) % 8];
      const int shifted_q_index = sidecar.ShiftedQp(options.q_index, frame_adaptation_index);
      SetInt32TensorValue(&q_tensor, shifted_q_index);
    }

    const std::array<mlvc::NamedTensorView, 3> encoder_inputs = {
        mlvc::NamedTensorView{"x", input_frame.frame->View()},
        mlvc::NamedTensorView{"ref_feature", encoder_ref.View()},
        mlvc::NamedTensorView{"q_index_shifted", q_tensor.View()},
    };
    const std::array<mlvc::NamedTensorView, 4> encoder_outputs = {
        mlvc::NamedTensorView{"feature", encoder_feature.View()},
        mlvc::NamedTensorView{"z_raw", encoder_z.View()},
        mlvc::NamedTensorView{"y_raw_0", encoder_y0.View()},
        mlvc::NamedTensorView{"y_raw_1", encoder_y1.View()},
    };
    if (dump_frame) {
      DumpTensor(debug_frame_dir, "x_fp16", *input_frame.frame);
      DumpTensor(debug_frame_dir, "encoder_ref_fp16", encoder_ref);
      DumpTensor(debug_frame_dir, "q_index_shifted_i32", q_tensor);
    }
    {
      Timer timer(&timings.encoder_om_ms);
      encoder.RunNamed(encoder_inputs.data(), encoder_inputs.size(), encoder_outputs.data(),
                       encoder_outputs.size());
    }
    if (dump_frame) {
      DumpTensor(debug_frame_dir, "encoder_feature_fp16", encoder_feature);
      DumpTensor(debug_frame_dir, "encoder_z_raw_fp16", encoder_z);
      DumpTensor(debug_frame_dir, "encoder_y_raw_0_fp16", encoder_y0);
      DumpTensor(debug_frame_dir, "encoder_y_raw_1_fp16", encoder_y1);
    }
    {
      Timer timer(&timings.encoder_feature_copy_ms);
      mlvc::codec::CloneTensorInto(encoder_feature, &encoder_ref);
      if (mark_as_ltr) {
        mlvc::codec::CloneTensorInto(encoder_feature, &encoder_ltr);
        has_encoder_ltr = true;
      }
    }

    {
      Timer timer(&timings.encode_symbol_convert_ms);
      mlvc::codec::TensorToInt8Symbols(encoder_z, &encoder_z_symbols);
      mlvc::codec::TensorToInt8Symbols(encoder_y0, &encoder_y0_symbols);
      mlvc::codec::TensorToInt8Symbols(encoder_y1, &encoder_y1_symbols);
    }
    {
      Timer timer(&timings.encode_scale_decode_ms);
      mlvc::codec::BuildMlvcScaleIndexesFromZRaw(encoder_z, y_channels, y_height, y_width,
                                                 mlvc::codec::kMlvcChannelRepeat, &encoder_scales0,
                                                 &encoder_scales1);
    }
    {
      Timer timer(&timings.rans_encode_ms);
      entropy_encoder.Encode(encoder_z_symbols, encoder_y0_symbols, encoder_y1_symbols,
                             encoder_scales0, encoder_scales1, options.q_index, z_channels,
                             z_height, z_width, &payload);
    }
    payload_bytes += payload.size();
    if (dump_frame) {
      std::ofstream output(debug_frame_dir / "rans_payload.bin", std::ios::binary);
      mlvc::Check(output.good(), "failed to create rANS debug payload");
      output.write(reinterpret_cast<const char*>(payload.data()),
                   static_cast<std::streamsize>(payload.size()));
      mlvc::Check(output.good(), "failed to write rANS debug payload");
    }
    if (bitstream_writer.has_value()) {
      Timer timer(&timings.bitstream_write_ms);
      bitstream_writer->WriteFrame(options.q_index, payload);
    }

    {
      Timer timer(&timings.rans_decode_z_ms);
      entropy_decoder.SetStream(payload);
      decoder_z_symbols = entropy_decoder.DecodeZ(options.q_index, z_channels, z_height, z_width);
    }
    {
      Timer timer(&timings.decode_z_restore_ms);
      mlvc::codec::FloatToFp16TensorFromInt8(decoder_z_symbols, decoder_z.shape, &decoder_z);
    }
    {
      Timer timer(&timings.decode_scale_decode_ms);
      mlvc::codec::BuildMlvcScaleIndexesFromZRaw(decoder_z, y_channels, y_height, y_width,
                                                 mlvc::codec::kMlvcChannelRepeat, &decoder_scales0,
                                                 &decoder_scales1);
    }
    {
      Timer timer(&timings.rans_decode_y_ms);
      decoder_y0_symbols = entropy_decoder.DecodeY(decoder_scales0, false);
      decoder_y1_symbols = entropy_decoder.DecodeY(decoder_scales1, true);
    }
    {
      Timer timer(&timings.decode_y_restore_ms);
      mlvc::codec::FloatToFp16TensorFromInt8(decoder_y0_symbols, decoder_y0.shape, &decoder_y0);
      mlvc::codec::FloatToFp16TensorFromInt8(decoder_y1_symbols, decoder_y1.shape, &decoder_y1);
    }
    if (dump_frame) {
      DumpTensor(debug_frame_dir, "decoder_z_raw_fp16", decoder_z);
      DumpTensor(debug_frame_dir, "decoder_y_raw_0_fp16", decoder_y0);
      DumpTensor(debug_frame_dir, "decoder_y_raw_1_fp16", decoder_y1);
    }

    const std::array<mlvc::NamedTensorView, 5> decoder_inputs = {
        mlvc::NamedTensorView{"z_raw", decoder_z.View()},
        mlvc::NamedTensorView{"y_raw_0", decoder_y0.View()},
        mlvc::NamedTensorView{"y_raw_1", decoder_y1.View()},
        mlvc::NamedTensorView{"ref_feature", decoder_ref.View()},
        mlvc::NamedTensorView{"q_index_shifted", q_tensor.View()},
    };
    const std::array<mlvc::NamedTensorView, 2> decoder_outputs = {
        mlvc::NamedTensorView{"x_hat", x_hat.View()},
        mlvc::NamedTensorView{"feature", decoder_feature.View()},
    };
    {
      Timer timer(&timings.decoder_om_ms);
      decoder.RunNamed(decoder_inputs.data(), decoder_inputs.size(), decoder_outputs.data(),
                       decoder_outputs.size());
    }
    if (dump_frame) {
      DumpTensor(debug_frame_dir, "x_hat_fp16", x_hat);
      DumpTensor(debug_frame_dir, "decoder_feature_fp16", decoder_feature);
    }
    {
      Timer timer(&timings.decoder_feature_copy_ms);
      mlvc::codec::CloneTensorInto(decoder_feature, &decoder_ref);
      if (mark_as_ltr) {
        mlvc::codec::CloneTensorInto(decoder_feature, &decoder_ltr);
        has_decoder_ltr = true;
      }
    }
    if (output_writer.has_value()) {
      Timer timer(&timings.output_write_ms);
      output_writer->WriteTensorFrame(x_hat, frame_index);
    }

    if (is_i_frame) {
      ++i_frames;
    } else if (use_ltr_recovery) {
      ++ltr_frames;
    } else {
      ++p_frames;
    }
    input_queue.Release(&input_frame);
    ++processed;
    if (options.frames > 0 && processed >= options.frames) {
      break;
    }
  }

  if (bitstream_writer.has_value()) {
    bitstream_writer->Close();
  }
  if (output_writer.has_value()) {
    output_writer->Close();
  }
  const double total_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - benchmark_begin).count();
  const double duration_seconds =
      source_info.fps > 0.0 ? static_cast<double>(processed) / source_info.fps : 0.0;
  const uint64_t official_stream_bytes =
      payload_bytes + static_cast<uint64_t>(processed) * mlvc::io::kOfficialMlvcFrameOverheadBytes;
  const double payload_rate_kib_per_s =
      duration_seconds > 0.0 ? static_cast<double>(payload_bytes) / 1024.0 / duration_seconds : 0.0;
  const double stream_rate_kib_per_s =
      duration_seconds > 0.0
          ? static_cast<double>(official_stream_bytes) / 1024.0 / duration_seconds
          : 0.0;
  const double timed_modules_ms = SumTimedModules(timings);
  const mlvc::app::AsyncFrameInputQueue::Stats input_stats = input_queue.stats();

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "mode=mlvc_official_single_process_benchmark\n";
  std::cout << "flow=encoder_om->rans_encode->rans_decode->decoder_om\n";
  std::cout << "input_mode=preprocessed_fp16_yuv444\n";
  std::cout << "input_frame_dir=" << options.input_frame_dir.string() << "\n";
  std::cout << "frames=" << processed << "\n";
  std::cout << "q_index=" << options.q_index << "\n";
  std::cout << "gop=" << options.gop << "\n";
  std::cout << "reset_interval=" << options.reset_interval << "\n";
  std::cout << "ltr_start_idx=" << options.ltr_start_idx << "\n";
  std::cout << "ltr_period=" << options.ltr_period << "\n";
  std::cout << "i_frames=" << i_frames << "\n";
  std::cout << "p_frames=" << p_frames << "\n";
  std::cout << "ltr_recovery_frames=" << ltr_frames << "\n";
  std::cout << "payload_bytes=" << payload_bytes << "\n";
  std::cout << "official_stream_bytes=" << official_stream_bytes << "\n";
  std::cout << "payload_rate_kB_per_s=" << payload_rate_kib_per_s << "\n";
  std::cout << "official_stream_rate_kB_per_s=" << stream_rate_kib_per_s << "\n";
  std::cout << "total_ms=" << total_ms << "\n";
  std::cout << "latency_ms_per_frame="
            << (processed > 0 ? total_ms / static_cast<double>(processed) : 0.0) << "\n";
  std::cout << "throughput_fps="
            << (total_ms > 0.0 ? 1000.0 * static_cast<double>(processed) / total_ms : 0.0) << "\n";
  std::cout << "timed_modules_ms=" << timed_modules_ms << "\n";
  std::cout << "unattributed_ms=" << std::max(0.0, total_ms - timed_modules_ms) << "\n";
  std::cout << "breakdown:\n";
  PrintTiming("input_wait", timings.input_wait_ms, processed);
  PrintTiming("reference_prepare", timings.reference_prepare_ms, processed);
  PrintTiming("q_prepare", timings.q_prepare_ms, processed);
  PrintTiming("encoder_om", timings.encoder_om_ms, processed);
  PrintTiming("encoder_feature_copy", timings.encoder_feature_copy_ms, processed);
  PrintTiming("encode_symbol_convert", timings.encode_symbol_convert_ms, processed);
  PrintTiming("encode_scale_decode", timings.encode_scale_decode_ms, processed);
  PrintTiming("rans_encode", timings.rans_encode_ms, processed);
  PrintTiming("bitstream_write", timings.bitstream_write_ms, processed);
  PrintTiming("rans_decode_z", timings.rans_decode_z_ms, processed);
  PrintTiming("decode_z_restore", timings.decode_z_restore_ms, processed);
  PrintTiming("decode_scale_decode", timings.decode_scale_decode_ms, processed);
  PrintTiming("rans_decode_y", timings.rans_decode_y_ms, processed);
  PrintTiming("decode_y_restore", timings.decode_y_restore_ms, processed);
  PrintTiming("decoder_om", timings.decoder_om_ms, processed);
  PrintTiming("decoder_feature_copy", timings.decoder_feature_copy_ms, processed);
  PrintTiming("output_write", timings.output_write_ms, processed);
  std::cout << "input_prepare_background_ms=" << input_stats.prepare_ms << "\n";
  if (!options.output_bitstream_path.empty()) {
    std::cout << "output_bitstream=" << options.output_bitstream_path.string() << "\n";
    std::cout << "output_bitstream_file_bytes="
              << std::filesystem::file_size(options.output_bitstream_path) << "\n";
  }
  if (!options.output_frame_dir.empty()) {
    std::cout << "output_frame_dir=" << options.output_frame_dir.string() << "\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    mlvc::PrepareAscendRuntimeEnvironment(argv);
    const Options options = ParseOptions(argc, argv);
    return Run(options);
  } catch (const std::exception& error) {
    std::cerr << "benchmark_mlvc_official_loop failed: " << error.what() << "\n";
    PrintUsage(argv[0]);
    return 1;
  }
}
