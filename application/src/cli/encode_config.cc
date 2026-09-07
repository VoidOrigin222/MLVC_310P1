#include <mlvc/application/cli/encoder_app.h>
#include <mlvc/codec/execution_profile.h>
#include <mlvc/application/stream/mlvc_stream.h>
#include <mlvc/application/stream/stream_encoder.h>
#include <mlvc/io/video_io.h>
#include <mlvc/runtime/model_manifest.h>
#include <toml++/toml.h>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>

#include "mlvc/core/status.h"

namespace {

toml::table ReadTomlConfig(const std::filesystem::path& path) {
  try {
    return toml::parse_file(path.string());
  } catch (const toml::parse_error& error) {
    throw mlvc::Error("failed to parse config TOML: " + std::string(error.description()));
  }
}

std::string GetString(const toml::table& config, const std::string& key,
                      const std::string& fallback = "") {
  return config[key].value_or(fallback);
}

std::string GetNestedString(const toml::table& config, const std::string& table_name,
                            const std::string& key, const std::string& fallback = "") {
  return config[table_name][key].value_or(fallback);
}

int GetInt(const toml::table& config, const std::string& key, int fallback) {
  const std::optional<int64_t> value = config[key].value<int64_t>();
  if (!value.has_value()) {
    return fallback;
  }
  mlvc::Check(
      *value >= std::numeric_limits<int>::min() && *value <= std::numeric_limits<int>::max(),
      "config integer is out of int range: " + key);
  return static_cast<int>(*value);
}

int GetNestedInt(const toml::table& config, const std::string& table_name,
                const std::string& key, int fallback) {
  const std::optional<int64_t> value = config[table_name][key].value<int64_t>();
  if (!value.has_value()) {
    return fallback;
  }
  mlvc::Check(
      *value >= std::numeric_limits<int>::min() && *value <= std::numeric_limits<int>::max(),
      "config integer is out of int range: " + table_name + "." + key);
  return static_cast<int>(*value);
}

double GetDouble(const toml::table& config, const std::string& key, double fallback) {
  const std::optional<double> value = config[key].value<double>();
  return value.value_or(fallback);
}

bool GetBool(const toml::table& config, const std::string& key, bool fallback) {
  return config[key].value_or(fallback);
}

}  // namespace

namespace mlvc {

EncoderApplicationConfig LoadEncoderConfig(const std::filesystem::path& config_path) {
  const toml::table config = ReadTomlConfig(config_path);
  const std::string mode = GetString(config, "mode", "encode");
  Check(mode == "encode", "encoder config must use mode = \"encode\"");
  const std::string input = GetString(config, "input");
  const std::string output = GetString(config, "output");
  const std::string input_frame_dir =
      GetString(config, "input_frame_dir", GetNestedString(config, "model", "input_frame_dir"));
  const std::string input_video =
      GetString(config, "input_video", GetNestedString(config, "model", "input_video"));
  const std::string execution_profile =
      GetString(config, "execution_profile", std::string(mlvc::codec::kCodecExecutionProfile));
  const std::string manifest_path =
      GetString(config, "manifest", GetNestedString(config, "model", "manifest"));

  Check(!manifest_path.empty(), "config requires manifest or [model].manifest");
  const mlvc::ModelManifest manifest = mlvc::ModelManifest::Load(manifest_path);
  Check(mlvc::codec::IsMlvcManifest(manifest),
        "encoder manifest must contain MLVCEncoder and MLVCDecoder models");
  const int qp = GetInt(config, "qp", 18);
  const int gop = GetInt(config, "gop", 96);
  const int reset_interval = GetInt(config, "reset_interval", 32);
  codec::EncodeStreamOptions options;
  options.manifest_path = manifest_path;
  options.device = GetInt(config, "device", 0);
  options.frame_num = GetInt(config, "frame_num", -1);
  options.fps = GetDouble(config, "fps", 30.0);
  Check(options.fps > 0.0, "fps must be positive");
  options.profile_warmup_frames = GetInt(config, "profile_warmup_frames", 0);
  options.profile_output_path = GetString(config, "profile_output");
  options.enable_stage_fusion = GetBool(config, "enable_stage_fusion", false);
  options.execution_profile = execution_profile;
  options.gop = gop;
  options.reset_interval = reset_interval;
  options.qp = qp;
  Check(!options.manifest_path.empty(), "config requires manifest or [model].manifest");
  Check(options.qp >= 0 && options.qp <= 63, "qp must be in [0, 63]");
  Check(options.gop > 0, "gop must be positive");
  Check(options.reset_interval > 0, "reset_interval must be positive");

  const std::string frame_input = input_frame_dir.empty() ? input : input_frame_dir;
  const std::string video_input =
      input_video.empty() && mlvc::io::IsVideoPath(input) ? input : input_video;
  if (!video_input.empty()) {
    options.input_video_path = video_input;
  } else {
    options.input_frame_dir = frame_input;
  }
  options.output_bitstream_path = output;
  options.ltr_start_idx = GetInt(config, "ltr_start_idx", 8);
  options.ltr_period = GetInt(config, "ltr_period", 64);
  options.ltr_qp_shift = GetInt(config, "ltr_qp_shift", 8);
  options.target_bitrate_bps = GetDouble(config, "target_bitrate_bps", 0.0);
  options.min_qp = GetInt(config, "min_qp", 0);
  options.max_qp = GetInt(config, "max_qp", 63);
  options.forced_ltr_recovery_frame = GetInt(config, "forced_ltr_recovery_frame", -1);
  options.forced_ltr_reference_frame = GetInt(config, "forced_ltr_reference_frame", -1);
  Check(config["udp_host"].node() == nullptr && config["udp_port"].node() == nullptr,
        "udp_host and udp_port are no longer supported; use output_transport_host and output_transport_port");
  options.udp_host = GetString(config, "output_transport_host");
  options.udp_port = GetInt(config, "output_transport_port", 0);
  Check(options.udp_port >= 0 && options.udp_port <= 65535, "output_transport_port must be in [0, 65535]");
  Check(options.udp_port == 0 || !options.udp_host.empty(),
        "output_transport_host is required when output_transport_port is set");
  if (options.udp_port > 0) {
    options.output_bitstream_path.clear();
  }
  const int stream_workers = GetNestedInt(config, "pipeline", "stream_workers", 1);
  const int queue_capacity = GetNestedInt(config, "pipeline", "queue_capacity", 3);
  const int frame_buffer_slots = GetNestedInt(config, "pipeline", "frame_buffer_slots", 3);
  const int entropy_workers = GetNestedInt(config, "pipeline", "entropy_workers", 1);
  const int graph_packet_capacity =
      GetNestedInt(config, "pipeline", "graph_packet_capacity", 2);
  Check(stream_workers > 0, "pipeline.stream_workers must be positive");
  Check(queue_capacity > 0, "pipeline.queue_capacity must be positive");
  Check(frame_buffer_slots > 0, "pipeline.frame_buffer_slots must be positive");
  Check(entropy_workers > 0, "pipeline.entropy_workers must be positive");
  Check(graph_packet_capacity > 0, "pipeline.graph_packet_capacity must be positive");
  options.pipeline.stream_workers = static_cast<std::size_t>(stream_workers);
  options.pipeline.queue_capacity = static_cast<std::size_t>(queue_capacity);
  options.pipeline.frame_buffer_slots = static_cast<std::size_t>(frame_buffer_slots);
  options.pipeline.entropy_workers = static_cast<std::size_t>(entropy_workers);
  options.pipeline.graph_packet_capacity = static_cast<std::size_t>(graph_packet_capacity);
  return EncoderApplicationConfig{std::move(options)};
}

}  // namespace mlvc
