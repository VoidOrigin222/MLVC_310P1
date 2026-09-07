#include <mlvc/application/cli/decoder_app.h>
#include <mlvc/codec/execution_profile.h>
#include <mlvc/application/stream/mlvc_stream.h>
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

}  // namespace

namespace mlvc {

DecoderApplicationConfig LoadDecoderConfig(const std::filesystem::path& config_path) {
  const toml::table config = ReadTomlConfig(config_path);
  const std::string mode = GetString(config, "mode", "decode");
  Check(mode == "decode", "decoder config must use mode = \"decode\"");
  const std::string input = GetString(config, "input");
  const std::string output = GetString(config, "output");
  const std::string output_video = GetString(config, "output_video");
  const std::string output_format = GetString(config, "format", GetString(config, "output_format"));
  const std::string bitrate = GetString(config, "bitrate", "6500k");
  const std::string preset = GetString(config, "preset", "medium");
  const std::string execution_profile =
      GetString(config, "execution_profile", std::string(mlvc::codec::kCodecExecutionProfile));
  const int crf = GetInt(config, "crf", 23);
  const double fps = std::stod(GetString(config, "fps", "30"));
  const std::string manifest_path =
      GetString(config, "manifest", GetNestedString(config, "model", "manifest"));
  Check(!manifest_path.empty(), "config requires manifest or [model].manifest");
  Check(config["udp_port"].node() == nullptr && config["forward_host"].node() == nullptr &&
            config["forward_port"].node() == nullptr && config["forward_mode"].node() == nullptr &&
            config["forward_queue_capacity"].node() == nullptr &&
            config["forward_jpeg_quality"].node() == nullptr,
        "udp_port and forward_* are no longer supported; use input_transport_port and output_transport_*");
  const int udp_port = GetInt(config, "input_transport_port", 0);
  Check(udp_port >= 0 && udp_port <= 65535, "input_transport_port must be in [0, 65535]");
  const std::string forward_host = GetString(config, "output_transport_host");
  const int forward_port = GetInt(config, "output_transport_port", 0);
  const std::string forward_mode = GetString(config, "output_transport_mode", "jpeg");
  const int forward_queue_capacity = GetInt(config, "output_transport_queue_capacity", 3);
  const int forward_jpeg_quality = GetInt(config, "output_transport_jpeg_quality", 75);
  Check(forward_port >= 0 && forward_port <= 65535, "output_transport_port must be in [0, 65535]");
  Check(forward_port == 0 || !forward_host.empty(),
        "output_transport_host is required when output_transport_port is set");
  Check(forward_mode == "jpeg" || forward_mode == "jpeg_async" ||
            forward_mode == "dvpp_jpeg_async" ||
            forward_mode == "dvpp_jpeg_device_async" ||
            forward_mode == "raw_fp16_yuv444",
        "forward_mode must be jpeg, jpeg_async, dvpp_jpeg_async, "
        "dvpp_jpeg_device_async, or raw_fp16_yuv444");
  Check(forward_queue_capacity > 0, "forward_queue_capacity must be positive");
  Check(forward_jpeg_quality >= 1 && forward_jpeg_quality <= 100,
        "forward_jpeg_quality must be in [1, 100]");
  const mlvc::ModelManifest manifest = mlvc::ModelManifest::Load(manifest_path);
  DecoderApplicationConfig result;
  result.stream.manifest_path = manifest_path;
  result.stream.input_bitstream_path = input;
  result.stream.output_video_path = !output_video.empty() ? output_video : output;
  result.stream.output_format = output_format;
  result.stream.bitrate = bitrate;
  result.stream.preset = preset;
  result.stream.execution_profile = execution_profile;
  result.stream.fps = fps;
  result.stream.crf = crf;
  result.stream.device = GetInt(config, "device", 0);
  result.stream.frame_num = GetInt(config, "frame_num", -1);
  result.stream.drop_frame_index = GetInt(config, "drop_frame_index", -1);
  result.stream.forced_ltr_reference_frame = GetInt(config, "forced_ltr_reference_frame", -1);
  result.stream.forced_ltr_recovery_frame = GetInt(config, "forced_ltr_recovery_frame", -1);
  result.stream.profile_output_path = GetString(config, "profile_output");
  result.stream.udp_port = udp_port;
  result.stream.forward_host = forward_host;
  result.stream.forward_port = forward_port;
  result.stream.forward_mode = forward_mode;
  result.stream.forward_queue_capacity = forward_queue_capacity;
  result.stream.forward_jpeg_quality = forward_jpeg_quality;
  Check(mlvc::codec::IsMlvcManifest(manifest),
        "decoder manifest must contain MLVCEncoder and MLVCDecoder models");
  const int stream_workers = GetNestedInt(config, "pipeline", "stream_workers", 1);
  const int queue_capacity = GetNestedInt(config, "pipeline", "queue_capacity", 3);
  const int frame_buffer_slots = GetNestedInt(config, "pipeline", "frame_buffer_slots", 3);
  const int entropy_workers = GetNestedInt(config, "pipeline", "entropy_workers", 1);
  Check(config["pipeline"]["entropy_execution"].node() == nullptr,
        "pipeline.entropy_execution is no longer supported; use pipeline.entropy_workers");
  const int graph_packet_capacity =
      GetNestedInt(config, "pipeline", "graph_packet_capacity", 2);
  Check(stream_workers > 0, "pipeline.stream_workers must be positive");
  Check(queue_capacity > 0, "pipeline.queue_capacity must be positive");
  Check(frame_buffer_slots > 0, "pipeline.frame_buffer_slots must be positive");
  Check(entropy_workers > 0, "pipeline.entropy_workers must be positive");
  Check(graph_packet_capacity > 0, "pipeline.graph_packet_capacity must be positive");
  result.stream.pipeline.stream_workers = static_cast<std::size_t>(stream_workers);
  result.stream.pipeline.queue_capacity = static_cast<std::size_t>(queue_capacity);
  result.stream.pipeline.frame_buffer_slots = static_cast<std::size_t>(frame_buffer_slots);
  result.stream.pipeline.entropy_workers = static_cast<std::size_t>(entropy_workers);
  result.stream.pipeline.graph_packet_capacity = static_cast<std::size_t>(graph_packet_capacity);
  return result;
}

}  // namespace mlvc
