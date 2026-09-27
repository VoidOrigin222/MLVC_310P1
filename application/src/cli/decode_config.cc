#include <mlvc/application/cli/decoder_app.h>
#include <mlvc/application/stream/mlvc_stream.h>
#include <mlvc/codec/execution_profile.h>
#include <mlvc/io/video_io.h>
#include <mlvc/runtime/model_manifest.h>
#include <toml++/toml.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

#include "mlvc/core/status.h"

namespace {

void CheckKnownKeys(const toml::table& table, std::initializer_list<std::string_view> allowed,
                    const std::string& section) {
  for (const auto& [key, value] : table) {
    (void)value;
    const std::string name(key.str());
    if (std::find(allowed.begin(), allowed.end(), name) == allowed.end()) {
      throw mlvc::Error("unknown config key: " +
                        (section.empty() ? name : section + "." + name));
    }
  }
}

void ValidateConfigKeys(const toml::table& config) {
  CheckKnownKeys(config,
                 {"mode", "input", "output", "output_video", "format", "output_format",
                  "bitrate", "preset", "execution_profile", "crf", "fps", "manifest",
                  "udp_port", "forward_host", "forward_port", "forward_mode",
                  "forward_queue_capacity", "forward_jpeg_quality",
                  "output_transport_jpeg_quality", "output_transport_host",
                  "output_transport_port", "input_transport_port", "input_transport_mode",
                  "input_transport_payload_type",
                  "output_transport_mode", "output_transport_queue_capacity",
                  "output_transport_rtsp_url", "output_transport_rtsp_encoder",
                  "output_transport_rtsp_preset", "output_transport_rtsp_transport",
                  "output_transport_rtsp_crf", "device", "frame_num", "drop_frame_index",
                  "forced_ltr_reference_frame", "forced_ltr_recovery_frame", "profile_output",
                  "model", "pipeline"},
                 "");
  if (const toml::node* model = config["model"].node(); model != nullptr) {
    mlvc::Check(model->is_table(), "config section must be a table: model");
    CheckKnownKeys(*model->as_table(), {"manifest"}, "model");
  }
  if (const toml::node* pipeline = config["pipeline"].node(); pipeline != nullptr) {
    mlvc::Check(pipeline->is_table(), "config section must be a table: pipeline");
    CheckKnownKeys(*pipeline->as_table(), {"stream_workers", "queue_capacity", "frame_buffer_slots",
                                            "entropy_workers", "graph_packet_capacity",
                                            "entropy_execution"},
                   "pipeline");
  }
}

toml::table ReadTomlConfig(const std::filesystem::path& path) {
  try {
    return toml::parse_file(path.string());
  } catch (const toml::parse_error& error) {
    throw mlvc::Error("failed to parse config TOML: " + std::string(error.description()));
  }
}

std::string GetString(const toml::table& config, const std::string& key,
                      const std::string& fallback = "") {
  if (config[key].node() == nullptr) return fallback;
  const std::optional<std::string> value = config[key].value<std::string>();
  mlvc::Check(value.has_value(), "config key must be a string: " + key);
  return *value;
}

std::string GetNestedString(const toml::table& config, const std::string& table_name,
                            const std::string& key, const std::string& fallback = "") {
  const toml::node* table_node = config[table_name].node();
  if (table_node == nullptr) return fallback;
  mlvc::Check(table_node->is_table(), "config section must be a table: " + table_name);
  if (config[table_name][key].node() == nullptr) return fallback;
  const std::optional<std::string> value = config[table_name][key].value<std::string>();
  mlvc::Check(value.has_value(), "config key must be a string: " + table_name + "." + key);
  return *value;
}

int GetInt(const toml::table& config, const std::string& key, int fallback) {
  if (config[key].node() == nullptr) return fallback;
  const std::optional<int64_t> value = config[key].value<int64_t>();
  mlvc::Check(value.has_value(), "config key must be an integer: " + key);
  mlvc::Check(
      *value >= std::numeric_limits<int>::min() && *value <= std::numeric_limits<int>::max(),
      "config integer is out of int range: " + key);
  return static_cast<int>(*value);
}

int GetNestedInt(const toml::table& config, const std::string& table_name, const std::string& key,
                 int fallback) {
  const toml::node* table_node = config[table_name].node();
  if (table_node == nullptr) return fallback;
  mlvc::Check(table_node->is_table(), "config section must be a table: " + table_name);
  if (config[table_name][key].node() == nullptr) return fallback;
  const std::optional<int64_t> value = config[table_name][key].value<int64_t>();
  mlvc::Check(value.has_value(), "config key must be an integer: " + table_name + "." + key);
  mlvc::Check(
      *value >= std::numeric_limits<int>::min() && *value <= std::numeric_limits<int>::max(),
      "config integer is out of int range: " + table_name + "." + key);
  return static_cast<int>(*value);
}

double GetDouble(const toml::table& config, const std::string& key, double fallback) {
  if (config[key].node() == nullptr) return fallback;
  if (const std::optional<double> value = config[key].value<double>(); value.has_value()) {
    return *value;
  }
  if (const std::optional<int64_t> value = config[key].value<int64_t>(); value.has_value()) {
    return static_cast<double>(*value);
  }
  mlvc::Check(false, "config key must be a number: " + key);
  return fallback;
}

}  // namespace

namespace mlvc {

DecoderApplicationConfig LoadDecoderConfig(const std::filesystem::path& config_path) {
  const toml::table config = ReadTomlConfig(config_path);
  ValidateConfigKeys(config);
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
  Check(execution_profile == "pipeline-v1",
        "only pipeline-v1 execution profile is supported, got: " + execution_profile);
  const int crf = GetInt(config, "crf", 23);
  const double fps = GetDouble(config, "fps", 30.0);
  Check(std::isfinite(fps) && fps >= 0.1 && fps <= 1000.0,
        "fps must be finite and in [0.1, 1000]");
  Check(crf >= 0 && crf <= 51, "crf must be in [0, 51]");
  const std::string manifest_path =
      GetString(config, "manifest", GetNestedString(config, "model", "manifest"));
  Check(!manifest_path.empty(), "config requires manifest or [model].manifest");
  Check(config["udp_port"].node() == nullptr && config["forward_host"].node() == nullptr &&
            config["forward_port"].node() == nullptr && config["forward_mode"].node() == nullptr &&
            config["forward_queue_capacity"].node() == nullptr &&
            config["forward_jpeg_quality"].node() == nullptr &&
            config["output_transport_jpeg_quality"].node() == nullptr &&
            config["output_transport_host"].node() == nullptr &&
            config["output_transport_port"].node() == nullptr,
        "udp_port, forward_*, output_transport_host, and output_transport_port are no longer "
        "supported");
  const int udp_port = GetInt(config, "input_transport_port", 0);
  const int payload_type = GetInt(config, "input_transport_payload_type", 96);
  const std::string input_transport_mode = GetString(config, "input_transport_mode", "udp");
  Check(input_transport_mode == "udp" || input_transport_mode == "rtp",
        "input_transport_mode must be udp or rtp");
  Check(udp_port >= 0 && udp_port <= 65535, "input_transport_port must be in [0, 65535]");
  Check(payload_type >= 96 && payload_type <= 127,
        "input_transport_payload_type must be in the dynamic RTP range [96, 127]");
  const std::string output_transport_mode = GetString(config, "output_transport_mode", "none");
  const int output_transport_queue_capacity = GetInt(config, "output_transport_queue_capacity", 3);
  const std::string output_transport_rtsp_url = GetString(config, "output_transport_rtsp_url");
  const std::string output_transport_rtsp_encoder =
      GetString(config, "output_transport_rtsp_encoder", "dvpp");
  const std::string output_transport_rtsp_preset =
      GetString(config, "output_transport_rtsp_preset", "ultrafast");
  const std::string output_transport_rtsp_transport =
      GetString(config, "output_transport_rtsp_transport", "udp");
  const int output_transport_rtsp_crf = GetInt(config, "output_transport_rtsp_crf", 0);
  Check(output_transport_mode == "none" || output_transport_mode == "rtsp",
        "output_transport_mode must be none or rtsp");
  Check(output_transport_rtsp_encoder == "dvpp" || output_transport_rtsp_encoder == "libx264",
        "output_transport_rtsp_encoder must be dvpp or libx264");
  Check(output_transport_queue_capacity > 0, "output_transport_queue_capacity must be positive");
  Check(output_transport_rtsp_crf >= 0 && output_transport_rtsp_crf <= 51,
        "output_transport_rtsp_crf must be in [0, 51]");
  Check(output_transport_rtsp_transport == "tcp" || output_transport_rtsp_transport == "udp",
        "output_transport_rtsp_transport must be tcp or udp");
  if (output_transport_mode == "rtsp") {
    Check(!output_transport_rtsp_url.empty(),
          "output_transport_rtsp_url is required when output_transport_mode is rtsp");
    Check(!output_transport_rtsp_preset.empty(), "output_transport_rtsp_preset must not be empty");
  }
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
  Check(result.stream.device >= 0, "device must be non-negative");
  result.stream.frame_num = GetInt(config, "frame_num", -1);
  Check(result.stream.frame_num != 0 && result.stream.frame_num >= -1,
        "frame_num must be -1 or a positive value");
  result.stream.drop_frame_index = GetInt(config, "drop_frame_index", -1);
  Check(result.stream.drop_frame_index >= -1, "drop_frame_index must be non-negative or -1");
  result.stream.forced_ltr_reference_frame = GetInt(config, "forced_ltr_reference_frame", -1);
  result.stream.forced_ltr_recovery_frame = GetInt(config, "forced_ltr_recovery_frame", -1);
  Check((result.stream.forced_ltr_recovery_frame >= 0) ==
            (result.stream.forced_ltr_reference_frame >= 0),
        "decoder forced LTR recovery and reference frames must be configured together");
  result.stream.profile_output_path = GetString(config, "profile_output");
  result.stream.input_transport_port = udp_port;
  result.stream.input_transport_mode = input_transport_mode;
  result.stream.input_transport_payload_type = static_cast<uint8_t>(payload_type);
  result.stream.output_transport_mode = output_transport_mode;
  result.stream.output_transport_queue_capacity = output_transport_queue_capacity;
  result.stream.output_transport_rtsp_url = output_transport_rtsp_url;
  result.stream.output_transport_rtsp_encoder = output_transport_rtsp_encoder;
  result.stream.output_transport_rtsp_preset = output_transport_rtsp_preset;
  result.stream.output_transport_rtsp_transport = output_transport_rtsp_transport;
  result.stream.output_transport_rtsp_crf = output_transport_rtsp_crf;
  Check(mlvc::codec::IsMlvcManifest(manifest),
        "decoder manifest must contain MLVCEncoder and MLVCDecoder models");
  const int stream_workers = GetNestedInt(config, "pipeline", "stream_workers", 1);
  const int queue_capacity = GetNestedInt(config, "pipeline", "queue_capacity", 3);
  const int frame_buffer_slots = GetNestedInt(config, "pipeline", "frame_buffer_slots", 3);
  const int entropy_workers = GetNestedInt(config, "pipeline", "entropy_workers", 1);
  Check(config["pipeline"]["entropy_execution"].node() == nullptr,
        "pipeline.entropy_execution is no longer supported; use pipeline.entropy_workers");
  const int graph_packet_capacity = GetNestedInt(config, "pipeline", "graph_packet_capacity", 2);
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
