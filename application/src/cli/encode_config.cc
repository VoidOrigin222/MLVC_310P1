#include <mlvc/application/cli/encoder_app.h>
#include <mlvc/application/stream/mlvc_stream.h>
#include <mlvc/application/stream/stream_encoder.h>
#include <mlvc/codec/execution_profile.h>
#include <mlvc/codec/mlvc_entropy.h>
#include <mlvc/io/video_io.h>
#include <mlvc/runtime/model_manifest.h>
#include <toml++/toml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
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
                 {"mode", "input", "output", "input_frame_dir", "input_video",
                  "input_camera_device", "input_synthetic", "camera_width", "camera_height", "camera_fps",
                  "camera_pixel_format", "camera_rtsp_url", "camera_rtsp_transport",
                  "camera_rtsp_bitrate_bps", "camera_rtsp_gop", "camera_rtsp_queue_capacity",
                  "camera_preload_frames",
                  "execution_profile", "manifest", "qp", "gop", "reset_interval", "device",
                  "frame_num", "fps", "profile_warmup_frames", "profile_output",
                  "enable_stage_fusion", "ltr_start_idx", "ltr_period", "ltr_qp_shift",
                  "target_bitrate_bps", "min_qp", "max_qp", "forced_ltr_recovery_frame",
                  "forced_ltr_reference_frame", "udp_host", "udp_port",
                  "output_transport_host", "output_transport_mode", "output_transport_port",
                  "output_transport_payload_type",
                  "output_transport_pacing_rate_bps", "output_transport_max_burst_bytes",
                  "output_transport_max_queue_bytes", "output_transport_max_queue_delay_ms",
                  "model", "pipeline"},
                 "");
  if (const toml::node* model = config["model"].node(); model != nullptr) {
    mlvc::Check(model->is_table(), "config section must be a table: model");
    CheckKnownKeys(*model->as_table(), {"manifest", "input_frame_dir", "input_video"}, "model");
  }
  if (const toml::node* pipeline = config["pipeline"].node(); pipeline != nullptr) {
    mlvc::Check(pipeline->is_table(), "config section must be a table: pipeline");
    CheckKnownKeys(*pipeline->as_table(), {"stream_workers", "queue_capacity", "frame_buffer_slots",
                                            "entropy_workers", "graph_packet_capacity"},
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

bool GetBool(const toml::table& config, const std::string& key, bool fallback) {
  if (config[key].node() == nullptr) return fallback;
  const std::optional<bool> value = config[key].value<bool>();
  mlvc::Check(value.has_value(), "config key must be boolean: " + key);
  return *value;
}

}  // namespace

namespace mlvc {

EncoderApplicationConfig LoadEncoderConfig(const std::filesystem::path& config_path) {
  const toml::table config = ReadTomlConfig(config_path);
  ValidateConfigKeys(config);
  const std::string mode = GetString(config, "mode", "encode");
  Check(mode == "encode", "encoder config must use mode = \"encode\"");
  const std::string input = GetString(config, "input");
  const std::string output = GetString(config, "output");
  const std::string input_frame_dir =
      GetString(config, "input_frame_dir", GetNestedString(config, "model", "input_frame_dir"));
  const std::string input_video =
      GetString(config, "input_video", GetNestedString(config, "model", "input_video"));
  const std::string camera_device = GetString(config, "input_camera_device");
  const std::string execution_profile =
      GetString(config, "execution_profile", std::string(mlvc::codec::kCodecExecutionProfile));
  // Issue #7 fix: Validate execution_profile
  Check(execution_profile == "pipeline-v1",
        "only pipeline-v1 execution profile is supported, got: " + execution_profile);

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
  Check(options.device >= 0, "device must be non-negative");
  options.frame_num = GetInt(config, "frame_num", -1);
  options.input_synthetic = GetBool(config, "input_synthetic", false);
  Check(options.frame_num == -1 || options.frame_num > 0,
        "frame_num must be -1 or a positive value");
  options.fps = GetDouble(config, "fps", 30.0);
  Check(std::isfinite(options.fps) && options.fps >= 0.1 && options.fps <= 1000.0,
        "fps must be finite and in [0.1, 1000]");
  options.profile_warmup_frames = GetInt(config, "profile_warmup_frames", 0);
  Check(options.profile_warmup_frames >= 0, "profile_warmup_frames must be non-negative");
  Check(options.frame_num <= 0 || options.profile_warmup_frames < options.frame_num,
        "profile_warmup_frames must be smaller than frame_num");
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
  Check(!options.input_synthetic ||
            (camera_device.empty() && input_video.empty() && input_frame_dir.empty() &&
             input.empty() && options.frame_num > 0),
        "synthetic input requires a finite frame_num and no other input source");
  Check(camera_device.empty() ||
            (input_video.empty() && input_frame_dir.empty() && input.empty()),
        "camera input cannot be combined with input_video or input_frame_dir");
  if (!camera_device.empty()) {
    options.input_camera_device = camera_device;
    mlvc::io::CameraCaptureOptions camera_options;
    camera_options.width = GetInt(config, "camera_width", 1920);
    camera_options.height = GetInt(config, "camera_height", 1080);
    camera_options.fps = GetDouble(config, "camera_fps", 30.0);
    camera_options.pixel_format = GetString(config, "camera_pixel_format", "MJPG");
    camera_options.rtsp_url = GetString(config, "camera_rtsp_url");
    camera_options.rtsp_transport = GetString(config, "camera_rtsp_transport", "tcp");
    const int camera_rtsp_bitrate = GetInt(config, "camera_rtsp_bitrate_bps", 8'000'000);
    const int camera_rtsp_gop = GetInt(config, "camera_rtsp_gop", 96);
    const int camera_rtsp_queue_capacity = GetInt(config, "camera_rtsp_queue_capacity", 3);
    const int camera_preload_frames = GetInt(config, "camera_preload_frames", 0);
    Check(camera_options.width > 0 && camera_options.height > 0 &&
              camera_options.width % 2 == 0 && camera_options.height % 2 == 0,
          "camera dimensions must be positive and even");
    Check(std::isfinite(camera_options.fps) && camera_options.fps > 0.0 &&
              camera_options.fps <= 240.0,
          "camera_fps must be in (0, 240]");
    Check(camera_options.pixel_format == "MJPG", "camera_pixel_format must be MJPG");
    Check(camera_options.rtsp_transport == "tcp" || camera_options.rtsp_transport == "udp",
          "camera_rtsp_transport must be tcp or udp");
    Check(camera_rtsp_bitrate >= 2'000 && camera_rtsp_bitrate <= 614'400'000,
          "camera_rtsp_bitrate_bps must be in [2000, 614400000]");
    Check(camera_rtsp_gop > 0, "camera_rtsp_gop must be positive");
    Check(camera_rtsp_queue_capacity > 0, "camera_rtsp_queue_capacity must be positive");
    Check(camera_preload_frames >= 0 && camera_preload_frames <= 3000,
          "camera_preload_frames must be in [0, 3000]");
    Check(camera_preload_frames == 0 || options.frame_num > 0,
          "camera preload benchmark requires a finite frame_num");
    Check(camera_preload_frames == 0 || camera_options.rtsp_url.empty(),
          "camera preload benchmark cannot publish RTSP");
    camera_options.rtsp_bitrate_bps = static_cast<uint32_t>(camera_rtsp_bitrate);
    camera_options.rtsp_gop = static_cast<uint32_t>(camera_rtsp_gop);
    camera_options.rtsp_queue_capacity = static_cast<std::size_t>(camera_rtsp_queue_capacity);
    camera_options.preload_frames = static_cast<std::size_t>(camera_preload_frames);
    options.camera_options = camera_options;
    options.input_video_path = camera_device;
  }
  const std::string video_input =
      input_video.empty() && mlvc::io::IsVideoPath(input) ? input : input_video;
  if (!camera_device.empty()) {
    // Camera input has already been selected above.
  } else if (!video_input.empty()) {
    options.input_video_path = video_input;
  } else {
    options.input_frame_dir = frame_input;
  }
  options.output_bitstream_path = output;
  options.ltr_start_idx = GetInt(config, "ltr_start_idx", 8);
  options.ltr_period = GetInt(config, "ltr_period", 64);
  options.ltr_qp_shift = GetInt(config, "ltr_qp_shift", 8);
  Check(options.ltr_start_idx >= 0, "ltr_start_idx must be non-negative");
  Check(options.ltr_period >= 0, "ltr_period must be non-negative");
  options.target_bitrate_bps = GetDouble(config, "target_bitrate_bps", 0.0);
  options.min_qp = GetInt(config, "min_qp", 0);
  options.max_qp = GetInt(config, "max_qp", 63);
  Check(options.ltr_qp_shift >= 0 && options.ltr_qp_shift <= 63, "ltr_qp_shift must be in [0, 63]");
  Check(std::isfinite(options.target_bitrate_bps) && options.target_bitrate_bps >= 0.0,
        "target_bitrate_bps must be finite and non-negative");
  const double max_target_bitrate_bps = std::min(
      static_cast<double>(std::numeric_limits<int>::max()) * 0.25,
      static_cast<double>(std::numeric_limits<int>::max()) * options.fps / 16.0);
  Check(options.target_bitrate_bps <= max_target_bitrate_bps,
        "target_bitrate_bps exceeds the rate controller's integer budget");
  Check(options.min_qp >= 0 && options.min_qp <= 63, "min_qp must be in [0, 63]");
  Check(options.max_qp >= 0 && options.max_qp <= 63, "max_qp must be in [0, 63]");
  Check(options.min_qp <= options.max_qp, "min_qp must not exceed max_qp");
  options.forced_ltr_recovery_frame = GetInt(config, "forced_ltr_recovery_frame", -1);
  options.forced_ltr_reference_frame = GetInt(config, "forced_ltr_reference_frame", -1);
  mlvc::codec::ValidateForcedLtrConfiguration(
      options.gop, options.ltr_start_idx, options.ltr_period, options.forced_ltr_recovery_frame,
      options.forced_ltr_reference_frame);
  Check(config["udp_host"].node() == nullptr && config["udp_port"].node() == nullptr,
        "udp_host and udp_port are no longer supported; use output_transport_host and "
        "output_transport_port");
  options.output_transport_host = GetString(config, "output_transport_host");
  options.output_transport_mode = GetString(config, "output_transport_mode", "udp");
  Check(options.output_transport_mode == "udp" || options.output_transport_mode == "rtp",
        "output_transport_mode must be udp or rtp");
  options.output_transport_port = GetInt(config, "output_transport_port", 0);
  const int payload_type = GetInt(config, "output_transport_payload_type", 96);
  const int pacing_rate_bps = GetInt(config, "output_transport_pacing_rate_bps", 0);
  const int max_burst_bytes = GetInt(config, "output_transport_max_burst_bytes", 4096);
  const int max_queue_bytes = GetInt(config, "output_transport_max_queue_bytes", 4 * 1024 * 1024);
  const int max_queue_delay_ms = GetInt(config, "output_transport_max_queue_delay_ms", 1000);
  Check(pacing_rate_bps >= 0, "output_transport_pacing_rate_bps must be non-negative");
  Check(payload_type >= 96 && payload_type <= 127,
        "output_transport_payload_type must be in the dynamic RTP range [96, 127]");
  const int min_packet_burst_bytes = options.output_transport_mode == "rtp" ? 1240 : 1068;
  Check(max_burst_bytes >= min_packet_burst_bytes,
        "output_transport_max_burst_bytes must hold at least one complete paced packet");
  Check(max_queue_bytes > 0, "output_transport_max_queue_bytes must be positive");
  Check(max_queue_delay_ms >= 0, "output_transport_max_queue_delay_ms must be non-negative");
  options.output_transport_pacing_rate_bps = static_cast<uint64_t>(pacing_rate_bps);
  options.output_transport_payload_type = static_cast<uint8_t>(payload_type);
  options.output_transport_max_burst_bytes = static_cast<std::size_t>(max_burst_bytes);
  options.output_transport_max_queue_bytes = static_cast<std::size_t>(max_queue_bytes);
  options.output_transport_max_queue_delay_ms = static_cast<uint64_t>(max_queue_delay_ms);
  Check(options.output_transport_port >= 0 && options.output_transport_port <= 65535,
        "output_transport_port must be in [0, 65535]");
  Check(options.output_transport_port == 0 || !options.output_transport_host.empty(),
        "output_transport_host is required when output_transport_port is set");
  if (options.output_transport_port > 0) {
    options.output_bitstream_path.clear();
  }
  const int stream_workers = GetNestedInt(config, "pipeline", "stream_workers", 1);
  const int queue_capacity = GetNestedInt(config, "pipeline", "queue_capacity", 3);
  const int frame_buffer_slots = GetNestedInt(config, "pipeline", "frame_buffer_slots", 3);
  const int entropy_workers = GetNestedInt(config, "pipeline", "entropy_workers", 1);
  const int graph_packet_capacity = GetNestedInt(config, "pipeline", "graph_packet_capacity", 2);
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
