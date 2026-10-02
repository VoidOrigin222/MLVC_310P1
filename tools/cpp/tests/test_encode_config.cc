#include <mlvc/application/cli/encoder_app.h>
#include <mlvc/core/status.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {

void WriteManifest(const std::filesystem::path& path) {
  std::ofstream output(path);
  output << R"({
  "dtype": "float16",
  "sidecar": {"file": "sidecar.bin", "bytes": 0, "sha256": ""},
  "models": [
    {"name": "MLVCEncoder", "bytes": 1, "sha256": ""},
    {"name": "MLVCDecoder", "bytes": 1, "sha256": ""}
  ]
})";
  mlvc::Check(output.good(), "failed to write encoder config manifest");
}

template <typename Function>
void ExpectReject(Function&& function, const char* description) {
  bool rejected = false;
  try {
    function();
  } catch (const std::exception&) {
    rejected = true;
  }
  mlvc::Check(rejected, std::string("expected encoder config rejection: ") + description);
}

}  // namespace

int main() {
  try {
    const std::filesystem::path manifest = "/tmp/mlvc_test_encode_manifest.json";
    const std::filesystem::path config = "/tmp/mlvc_test_encode_config.toml";
    WriteManifest(manifest);
    {
      std::ofstream output(config);
      output << "mode = \"encode\"\n"
             << "input_frame_dir = \"/tmp\"\n"
             << "output = \"/tmp/out.mlvc\"\n"
             << "manifest = \"" << manifest.string() << "\"\n"
             << "fps = 25.0\nframe_num = 10\nprofile_warmup_frames = 3\n"
             << "ltr_qp_shift = 8\nmin_qp = 4\nmax_qp = 40\n"
             << "output_transport_mode = \"rtp\"\n"
             << "output_transport_payload_type = 110\n"
             << "output_transport_pacing_rate_bps = 1000000\n"
             << "output_transport_max_queue_bytes = 65536\n"
             << "output_transport_max_queue_delay_ms = 250\n"
             << "[pipeline]\nstream_workers = 3\nframe_buffer_slots = 4\n";
    }
    const mlvc::EncoderApplicationConfig parsed = mlvc::LoadEncoderConfig(config);
    mlvc::Check(parsed.stream.fps == 25.0, "floating-point encoder fps was not preserved");
    mlvc::Check(parsed.stream.profile_warmup_frames == 3,
                "profile warmup frames were not preserved");
    mlvc::Check(parsed.stream.min_qp == 4 && parsed.stream.max_qp == 40,
                "encoder QP range was not preserved");
    mlvc::Check(parsed.stream.output_transport_payload_type == 110,
                "negotiated RTP payload type was not preserved");
    mlvc::Check(parsed.stream.output_transport_max_queue_bytes == 65536 &&
                    parsed.stream.output_transport_max_queue_delay_ms == 250,
                "transport queue limits were not preserved");
    mlvc::Check(parsed.stream.pipeline.stream_workers == 3,
                "configured encoder stream worker count was not preserved");

    const auto expect_invalid = [&](const char* extra, const char* description) {
      std::ofstream output(config);
      output << "mode = \"encode\"\ninput_frame_dir = \"/tmp\"\n"
             << "output = \"/tmp/out.mlvc\"\nmanifest = \"" << manifest.string()
             << "\"\nframe_num = 10\n"
             << extra;
      output.close();
      ExpectReject([&] { (void)mlvc::LoadEncoderConfig(config); }, description);
    };
    expect_invalid("output_transport_pacing_rate_bps = -1\n", "negative pacing rate");
    expect_invalid("output_transport_payload_type = 128\n", "payload type above dynamic range");
    expect_invalid("ltr_qp_shift = -1\n", "negative LTR shift");
    expect_invalid("min_qp = 50\nmax_qp = 40\n", "inverted QP range");
    expect_invalid("profile_warmup_frames = 10\n", "warmup consumes all frames");
    expect_invalid("frame_num = -2\n", "frame count below sentinel range");
    expect_invalid("device = -1\n", "negative device index");
    expect_invalid("ltr_start_idx = -1\n", "negative LTR start index");
    expect_invalid("ltr_period = -1\n", "negative LTR period");
    expect_invalid("execution_profile = \"legacy\"\n", "unsupported execution profile");
    expect_invalid("qp = 64\n", "QP above supported range");
    expect_invalid("ltr_qp_shift = 64\n", "LTR shift above supported range");
    expect_invalid("fps = nan\n", "non-finite FPS");
    expect_invalid("output_transport_max_queue_bytes = 0\n", "zero transport queue bytes");
    expect_invalid("output_transport_max_burst_bytes = 1000\n",
                   "burst smaller than a UDP packet");
    expect_invalid("output_transport_mode = \"rtp\"\n"
                   "output_transport_max_burst_bytes = 1200\n",
                   "burst smaller than an RTP/UDP packet");
    expect_invalid("output_transport_max_queue_delay_ms = -1\n",
                   "negative transport queue delay");
    expect_invalid("fps = \"25\"\n", "string FPS");
    expect_invalid("execution_profile = 123\n", "non-string execution profile");
    expect_invalid("qp = 1.5\n", "floating-point QP");
    expect_invalid("target_bitrate_bps = \"1000000\"\n", "string target bitrate");
    expect_invalid("target_bitrate_bps = 1e308\n", "target bitrate beyond integer budget");
    expect_invalid("[pipeline]\nstream_workers = \"4\"\n",
                   "string pipeline worker count");
    expect_invalid("pipeline = 4\n", "non-table pipeline section");
    expect_invalid("target_bitrate = 1000000\n", "misspelled bitrate config key");
    expect_invalid("motion_backend = \"unknown\"\n", "unknown motion backend");
    expect_invalid("motion_backend = \"dvpp\"\n", "hardware motion without warp enabled");
    expect_invalid("translation_warp = true\nmotion_backend = \"dvpp\"\n"
                   "motion_shifts_file = \"/tmp/shifts.csv\"\n",
                   "simultaneous hardware motion and CSV replay");
    expect_invalid("motion_shifts_file = \"/tmp/shifts.csv\"\n",
                   "CSV replay without warp enabled");
    {
      std::ofstream output(config);
      output << "mode = \"encode\"\ninput_frame_dir = \"/tmp\"\n"
             << "output = \"/tmp/out.mlvc\"\nmanifest = \"" << manifest.string()
             << "\"\nframe_num = 10\ntranslation_warp = true\n"
             << "motion_backend = \"dvpp\"\nltr_period = 0\n";
    }
    const auto warp_config = mlvc::LoadEncoderConfig(config);
    mlvc::Check(warp_config.stream.translation_warp && warp_config.stream.motion_backend == "dvpp",
                "valid hardware motion configuration was not preserved");

    std::cout << "encode config test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_encode_config failed: " << error.what() << "\n";
    return 1;
  }
}
