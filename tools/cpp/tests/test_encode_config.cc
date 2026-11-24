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
             << "mlvc_stats_host = \"127.0.0.1\"\n"
             << "mlvc_stats_port = 39341\n"
             << "mlvc_stats_interval_frames = 30\n"
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
    mlvc::Check(parsed.stream.mlvc_stats_host == "127.0.0.1" &&
                    parsed.stream.mlvc_stats_port == 39341 &&
                    parsed.stream.mlvc_stats_interval_frames == 30,
                "MLVC stats destination was not preserved");
    mlvc::Check(parsed.stream.pipeline.stream_workers == 3,
                "configured encoder stream worker count was not preserved");
    mlvc::Check(parsed.stream.motion_prefetch_frames == 0,
                "motion prefetch must preserve the serial default");
    mlvc::Check(parsed.stream.motion_x264_preset == "medium" && parsed.stream.motion_x264_threads == 1,
                "x264 motion proxy defaults must remain medium and one thread");
    mlvc::Check(!parsed.stream.motion_skip_loop_filter,
                "motion proxy loop filtering must remain enabled by default");
    mlvc::Check(!parsed.stream.motion_camera_nv12,
                "default motion must remain full resolution without a camera sidecar");

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
    expect_invalid("mlvc_stats_port = 65536\n", "MLVC stats port above range");
    expect_invalid("mlvc_stats_interval_frames = 0\n", "zero MLVC stats interval");
    expect_invalid("mlvc_stats_port = 39341\n", "MLVC stats port without host");
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
    expect_invalid("translation_warp = true\nmotion_prefetch_frames = -1\n",
                   "negative motion prefetch depth");
    expect_invalid("translation_warp = true\nmotion_prefetch_frames = 3\n",
                   "motion prefetch above two frames");
    expect_invalid("motion_prefetch_frames = 1\n", "motion prefetch without warp enabled");
    expect_invalid("translation_warp = true\nmotion_prefetch_frames = 1.5\n",
                   "fractional motion prefetch depth");
    expect_invalid("translation_warp = true\nmotion_x264_preset = \"slow\"\n",
                   "unsupported motion x264 preset");
    expect_invalid("translation_warp = true\nmotion_x264_threads = 0\n",
                   "zero motion x264 threads");
    expect_invalid("translation_warp = true\nmotion_x264_threads = 17\n",
                   "motion x264 threads above limit");
    expect_invalid("translation_warp = true\nmotion_x264_threads = 1.5\n",
                   "fractional motion x264 threads");
    expect_invalid("motion_x264_preset = \"ultrafast\"\n",
                   "non-default x264 preset without warp");
    expect_invalid("motion_x264_threads = 2\n", "non-default x264 threads without warp");
    expect_invalid("motion_skip_loop_filter = true\n", "skip motion loop filter without warp");
    expect_invalid("translation_warp = true\nmotion_shifts_file = \"/tmp/shifts.csv\"\n"
                   "motion_skip_loop_filter = true\n",
                   "skip motion loop filter with CSV replay");
    expect_invalid("translation_warp = true\nmotion_skip_loop_filter = \"true\"\n",
                   "non-boolean motion loop filter option");
    expect_invalid("translation_warp = true\nmotion_backend = \"dvpp\"\n"
                   "motion_x264_preset = \"veryfast\"\n",
                   "x264 preset with hardware motion");
    expect_invalid("translation_warp = true\nmotion_backend = \"dvpp\"\n"
                   "motion_x264_threads = 2\n",
                   "x264 threads with hardware motion");
    expect_invalid("translation_warp = true\nmotion_shifts_file = \"/tmp/shifts.csv\"\n"
                   "motion_x264_preset = \"superfast\"\n",
                   "x264 preset with CSV replay");
    expect_invalid("translation_warp = true\nmotion_shifts_file = \"/tmp/shifts.csv\"\n"
                   "motion_x264_threads = 2\n",
                   "x264 threads with CSV replay");
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
    for (const int depth : {1, 2}) {
      {
        std::ofstream output(config);
        output << "mode = \"encode\"\ninput_frame_dir = \"/tmp\"\n"
               << "output = \"/tmp/out.mlvc\"\nmanifest = \"" << manifest.string()
               << "\"\nframe_num = 10\ntranslation_warp = true\n"
               << "motion_backend = \"dvpp\"\nltr_period = 0\n"
               << "motion_prefetch_frames = " << depth << "\n";
      }
      mlvc::Check(mlvc::LoadEncoderConfig(config).stream.motion_prefetch_frames == depth,
                  "valid bounded motion prefetch depth was not preserved");
    }
    for (const std::string preset : {"medium", "veryfast", "superfast", "ultrafast"}) {
      for (const int threads : {1, 16}) {
        {
          std::ofstream output(config);
          output << "mode = \"encode\"\ninput_frame_dir = \"/tmp\"\n"
                 << "output = \"/tmp/out.mlvc\"\nmanifest = \"" << manifest.string()
                 << "\"\nframe_num = 10\ntranslation_warp = true\nltr_period = 0\n"
                 << "motion_backend = \"libx264\"\nmotion_x264_preset = \"" << preset
                 << "\"\nmotion_x264_threads = " << threads << "\n";
        }
        const auto online_config = mlvc::LoadEncoderConfig(config);
        mlvc::Check(online_config.stream.motion_x264_preset == preset &&
                        online_config.stream.motion_x264_threads == threads,
                    "valid online x264 tuning was not preserved");
      }
    }
    for (const std::string backend : {"dvpp", "libx264"}) {
      for (const bool skip_filter : {false, true}) {
        {
          std::ofstream output(config);
          output << "mode = \"encode\"\ninput_frame_dir = \"/tmp\"\n"
                 << "output = \"/tmp/out.mlvc\"\nmanifest = \"" << manifest.string()
                 << "\"\nframe_num = 10\ntranslation_warp = true\nltr_period = 0\n"
                 << "motion_backend = \"" << backend << "\"\n"
                 << "motion_skip_loop_filter = " << (skip_filter ? "true" : "false") << "\n";
        }
        const auto filtered_config = mlvc::LoadEncoderConfig(config);
        mlvc::Check(filtered_config.stream.motion_backend == backend &&
                        filtered_config.stream.motion_skip_loop_filter == skip_filter,
                    "valid online motion loop filter option was not preserved");
      }
    }

    const auto write_motion_input = [&](bool camera, const std::string& fields) {
      std::ofstream output(config);
      output << "mode = \"encode\"\noutput = \"/tmp/out.mlvc\"\nmanifest = \""
             << manifest.string() << "\"\nframe_num = 10\nltr_period = 0\n";
      if (camera) {
        // Parsing must not open or capture this test-only device.
        output << "input_camera_device = \"/dev/mlvc-config-test\"\n"
               << "camera_width = 1920\ncamera_height = 1080\n";
      } else {
        output << "input_frame_dir = \"/tmp\"\n";
      }
      output << fields;
    };
    const auto reject_motion_input = [&](bool camera, const char* fields, const char* reason) {
      write_motion_input(camera, fields);
      ExpectReject([&] { (void)mlvc::LoadEncoderConfig(config); }, reason);
    };
    reject_motion_input(false, "translation_warp = true\nmotion_prefetch_frames = 2\n"
                              "motion_camera_nv12 = true\n", "camera sidecar without camera input");
    reject_motion_input(true, "motion_camera_nv12 = true\n", "camera sidecar without warp");
    reject_motion_input(true, "translation_warp = true\nmotion_camera_nv12 = true\n",
                             "camera sidecar without motion prefetch");
    reject_motion_input(true, "translation_warp = true\nmotion_prefetch_frames = 2\n"
                             "motion_shifts_file = \"/tmp/shifts.csv\"\nmotion_camera_nv12 = true\n",
                             "camera sidecar with CSV replay");
    reject_motion_input(true, "translation_warp = true\nmotion_prefetch_frames = 2\n"
                             "motion_camera_nv12 = \"true\"\n", "non-boolean camera sidecar option");
    for (const std::string backend : {"dvpp", "libx264"}) {
        for (bool camera : {false, true}) {
          const std::string fields = "translation_warp = true\nmotion_prefetch_frames = 2\n"
              "motion_backend = \"" + backend + "\"\nmotion_camera_nv12 = " +
              (camera ? "true\n" : "false\n");
          write_motion_input(camera, fields);
          const auto camera_config = mlvc::LoadEncoderConfig(config);
          mlvc::Check(camera_config.stream.motion_camera_nv12 == camera,
                      "valid full-resolution sidecar configuration was not preserved");
          if (camera) {
            mlvc::Check(camera_config.stream.camera_options.has_value() &&
                            camera_config.stream.camera_options->width == 1920 &&
                            camera_config.stream.camera_options->height == 1080 &&
                            camera_config.stream.camera_options->motion_nv12,
                        "camera sidecar must preserve full-resolution camera geometry");
          }
        }
    }

    write_motion_input(true,
        "camera_fps = 30\ncamera_rtsp_url = \"rtsp://127.0.0.1:8554/original\"\n"
        "camera_rtsp_transport = \"udp\"\ncamera_rtsp_bitrate_bps = 6000000\n"
        "camera_rtsp_gop = 60\ncamera_rtsp_queue_capacity = 2\n"
        "translation_warp = true\nmotion_backend = \"dvpp\"\n"
        "motion_prefetch_frames = 2\nmotion_camera_nv12 = true\n");
    const auto rtsp_config = mlvc::LoadEncoderConfig(config);
    mlvc::Check(rtsp_config.stream.camera_options.has_value(), "camera RTSP options missing");
    const auto& rtsp_options = *rtsp_config.stream.camera_options;
    mlvc::Check(rtsp_options.fps == 30.0 &&
                    rtsp_options.rtsp_url == "rtsp://127.0.0.1:8554/original" &&
                    rtsp_options.rtsp_transport == "udp" &&
                    rtsp_options.rtsp_bitrate_bps == 6'000'000 &&
                    rtsp_options.rtsp_gop == 60 && rtsp_options.rtsp_queue_capacity == 2 &&
                    rtsp_config.stream.motion_backend == "dvpp" && rtsp_options.motion_nv12,
                "DVPP camera RTSP tuning and simultaneous hardware motion were not preserved");
    reject_motion_input(true,
        "camera_fps = 29.97\ncamera_rtsp_url = \"rtsp://127.0.0.1:8554/original\"\n",
        "fractional DVPP camera RTSP frame rate");
    reject_motion_input(true,
        "camera_fps = 0.5\ncamera_rtsp_url = \"rtsp://127.0.0.1:8554/original\"\n",
        "DVPP camera RTSP frame rate below one");
    write_motion_input(true, "camera_fps = 29.97\n");
    mlvc::Check(mlvc::LoadEncoderConfig(config).stream.camera_options->fps == 29.97,
                "fractional camera FPS without RTSP must remain supported");

    std::cout << "encode config test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_encode_config failed: " << error.what() << "\n";
    return 1;
  }
}
