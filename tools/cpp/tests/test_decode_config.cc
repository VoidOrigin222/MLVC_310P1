#include <mlvc/application/cli/decoder_app.h>
#include <mlvc/core/status.h>

#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>

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
  mlvc::Check(output.good(), "failed to write decoder config manifest");
}

}  // namespace

int main() {
  try {
    const auto temp_dir = std::filesystem::temp_directory_path();
    const std::filesystem::path config = temp_dir / "mlvc_test_decode_config.toml";
    const std::filesystem::path manifest = temp_dir / "mlvc_test_decode_manifest.json";
    WriteManifest(manifest);
    std::ofstream output(config);
    output << "mode = \"decode\"\n"
           << "format = \"none\"\n"
           << "input_transport_port = 39089\n"
           << "input_transport_mode = \"rtp\"\n"
           << "input_transport_payload_type = 110\n"
           << "manifest = \"" << manifest.string() << "\"\n"
           << "[pipeline]\n"
           << "queue_capacity = 7\n"
           << "frame_buffer_slots = 4\n"
           << "entropy_workers = 2\n";
    output.close();
    mlvc::Check(output.good(), "failed to write decode config fixture");
    const mlvc::DecoderApplicationConfig parsed = mlvc::LoadDecoderConfig(config);
    mlvc::Check(parsed.stream.input_transport_port == 39089,
                "input transport port was not preserved");
    mlvc::Check(parsed.stream.input_transport_mode == "rtp",
                "input transport mode was not preserved");
    mlvc::Check(parsed.stream.input_transport_payload_type == 110,
                "negotiated RTP payload type was not preserved");
    mlvc::Check(parsed.stream.pipeline.entropy_workers == 2,
                "configured entropy worker count was not preserved");
    mlvc::Check(parsed.stream.pipeline.queue_capacity == 7,
                "configured decoder queue capacity was not preserved");
    mlvc::Check(parsed.stream.pipeline.frame_buffer_slots == 4,
                "configured decoder frame buffer slots were not preserved");

    {
      std::ofstream rtsp(config);
      rtsp << "mode = \"decode\"\nmanifest = \"" << manifest.string()
           << "\"\noutput_transport_mode = \"rtsp\"\n"
           << "output_transport_queue_capacity = 5\n"
           << "output_transport_rtsp_url = \"rtsp://127.0.0.1:8554/mlvc\"\n"
           << "output_transport_rtsp_encoder = \"dvpp\"\n"
           << "output_transport_rtsp_preset = \"ultrafast\"\n"
           << "output_transport_rtsp_transport = \"udp\"\n"
           << "output_transport_rtsp_crf = 0\n";
      rtsp.close();
      const mlvc::DecoderApplicationConfig parsed_rtsp = mlvc::LoadDecoderConfig(config);
      mlvc::Check(parsed_rtsp.stream.output_transport_mode == "rtsp",
                  "RTSP output mode was not preserved");
      mlvc::Check(parsed_rtsp.stream.output_transport_rtsp_url == "rtsp://127.0.0.1:8554/mlvc",
                  "RTSP URL was not preserved");
      mlvc::Check(parsed_rtsp.stream.output_transport_rtsp_encoder == "dvpp",
                  "RTSP encoder was not preserved");
      mlvc::Check(parsed_rtsp.stream.output_transport_rtsp_preset == "ultrafast",
                  "RTSP preset was not preserved");
      mlvc::Check(parsed_rtsp.stream.output_transport_rtsp_transport == "udp",
                  "RTSP transport was not preserved");
      mlvc::Check(parsed_rtsp.stream.output_transport_queue_capacity == 5,
                  "RTSP queue capacity was not preserved");
    }

    bool rejected_profile = false;
    {
      std::ofstream invalid(config);
      invalid << "mode = \"decode\"\nmanifest = \"" << manifest.string()
              << "\"\nexecution_profile = \"legacy\"\n";
      invalid.close();
      try {
        (void)mlvc::LoadDecoderConfig(config);
      } catch (const std::exception&) {
        rejected_profile = true;
      }
    }
    mlvc::Check(rejected_profile, "unsupported execution profile must be rejected");

    {
      std::ofstream valid(config);
      valid << "mode = \"decode\"\nmanifest = \"" << manifest.string() << "\"\nfps = 25.0\n";
      valid.close();
      const mlvc::DecoderApplicationConfig parsed_fps = mlvc::LoadDecoderConfig(config);
      mlvc::Check(parsed_fps.stream.fps == 25.0, "floating-point fps was not preserved");
    }

    bool rejected_frame_num = false;
    {
      std::ofstream invalid(config);
      invalid << "mode = \"decode\"\nmanifest = \"" << manifest.string() << "\"\nframe_num = 0\n";
      invalid.close();
      try {
        (void)mlvc::LoadDecoderConfig(config);
      } catch (const std::exception&) {
        rejected_frame_num = true;
      }
    }
    mlvc::Check(rejected_frame_num, "zero frame_num must be rejected");

    bool rejected_capacity = false;
    {
      std::ofstream invalid(config);
      invalid << "mode = \"decode\"\nmanifest = \"" << manifest.string()
              << "\"\noutput_transport_queue_capacity = 0\n";
      invalid.close();
      try {
        (void)mlvc::LoadDecoderConfig(config);
      } catch (const std::exception&) {
        rejected_capacity = true;
      }
    }
    mlvc::Check(rejected_capacity, "zero forwarding queue capacity must be rejected");

    for (const std::string& removed_mode : {"jpeg_async", "raw_fp16_yuv444"}) {
      bool rejected_mode = false;
      std::ofstream invalid(config);
      invalid << "mode = \"decode\"\nmanifest = \"" << manifest.string()
              << "\"\noutput_transport_mode = \"" << removed_mode << "\"\n";
      invalid.close();
      try {
        (void)mlvc::LoadDecoderConfig(config);
      } catch (const std::exception&) {
        rejected_mode = true;
      }
      mlvc::Check(rejected_mode, "removed forwarding mode must be rejected");
    }

    bool rejected_rtsp_without_url = false;
    {
      std::ofstream invalid(config);
      invalid << "mode = \"decode\"\nmanifest = \"" << manifest.string()
              << "\"\noutput_transport_mode = \"rtsp\"\n";
      invalid.close();
      try {
        (void)mlvc::LoadDecoderConfig(config);
      } catch (const std::exception&) {
        rejected_rtsp_without_url = true;
      }
    }
    mlvc::Check(rejected_rtsp_without_url, "RTSP mode without URL must be rejected");

    bool rejected_rtsp_transport = false;
    {
      std::ofstream invalid(config);
      invalid << "mode = \"decode\"\nmanifest = \"" << manifest.string()
              << "\"\noutput_transport_mode = \"rtsp\"\n"
              << "output_transport_rtsp_url = \"rtsp://127.0.0.1:8554/mlvc\"\n"
              << "output_transport_rtsp_transport = \"sctp\"\n";
      invalid.close();
      try {
        (void)mlvc::LoadDecoderConfig(config);
      } catch (const std::exception&) {
        rejected_rtsp_transport = true;
      }
    }
    mlvc::Check(rejected_rtsp_transport, "unsupported RTSP transport must be rejected");

    bool rejected_rtsp_encoder = false;
    {
      std::ofstream invalid(config);
      invalid << "mode = \"decode\"\nmanifest = \"" << manifest.string()
              << "\"\noutput_transport_rtsp_encoder = \"invalid\"\n";
      invalid.close();
      try {
        (void)mlvc::LoadDecoderConfig(config);
      } catch (const std::exception&) {
        rejected_rtsp_encoder = true;
      }
    }
    mlvc::Check(rejected_rtsp_encoder, "unsupported RTSP encoder must be rejected");

    for (const std::string& invalid_execution :
         {"rans-convert-pipeline", "rans-narrow-convert-pipeline", "unknown"}) {
      bool rejected_entropy_execution = false;
      std::ofstream invalid(config);
      invalid << "mode = \"decode\"\nmanifest = \"" << manifest.string()
              << "\"\n[pipeline]\nentropy_execution = \"" << invalid_execution << "\"\n";
      invalid.close();
      try {
        (void)mlvc::LoadDecoderConfig(config);
      } catch (const std::exception&) {
        rejected_entropy_execution = true;
      }
      mlvc::Check(rejected_entropy_execution,
                  "non-frame-parallel entropy execution must be rejected");
    }

    for (const auto& invalid_case :
         {std::pair<const char*, const char*>{"fps = \"25\"\n", "string FPS"},
         {"execution_profile = 123\n", "non-string execution profile"},
          {"device = -1\n", "negative device index"},
          {"input_transport_port = \"39089\"\n", "string input port"},
          {"output_transport_queue_capacity = 1.5\n", "floating-point output queue capacity"},
          {"[pipeline]\nqueue_capacity = \"4\"\n", "string pipeline capacity"},
         {"pipeline = 4\n", "non-table pipeline section"},
         {"input_transprot_port = 39089\n", "misspelled input transport port"}}) {
      bool rejected_type = false;
      std::ofstream invalid(config);
      invalid << "mode = \"decode\"\nmanifest = \"" << manifest.string() << "\"\n"
              << invalid_case.first;
      invalid.close();
      try {
        (void)mlvc::LoadDecoderConfig(config);
      } catch (const std::exception&) {
        rejected_type = true;
      }
      mlvc::Check(rejected_type,
                  std::string("invalid decoder config type was accepted: ") + invalid_case.second);
    }

    for (int payload_type : {95, 128}) {
      std::ofstream invalid(config);
      invalid << "mode = \"decode\"\nmanifest = \"" << manifest.string()
              << "\"\ninput_transport_payload_type = " << payload_type << "\n";
      invalid.close();
      bool rejected_payload_type = false;
      try {
        (void)mlvc::LoadDecoderConfig(config);
      } catch (const std::exception&) {
        rejected_payload_type = true;
      }
      mlvc::Check(rejected_payload_type, "RTP payload type outside the dynamic range was accepted");
    }

    std::cout << "decode config test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_decode_config failed: " << error.what() << "\n";
    return 1;
  }
}
