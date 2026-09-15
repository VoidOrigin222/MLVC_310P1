#include <mlvc/application/cli/decoder_app.h>
#include <mlvc/core/status.h>

#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  try {
    mlvc::Check(argc == 2, "test_decode_config requires a manifest path");
    const std::filesystem::path config = "/tmp/mlvc_test_decode_config.toml";
    std::ofstream output(config);
    output << "mode = \"decode\"\n"
           << "format = \"none\"\n"
           << "input_transport_port = 39089\n"
           << "input_transport_mode = \"rtp\"\n"
           << "output_transport_host = \"127.0.0.1\"\n"
           << "output_transport_port = 50000\n"
           << "output_transport_mode = \"raw_fp16_yuv444\"\n"
           << "output_transport_queue_capacity = 5\n"
           << "manifest = \"" << argv[1] << "\"\n"
           << "[pipeline]\n"
           << "queue_capacity = 7\n"
           << "frame_buffer_slots = 4\n"
           << "entropy_workers = 2\n";
    output.close();
    mlvc::Check(output.good(), "failed to write decode config fixture");
    const mlvc::DecoderApplicationConfig parsed = mlvc::LoadDecoderConfig(config);
    mlvc::Check(parsed.stream.output_transport_mode == "raw_fp16_yuv444",
                "raw forward mode was not preserved");
    mlvc::Check(parsed.stream.output_transport_queue_capacity == 5,
                "DVPP queue capacity was not preserved");
    mlvc::Check(parsed.stream.input_transport_port == 39089,
                "input transport port was not preserved");
    mlvc::Check(parsed.stream.input_transport_mode == "rtp",
                "input transport mode was not preserved");
    mlvc::Check(parsed.stream.pipeline.entropy_workers == 2,
                "configured entropy worker count was not preserved");
    mlvc::Check(parsed.stream.pipeline.queue_capacity == 7,
                "configured decoder queue capacity was not preserved");
    mlvc::Check(parsed.stream.pipeline.frame_buffer_slots == 4,
                "configured decoder frame buffer slots were not preserved");

    {
      std::ofstream rtsp(config);
      rtsp << "mode = \"decode\"\nmanifest = \"" << argv[1]
           << "\"\noutput_transport_mode = \"rtsp\"\n"
           << "output_transport_rtsp_url = \"rtsp://127.0.0.1:8554/mlvc\"\n"
           << "output_transport_rtsp_preset = \"ultrafast\"\n"
           << "output_transport_rtsp_transport = \"udp\"\n"
           << "output_transport_rtsp_crf = 0\n";
      rtsp.close();
      const mlvc::DecoderApplicationConfig parsed_rtsp = mlvc::LoadDecoderConfig(config);
      mlvc::Check(parsed_rtsp.stream.output_transport_mode == "rtsp",
                  "RTSP output mode was not preserved");
      mlvc::Check(parsed_rtsp.stream.output_transport_rtsp_url == "rtsp://127.0.0.1:8554/mlvc",
                  "RTSP URL was not preserved");
      mlvc::Check(parsed_rtsp.stream.output_transport_rtsp_preset == "ultrafast",
                  "RTSP preset was not preserved");
      mlvc::Check(parsed_rtsp.stream.output_transport_rtsp_transport == "udp",
                  "RTSP transport was not preserved");
    }

    bool rejected_profile = false;
    {
      std::ofstream invalid(config);
      invalid << "mode = \"decode\"\nmanifest = \"" << argv[1]
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
      valid << "mode = \"decode\"\nmanifest = \"" << argv[1] << "\"\nfps = 25.0\n";
      valid.close();
      const mlvc::DecoderApplicationConfig parsed_fps = mlvc::LoadDecoderConfig(config);
      mlvc::Check(parsed_fps.stream.fps == 25.0, "floating-point fps was not preserved");
    }

    bool rejected_frame_num = false;
    {
      std::ofstream invalid(config);
      invalid << "mode = \"decode\"\nmanifest = \"" << argv[1] << "\"\nframe_num = 0\n";
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
      invalid << "mode = \"decode\"\nmanifest = \"" << argv[1]
              << "\"\noutput_transport_queue_capacity = 0\n";
      invalid.close();
      try {
        (void)mlvc::LoadDecoderConfig(config);
      } catch (const std::exception&) {
        rejected_capacity = true;
      }
    }
    mlvc::Check(rejected_capacity, "zero forwarding queue capacity must be rejected");

    bool rejected_jpeg = false;
    {
      std::ofstream invalid(config);
      invalid << "mode = \"decode\"\nmanifest = \"" << argv[1]
              << "\"\noutput_transport_mode = \"jpeg_async\"\n";
      invalid.close();
      try {
        (void)mlvc::LoadDecoderConfig(config);
      } catch (const std::exception&) {
        rejected_jpeg = true;
      }
    }
    mlvc::Check(rejected_jpeg, "JPEG forwarding mode must be rejected");

    bool rejected_rtsp_without_url = false;
    {
      std::ofstream invalid(config);
      invalid << "mode = \"decode\"\nmanifest = \"" << argv[1]
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
      invalid << "mode = \"decode\"\nmanifest = \"" << argv[1]
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

    for (const std::string& invalid_execution :
         {"rans-convert-pipeline", "rans-narrow-convert-pipeline", "unknown"}) {
      bool rejected_entropy_execution = false;
      std::ofstream invalid(config);
      invalid << "mode = \"decode\"\nmanifest = \"" << argv[1]
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
    std::cout << "decode config test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_decode_config failed: " << error.what() << "\n";
    return 1;
  }
}
