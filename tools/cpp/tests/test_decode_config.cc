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
           << "output_transport_host = \"127.0.0.1\"\n"
           << "output_transport_port = 50000\n"
           << "output_transport_mode = \"dvpp_jpeg_device_async\"\n"
           << "output_transport_queue_capacity = 5\n"
           << "output_transport_jpeg_quality = 80\n"
           << "manifest = \"" << argv[1] << "\"\n"
           << "[pipeline]\n"
           << "entropy_workers = 2\n";
    output.close();
    mlvc::Check(output.good(), "failed to write decode config fixture");
    const mlvc::DecoderApplicationConfig parsed = mlvc::LoadDecoderConfig(config);
    mlvc::Check(parsed.stream.forward_mode == "dvpp_jpeg_device_async",
                "device-resident DVPP forward mode was not preserved");
    mlvc::Check(parsed.stream.forward_queue_capacity == 5,
                "DVPP queue capacity was not preserved");
    mlvc::Check(parsed.stream.forward_jpeg_quality == 80,
                "DVPP JPEG quality was not preserved");
    mlvc::Check(parsed.stream.udp_port == 39089,
                "input transport port was not preserved");
    mlvc::Check(parsed.stream.pipeline.entropy_workers == 2,
                "configured entropy worker count was not preserved");

    bool rejected_capacity = false;
    {
      std::ofstream invalid(config);
      invalid << "mode = \"decode\"\nmanifest = \"" << argv[1]
              << "\"\nforward_queue_capacity = 0\n";
      invalid.close();
      try {
        (void)mlvc::LoadDecoderConfig(config);
      } catch (const std::exception&) {
        rejected_capacity = true;
      }
    }
    mlvc::Check(rejected_capacity, "zero forwarding queue capacity must be rejected");

    bool rejected_quality = false;
    {
      std::ofstream invalid(config);
      invalid << "mode = \"decode\"\nmanifest = \"" << argv[1]
              << "\"\nforward_jpeg_quality = 101\n";
      invalid.close();
      try {
        (void)mlvc::LoadDecoderConfig(config);
      } catch (const std::exception&) {
        rejected_quality = true;
      }
    }
    mlvc::Check(rejected_quality, "JPEG quality above 100 must be rejected");

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
