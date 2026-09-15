#include <mlvc/application/cli/decoder_app.h>
#include <mlvc/application/cli/runtime_environment.h>

#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>

int main(int argc, char** argv) {
  try {
    // Keep process setup, configuration, and application execution explicit.
    mlvc::PrepareAscendRuntimeEnvironment(argv);
    const std::filesystem::path config_path = mlvc::ParseConfigPath(argc, argv);
    const mlvc::DecoderApplicationConfig config = mlvc::LoadDecoderConfig(config_path);
    const bool use_device_resident_outputs =
        config.stream.output_format == "none" && config.stream.output_transport_port == 0 &&
        (config.stream.output_transport_mode == "none" ||
         config.stream.output_transport_mode == "rtsp");
    const auto output_binding_mode =
        use_device_resident_outputs
            ? mlvc::codec::StageOutputBindingMode::kAclDeviceWithCpuMirror
            : mlvc::codec::StageOutputBindingMode::kCpu;
    auto codec_runtime = std::make_unique<mlvc::app::MlvcCodecRuntime>(
        config.stream.manifest_path, config.stream.device, output_binding_mode);
    mlvc::Profiler profiler;
    mlvc::CodecGraphExecutor graph_executor(config.stream.pipeline.graph_packet_capacity);
    mlvc::EntropyWorker entropy_worker(config.stream.pipeline.entropy_workers);
    mlvc::codec::DecodePipelineServices services{&profiler, &graph_executor, &entropy_worker,
                                                 codec_runtime.get()};
    return mlvc::codec::RunDecodeStream(config.stream, &services);
  } catch (const std::exception& error) {
    std::cerr << argv[0] << " failed: " << error.what() << "\n";
    return 1;
  }
}
