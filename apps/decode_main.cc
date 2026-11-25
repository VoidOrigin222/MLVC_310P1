#include <ulbvc/application/cli/decoder_app.h>
#include <ulbvc/application/cli/runtime_environment.h>

#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>

int main(int argc, char** argv) {
  try {
    // Keep process setup, configuration, and application execution explicit.
    ulbvc::PrepareAscendRuntimeEnvironment(argv);
    const std::filesystem::path config_path = ulbvc::ParseConfigPath(argc, argv);
    const ulbvc::DecoderApplicationConfig config = ulbvc::LoadDecoderConfig(config_path);
    const bool use_device_resident_outputs = config.stream.output_format == "none";
    const auto output_binding_mode =
        use_device_resident_outputs
            ? ulbvc::codec::StageOutputBindingMode::kAclDeviceWithCpuMirror
            : ulbvc::codec::StageOutputBindingMode::kCpu;
    auto codec_runtime = std::make_unique<ulbvc::app::UlbvcCodecRuntime>(
        config.stream.manifest_path, config.stream.device, output_binding_mode);
    ulbvc::Profiler profiler;
    ulbvc::CodecGraphExecutor graph_executor(config.stream.pipeline.graph_packet_capacity);
    ulbvc::EntropyWorker entropy_worker(config.stream.pipeline.entropy_workers);
    ulbvc::codec::DecodePipelineServices services{&profiler, &graph_executor, &entropy_worker,
                                                 codec_runtime.get()};
    return ulbvc::codec::RunDecodeStream(config.stream, &services);
  } catch (const std::exception& error) {
    std::cerr << argv[0] << " failed: " << error.what() << "\n";
    return 1;
  }
}
