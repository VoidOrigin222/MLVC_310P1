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
    const bool device_dvpp_mode = config.stream.forward_port > 0 &&
                                  config.stream.forward_mode == "dvpp_jpeg_device_async";
    auto codec_runtime = std::make_unique<mlvc::app::MlvcCodecRuntime>(
        config.stream.manifest_path, config.stream.device,
        device_dvpp_mode ? mlvc::codec::StageOutputBindingMode::kAclMirror
                         : mlvc::codec::StageOutputBindingMode::kCpu);
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
