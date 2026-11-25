#include <ulbvc/application/cli/encoder_app.h>
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
    const ulbvc::EncoderApplicationConfig config = ulbvc::LoadEncoderConfig(config_path);
    auto codec_runtime = std::make_unique<ulbvc::app::UlbvcCodecRuntime>(
        config.stream.manifest_path, config.stream.device,
        ulbvc::codec::StageOutputBindingMode::kCpu);
    ulbvc::Profiler profiler;
    ulbvc::CodecGraphExecutor graph_executor(config.stream.pipeline.graph_packet_capacity);
    ulbvc::EntropyWorker entropy_worker(config.stream.pipeline.entropy_workers);
    ulbvc::codec::EncodePipelineServices services{&profiler, &graph_executor, &entropy_worker,
                                                 codec_runtime.get()};
    return ulbvc::codec::RunEncodeStream(config.stream, &services);
  } catch (const std::exception& error) {
    std::cerr << argv[0] << " failed: " << error.what() << "\n";
    return 1;
  }
}
