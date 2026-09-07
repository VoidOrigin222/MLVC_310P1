#ifndef MLVC_APPLICATION_CLI_DECODER_APP_H_
#define MLVC_APPLICATION_CLI_DECODER_APP_H_

#include <filesystem>
#include <string>

#include <mlvc/application/cli/command_line.h>
#include <mlvc/application/stream/mlvc_stream.h>

namespace mlvc {

struct DecoderApplicationConfig {
  codec::DecodeStreamOptions stream;
};

DecoderApplicationConfig LoadDecoderConfig(const std::filesystem::path& config_path);

}  // namespace mlvc

#endif  // MLVC_APPLICATION_CLI_DECODER_APP_H_
