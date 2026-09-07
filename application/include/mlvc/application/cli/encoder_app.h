#ifndef MLVC_APPLICATION_CLI_ENCODER_APP_H_
#define MLVC_APPLICATION_CLI_ENCODER_APP_H_

#include <filesystem>

#include <mlvc/application/cli/command_line.h>
#include <mlvc/application/stream/mlvc_stream.h>
#include <mlvc/application/stream/stream_encoder.h>

namespace mlvc {

struct EncoderApplicationConfig {
  codec::EncodeStreamOptions stream;
};

EncoderApplicationConfig LoadEncoderConfig(const std::filesystem::path& config_path);

}  // namespace mlvc

#endif  // MLVC_APPLICATION_CLI_ENCODER_APP_H_
