#ifndef ULBVC_APPLICATION_CLI_ENCODER_APP_H_
#define ULBVC_APPLICATION_CLI_ENCODER_APP_H_

#include <ulbvc/application/cli/command_line.h>
#include <ulbvc/application/stream/ulbvc_stream.h>
#include <ulbvc/application/stream/stream_encoder.h>

#include <filesystem>

namespace ulbvc {

struct EncoderApplicationConfig {
  codec::EncodeStreamOptions stream;
};

EncoderApplicationConfig LoadEncoderConfig(const std::filesystem::path& config_path);

}  // namespace ulbvc

#endif  // ULBVC_APPLICATION_CLI_ENCODER_APP_H_
