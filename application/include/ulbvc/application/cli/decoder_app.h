#ifndef ULBVC_APPLICATION_CLI_DECODER_APP_H_
#define ULBVC_APPLICATION_CLI_DECODER_APP_H_

#include <ulbvc/application/cli/command_line.h>
#include <ulbvc/application/stream/ulbvc_stream.h>

#include <filesystem>
#include <string>

namespace ulbvc {

struct DecoderApplicationConfig {
  codec::DecodeStreamOptions stream;
};

DecoderApplicationConfig LoadDecoderConfig(const std::filesystem::path& config_path);

}  // namespace ulbvc

#endif  // ULBVC_APPLICATION_CLI_DECODER_APP_H_
