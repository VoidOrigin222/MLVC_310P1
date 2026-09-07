#ifndef MLVC_APPLICATION_CLI_COMMAND_LINE_H_
#define MLVC_APPLICATION_CLI_COMMAND_LINE_H_

#include <filesystem>

namespace mlvc {

std::filesystem::path ParseConfigPath(int argc, char** argv);

}  // namespace mlvc

#endif  // MLVC_APPLICATION_CLI_COMMAND_LINE_H_
