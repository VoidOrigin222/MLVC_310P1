#ifndef ULBVC_APPLICATION_CLI_COMMAND_LINE_H_
#define ULBVC_APPLICATION_CLI_COMMAND_LINE_H_

#include <filesystem>

namespace ulbvc {

std::filesystem::path ParseConfigPath(int argc, char** argv);

}  // namespace ulbvc

#endif  // ULBVC_APPLICATION_CLI_COMMAND_LINE_H_
