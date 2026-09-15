#include <mlvc/application/cli/command_line.h>

#include <string>

#include "mlvc/core/status.h"

namespace mlvc {

std::filesystem::path ParseConfigPath(int argc, char** argv) {
  Check(argc == 3 && std::string(argv[1]) == "--config",
        "usage: mlvc application --config <config.toml>");
  return std::filesystem::path(argv[2]);
}

}  // namespace mlvc
