#include <ulbvc/application/cli/command_line.h>

#include <string>

#include "ulbvc/core/status.h"

namespace ulbvc {

std::filesystem::path ParseConfigPath(int argc, char** argv) {
  Check(argc == 3 && std::string(argv[1]) == "--config",
        "usage: ulbvc application --config <config.toml>");
  return std::filesystem::path(argv[2]);
}

}  // namespace ulbvc
