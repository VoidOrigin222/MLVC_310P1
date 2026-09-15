#include "mlvc/core/status.h"

namespace mlvc {

Error::Error(const std::string& message) : std::runtime_error(message) {}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    throw Error(message);
  }
}

}  // namespace mlvc
