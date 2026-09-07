#ifndef MLVC_CORE_STATUS_H_
#define MLVC_CORE_STATUS_H_

#include <stdexcept>
#include <string>

namespace mlvc {

class Error : public std::runtime_error {
 public:
  explicit Error(const std::string& message);
};

void Check(bool condition, const std::string& message);

}  // namespace mlvc

#endif  // MLVC_CORE_STATUS_H_
