#ifndef ULBVC_CORE_STATUS_H_
#define ULBVC_CORE_STATUS_H_

#include <stdexcept>
#include <string>

namespace ulbvc {

class Error : public std::runtime_error {
 public:
  explicit Error(const std::string& message);
};

void Check(bool condition, const std::string& message);

}  // namespace ulbvc

#endif  // ULBVC_CORE_STATUS_H_
