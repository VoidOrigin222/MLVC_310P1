#ifndef MLVC_FRAMEWORK_DATA_OBJECT_H_
#define MLVC_FRAMEWORK_DATA_OBJECT_H_

#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>

namespace mlvc {

// Base type passed between pipeline nodes. Applications can derive FramePacket
// or another domain object without coupling the framework to codec details.
class DataObject {
 public:
  virtual ~DataObject() = default;

  virtual std::type_index type() const { return std::type_index(typeid(*this)); }
  virtual std::string type_name() const { return typeid(*this).name(); }

  template <typename T>
  T* As() {
    return dynamic_cast<T*>(this);
  }

  template <typename T>
  const T* As() const {
    return dynamic_cast<const T*>(this);
  }

  template <typename T>
  bool Is() const {
    return As<T>() != nullptr;
  }
};

}  // namespace mlvc

#endif  // MLVC_FRAMEWORK_DATA_OBJECT_H_
