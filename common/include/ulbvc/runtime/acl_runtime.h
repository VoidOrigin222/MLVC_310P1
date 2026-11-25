#ifndef ULBVC_RUNTIME_ACL_RUNTIME_H_
#define ULBVC_RUNTIME_ACL_RUNTIME_H_

#include <acl/acl.h>

#include <cstddef>
#include <filesystem>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "ulbvc/core/tensor.h"
#include "ulbvc/runtime/model_manifest.h"

namespace ulbvc {

struct NamedTensorView {
  const char* name = nullptr;
  TensorView view;
};

void CheckAcl(aclError status, const std::string& operation);
void BootstrapAclEnvironment();
void ValidateAclManifestForAclRuntime(const ModelManifest& manifest);

class AclRuntime {
 public:
  explicit AclRuntime(int device_id);
  // Owns a separate stream while borrowing the codec's live context/device.
  explicit AclRuntime(aclrtContext borrowed_context);
  ~AclRuntime();

  AclRuntime(const AclRuntime&) = delete;
  AclRuntime& operator=(const AclRuntime&) = delete;

  int device_id() const { return device_id_; }
  aclrtContext context() const { return context_; }
  aclrtStream stream() const { return stream_; }
  void MakeCurrent() const;

 private:
  int device_id_ = 0;
  aclrtContext context_ = nullptr;
  aclrtStream stream_ = nullptr;
  bool initialized_ = false;
  bool device_set_ = false;
  bool owns_context_ = true;
};

class AclStage {
 public:
  AclStage(AclRuntime* runtime, ModelRecord record, std::filesystem::path model_path,
           bool async_execution = false);
  ~AclStage();

  AclStage(const AclStage&) = delete;
  AclStage& operator=(const AclStage&) = delete;

  const ModelRecord& record() const { return record_; }

  void Run(const std::map<std::string, TensorView>& inputs,
           const std::map<std::string, TensorView>& outputs);
  void Run(std::initializer_list<NamedTensorView> inputs,
           std::initializer_list<NamedTensorView> outputs);
  void RunNamed(const NamedTensorView* inputs, std::size_t input_count,
                const NamedTensorView* outputs, std::size_t output_count);

 private:
  void Cleanup() noexcept;
  struct DeviceBuffer {
    void* data = nullptr;
    std::size_t bytes = 0;
  };

  static void AllocateDeviceBuffer(DeviceBuffer* buffer, std::size_t bytes);
  static void FreeDeviceBuffer(DeviceBuffer* buffer);
  static TensorSpec ReadInputSpec(aclmdlDesc* desc, std::size_t index);
  static TensorSpec ReadOutputSpec(aclmdlDesc* desc, std::size_t index);

  void BuildDatasets();
  void DestroyDataset(aclmdlDataset*& dataset, std::vector<aclDataBuffer*>& data_buffers);
  const NamedTensorView* FindNamedView(const NamedTensorView* views, std::size_t count,
                                       const std::string& name) const;

  AclRuntime* runtime_ = nullptr;
  ModelRecord record_;
  uint32_t model_id_ = 0;
  bool model_loaded_ = false;
  bool async_execution_ = false;
  aclmdlDesc* desc_ = nullptr;
  aclmdlDataset* input_dataset_ = nullptr;
  aclmdlDataset* output_dataset_ = nullptr;
  std::vector<DeviceBuffer> input_buffers_;
  std::vector<DeviceBuffer> output_buffers_;
  std::vector<aclDataBuffer*> input_data_buffers_;
  std::vector<aclDataBuffer*> output_data_buffers_;
};

class AclModelSet {
 public:
  AclModelSet(AclRuntime* runtime, ModelManifest manifest);

  const ModelManifest& manifest() const { return manifest_; }
  bool HasStage(const std::string& name) const;
  AclStage& GetStage(const std::string& name);
  AclStage& GetStage(std::string_view name);

 private:
  ModelManifest manifest_;
  std::map<std::string, std::unique_ptr<AclStage>> stages_;
};

}  // namespace ulbvc

#endif  // ULBVC_RUNTIME_ACL_RUNTIME_H_
