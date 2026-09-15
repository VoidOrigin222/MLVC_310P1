#ifndef MLVC_RUNTIME_MODEL_MANIFEST_H_
#define MLVC_RUNTIME_MODEL_MANIFEST_H_

#include <filesystem>
#include <initializer_list>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "mlvc/core/types.h"

namespace mlvc {

struct TensorSpec {
  std::string name;
  DataType dtype = DataType::kFloat16;
  std::vector<int64_t> shape;
};

struct ModelRecord {
  std::string name;
  std::string frame_type;
  std::string route;
  std::string backend;
  std::filesystem::path model;
  std::filesystem::path onnx_model;
  std::filesystem::path optimized_onnx_model;
  std::filesystem::path atc_model;
  uint64_t bytes = 0;
  std::string sha256;
  std::vector<std::string> replaces;
  std::vector<TensorSpec> inputs;
  std::vector<TensorSpec> outputs;
};

struct SidecarRecord {
  std::filesystem::path file;
  uint64_t bytes = 0;
  std::string sha256;
};

class ModelManifest {
 public:
  static ModelManifest Load(const std::filesystem::path& manifest_path);

  const std::filesystem::path& path() const { return path_; }
  const std::filesystem::path& directory() const { return directory_; }
  const std::string& runtime() const { return runtime_; }
  const std::string& soc_version() const { return soc_version_; }
  const std::string& dtype() const { return dtype_; }
  const SidecarRecord& sidecar() const { return sidecar_; }
  const std::vector<ModelRecord>& models() const { return models_; }
  std::vector<ModelRecord>& models() { return models_; }
  bool HasModel(const std::string& name) const;
  const ModelRecord* FindModel(const std::string& name) const;
  const ModelRecord* FindFusionCandidate(std::initializer_list<std::string_view> replaces) const;
  const ModelRecord& GetModel(const std::string& name) const;

 private:
  std::filesystem::path path_;
  std::filesystem::path directory_;
  std::string runtime_;
  std::string soc_version_;
  std::string dtype_;
  SidecarRecord sidecar_;
  std::vector<ModelRecord> models_;
  std::map<std::string, std::size_t> model_index_;
};

}  // namespace mlvc

#endif  // MLVC_RUNTIME_MODEL_MANIFEST_H_
