#include "mlvc/runtime/acl_runtime.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>

#include "mlvc/core/status.h"
#include "mlvc/framework/profile_range.h"

namespace mlvc {
namespace {

DataType MlvcDataType(aclDataType dtype) {
  switch (dtype) {
    case ACL_FLOAT16:
      return DataType::kFloat16;
    case ACL_FLOAT:
      return DataType::kFloat32;
    case ACL_INT8:
      return DataType::kInt8;
    case ACL_INT16:
      return DataType::kInt16;
    case ACL_INT32:
      return DataType::kInt32;
    case ACL_UINT8:
      return DataType::kUInt8;
    default:
      throw Error("unsupported ACL tensor dtype: " + std::to_string(static_cast<int>(dtype)));
  }
}

std::vector<int64_t> DimsVector(const aclmdlIODims& dims) {
  return std::vector<int64_t>(dims.dims, dims.dims + dims.dimCount);
}

aclrtMemcpyKind InputCopyKind(MemoryLocation location) {
  switch (location) {
    case MemoryLocation::kCpu:
    case MemoryLocation::kPinnedCpu:
      return ACL_MEMCPY_HOST_TO_DEVICE;
    case MemoryLocation::kAcl:
      return ACL_MEMCPY_DEVICE_TO_DEVICE;
  }
  throw Error("unsupported ACL input memory location");
}

aclrtMemcpyKind OutputCopyKind(MemoryLocation location) {
  switch (location) {
    case MemoryLocation::kCpu:
    case MemoryLocation::kPinnedCpu:
      return ACL_MEMCPY_DEVICE_TO_HOST;
    case MemoryLocation::kAcl:
      return ACL_MEMCPY_DEVICE_TO_DEVICE;
  }
  throw Error("unsupported ACL output memory location");
}

void CheckTensorCompatible(const TensorSpec& spec, const TensorView& view,
                           const std::string& stage_name) {
  Check(view.dtype() == spec.dtype, "ACL stage input/output dtype mismatch: " + stage_name + "." +
                                        spec.name + " expected " + DataTypeName(spec.dtype) +
                                        " got " + DataTypeName(view.dtype()));
  Check(view.shape().NumElements() == TensorShape(spec.shape).NumElements(),
        "ACL stage input/output element count mismatch: " + stage_name + "." + spec.name);
}

void AppendEnvPath(const char* name, const std::filesystem::path& path) {
  std::error_code error;
  if (!std::filesystem::exists(path, error)) {
    return;
  }
  const std::string value = path.string();
  const char* existing = std::getenv(name);
  if (existing == nullptr || existing[0] == '\0') {
    setenv(name, value.c_str(), 1);
    return;
  }
  const std::string current(existing);
  const bool is_suffix = current.size() > value.size() &&
                         current.compare(current.size() - value.size(), value.size(), value) == 0 &&
                         current[current.size() - value.size() - 1] == ':';
  if (current == value || current.find(value + ":") == 0 ||
      current.find(":" + value + ":") != std::string::npos || is_suffix) {
    return;
  }
  const std::string combined = value + ":" + current;
  setenv(name, combined.c_str(), 1);
}

std::filesystem::path RepoRootFromEnvironmentOrCwd() {
  const char* repo_root = std::getenv("MLVC_ACL_REPO_ROOT");
  if (repo_root != nullptr && repo_root[0] != '\0') {
    return repo_root;
  }
  std::error_code error;
  return std::filesystem::current_path(error);
}

std::filesystem::path CannHomeFromEnvironment() {
  const char* cann_home = std::getenv("CANN_HOME");
  if (cann_home != nullptr && cann_home[0] != '\0') {
    return cann_home;
  }
  const char* ascend_home = std::getenv("ASCEND_HOME_PATH");
  if (ascend_home != nullptr && ascend_home[0] != '\0') {
    return ascend_home;
  }
  return "/usr/local/Ascend/cann";
}

void SetEnvIfUnset(const char* name, const std::filesystem::path& value) {
  if (std::getenv(name) == nullptr) {
    setenv(name, value.c_str(), 1);
  }
}

void ValidateAclTensorSpecs(const std::vector<TensorSpec>& specs, const std::string& record_name,
                            const char* label) {
  Check(!specs.empty(), "ACL runtime requires non-empty " + std::string(label) +
                            " specs for model record: " + record_name);
  for (const TensorSpec& spec : specs) {
    Check(!spec.name.empty(), "ACL runtime requires named " + std::string(label) +
                                  " tensors for model record: " + record_name);
    Check(spec.dtype == DataType::kFloat16 || spec.dtype == DataType::kInt32,
          "ACL runtime requires FP16 or INT32 " + std::string(label) +
              " tensors for model record: " + record_name + "." + spec.name);
    Check(!spec.shape.empty(), "ACL runtime requires static " + std::string(label) +
                                   " tensor shapes for model record: " + record_name + "." +
                                   spec.name);
    for (int64_t dim : spec.shape) {
      Check(dim > 0, "ACL runtime requires positive " + std::string(label) +
                         " tensor shapes for model record: " + record_name + "." + spec.name);
    }
  }
}

}  // namespace

void CheckAcl(aclError status, const std::string& operation) {
  if (status != ACL_ERROR_NONE) {
    throw Error(operation + " failed: ret=" + std::to_string(status));
  }
}

void BootstrapAclEnvironment() {
  static std::once_flag once;
  std::call_once(once, [] {
    const std::filesystem::path repo_root = RepoRootFromEnvironmentOrCwd();
    const std::filesystem::path cann_home = CannHomeFromEnvironment();
    SetEnvIfUnset("CANN_HOME", cann_home);
    SetEnvIfUnset("ASCEND_HOME_PATH", cann_home);
    SetEnvIfUnset("ASCEND_TOOLKIT_HOME", cann_home);
    SetEnvIfUnset("ASCEND_AICPU_PATH", cann_home);
    SetEnvIfUnset("ASCEND_OPP_PATH", cann_home / "opp");
    SetEnvIfUnset("TOOLCHAIN_HOME", cann_home / "toolkit");
    setenv("MLVC_ACL_REPO_ROOT", repo_root.c_str(), 0);
    const std::filesystem::path custom_opp_root = repo_root / "output" / "custom_opp";
    AppendEnvPath("ASCEND_CUSTOM_OPP_PATH", custom_opp_root / "wsiluchunkadd" / "vendors" / "mlvc");
    AppendEnvPath("ASCEND_CUSTOM_OPP_PATH",
                  custom_opp_root / "mlvc_prior_ops" / "vendors" / "mlvc");

    const std::filesystem::path prior_opapi = custom_opp_root / "mlvc_prior_ops" / "vendors" /
                                              "mlvc" / "op_api" / "lib" / "libcust_opapi.so";
    std::error_code error;
    if (std::getenv("MLVC_PRIOR_OPAPI_LIB") == nullptr &&
        std::filesystem::is_regular_file(prior_opapi, error)) {
      setenv("MLVC_PRIOR_OPAPI_LIB", prior_opapi.c_str(), 1);
    }
  });
}

void ValidateAclManifestForAclRuntime(const ModelManifest& manifest) {
  Check(manifest.runtime() == "acl", "ACL runtime requires a manifest with runtime = \"acl\"");
  Check(!manifest.soc_version().empty(), "ACL runtime requires an ACL manifest soc_version");
  Check(manifest.dtype() == "fp16" || manifest.dtype() == "float16",
        "ACL runtime requires ACL manifest dtype = fp16");
  for (const ModelRecord& record : manifest.models()) {
    Check(record.backend == "acl",
          "ACL runtime requires backend = \"acl\" for model record: " + record.name);
    Check(record.model.extension() == ".om",
          "ACL runtime requires OM model files for model record: " + record.name);
    ValidateAclTensorSpecs(record.inputs, record.name, "input");
    ValidateAclTensorSpecs(record.outputs, record.name, "output");
  }
}

AclRuntime::AclRuntime(int device_id) : device_id_(device_id) {
  try {
    BootstrapAclEnvironment();
    CheckAcl(aclInit(nullptr), "aclInit");
    initialized_ = true;
    CheckAcl(aclrtSetDevice(device_id_), "aclrtSetDevice");
    device_set_ = true;
    CheckAcl(aclrtCreateContext(&context_, device_id_), "aclrtCreateContext");
    CheckAcl(aclrtCreateStream(&stream_), "aclrtCreateStream");
  } catch (...) {
    if (stream_ != nullptr) (void)aclrtDestroyStream(stream_);
    if (context_ != nullptr) (void)aclrtDestroyContext(context_);
    if (device_set_) (void)aclrtResetDevice(device_id_);
    if (initialized_) (void)aclFinalize();
    stream_ = nullptr;
    context_ = nullptr;
    device_set_ = false;
    initialized_ = false;
    throw;
  }
}

AclRuntime::~AclRuntime() {
  if (stream_ != nullptr) {
    (void)aclrtDestroyStream(stream_);
    stream_ = nullptr;
  }
  if (context_ != nullptr) {
    (void)aclrtDestroyContext(context_);
    context_ = nullptr;
  }
  if (device_set_) {
    (void)aclrtResetDevice(device_id_);
  }
  if (initialized_) {
    (void)aclFinalize();
  }
}

void AclRuntime::MakeCurrent() const {
  CheckAcl(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext");
}

AclStage::AclStage(AclRuntime* runtime, ModelRecord record, std::filesystem::path model_path)
    : runtime_(runtime), record_(std::move(record)) {
  try {
    Check(runtime_ != nullptr, "ACL runtime is required");
    Check(std::filesystem::exists(model_path),
          "ACL OM file does not exist: " + model_path.string());
    CheckAcl(aclrtSetCurrentContext(runtime_->context()), "aclrtSetCurrentContext");
    CheckAcl(aclmdlLoadFromFile(model_path.c_str(), &model_id_),
             "aclmdlLoadFromFile " + model_path.string());
    model_loaded_ = true;
    desc_ = aclmdlCreateDesc();
    Check(desc_ != nullptr, "aclmdlCreateDesc failed");
    CheckAcl(aclmdlGetDesc(desc_, model_id_), "aclmdlGetDesc");

    const std::size_t om_input_count = aclmdlGetNumInputs(desc_);
    const std::size_t om_output_count = aclmdlGetNumOutputs(desc_);
    if (record_.inputs.empty()) {
      record_.inputs.reserve(om_input_count);
      for (std::size_t index = 0; index < om_input_count; ++index) {
        record_.inputs.push_back(ReadInputSpec(desc_, index));
      }
    } else {
      Check(record_.inputs.size() == om_input_count,
            "ACL manifest input count does not match OM desc: " + record_.name);
    }
    if (record_.outputs.empty()) {
      record_.outputs.reserve(om_output_count);
      for (std::size_t index = 0; index < om_output_count; ++index) {
        record_.outputs.push_back(ReadOutputSpec(desc_, index));
      }
    } else {
      Check(record_.outputs.size() == om_output_count,
            "ACL manifest output count does not match OM desc: " + record_.name);
    }
    BuildDatasets();
  } catch (...) {
    Cleanup();
    throw;
  }
}

void AclStage::Cleanup() noexcept {
  if (runtime_ != nullptr) {
    (void)aclrtSetCurrentContext(runtime_->context());
  }
  DestroyDataset(input_dataset_, input_data_buffers_);
  DestroyDataset(output_dataset_, output_data_buffers_);
  for (DeviceBuffer& buffer : input_buffers_) {
    FreeDeviceBuffer(&buffer);
  }
  for (DeviceBuffer& buffer : output_buffers_) {
    FreeDeviceBuffer(&buffer);
  }
  if (desc_ != nullptr) {
    (void)aclmdlDestroyDesc(desc_);
    desc_ = nullptr;
  }
  if (model_loaded_) {
    (void)aclmdlUnload(model_id_);
    model_loaded_ = false;
  }
}

AclStage::~AclStage() { Cleanup(); }

void AclStage::AllocateDeviceBuffer(DeviceBuffer* buffer, std::size_t bytes) {
  Check(buffer != nullptr, "ACL device buffer is required");
  FreeDeviceBuffer(buffer);
  if (bytes == 0) {
    return;
  }
  CheckAcl(aclrtMalloc(&buffer->data, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
  buffer->bytes = bytes;
}

void AclStage::FreeDeviceBuffer(DeviceBuffer* buffer) {
  if (buffer != nullptr && buffer->data != nullptr) {
    (void)aclrtFree(buffer->data);
    buffer->data = nullptr;
    buffer->bytes = 0;
  }
}

TensorSpec AclStage::ReadInputSpec(aclmdlDesc* desc, std::size_t index) {
  TensorSpec spec;
  const char* name = aclmdlGetInputNameByIndex(desc, index);
  spec.name = name != nullptr ? name : "input_" + std::to_string(index);
  spec.dtype = MlvcDataType(aclmdlGetInputDataType(desc, index));
  aclmdlIODims dims{};
  CheckAcl(aclmdlGetInputDims(desc, index, &dims), "aclmdlGetInputDims");
  spec.shape = DimsVector(dims);
  return spec;
}

TensorSpec AclStage::ReadOutputSpec(aclmdlDesc* desc, std::size_t index) {
  TensorSpec spec;
  const char* name = aclmdlGetOutputNameByIndex(desc, index);
  spec.name = name != nullptr ? name : "output_" + std::to_string(index);
  spec.dtype = MlvcDataType(aclmdlGetOutputDataType(desc, index));
  aclmdlIODims dims{};
  CheckAcl(aclmdlGetOutputDims(desc, index, &dims), "aclmdlGetOutputDims");
  spec.shape = DimsVector(dims);
  return spec;
}

void AclStage::BuildDatasets() {
  input_dataset_ = aclmdlCreateDataset();
  output_dataset_ = aclmdlCreateDataset();
  Check(input_dataset_ != nullptr, "aclmdlCreateDataset input failed");
  Check(output_dataset_ != nullptr, "aclmdlCreateDataset output failed");
  input_buffers_.resize(record_.inputs.size());
  output_buffers_.resize(record_.outputs.size());
  input_data_buffers_.resize(record_.inputs.size(), nullptr);
  output_data_buffers_.resize(record_.outputs.size(), nullptr);

  for (std::size_t index = 0; index < record_.inputs.size(); ++index) {
    const std::size_t bytes = aclmdlGetInputSizeByIndex(desc_, index);
    AllocateDeviceBuffer(&input_buffers_[index], bytes);
    input_data_buffers_[index] = aclCreateDataBuffer(input_buffers_[index].data, bytes);
    Check(input_data_buffers_[index] != nullptr, "aclCreateDataBuffer input failed");
    CheckAcl(aclmdlAddDatasetBuffer(input_dataset_, input_data_buffers_[index]),
             "aclmdlAddDatasetBuffer input");
  }
  for (std::size_t index = 0; index < record_.outputs.size(); ++index) {
    const std::size_t bytes = aclmdlGetOutputSizeByIndex(desc_, index);
    AllocateDeviceBuffer(&output_buffers_[index], bytes);
    output_data_buffers_[index] = aclCreateDataBuffer(output_buffers_[index].data, bytes);
    Check(output_data_buffers_[index] != nullptr, "aclCreateDataBuffer output failed");
    CheckAcl(aclmdlAddDatasetBuffer(output_dataset_, output_data_buffers_[index]),
             "aclmdlAddDatasetBuffer output");
  }
}

void AclStage::DestroyDataset(aclmdlDataset*& dataset, std::vector<aclDataBuffer*>& data_buffers) {
  for (aclDataBuffer*& buffer : data_buffers) {
    if (buffer != nullptr) {
      (void)aclDestroyDataBuffer(buffer);
      buffer = nullptr;
    }
  }
  data_buffers.clear();
  if (dataset != nullptr) {
    (void)aclmdlDestroyDataset(dataset);
    dataset = nullptr;
  }
}

const NamedTensorView* AclStage::FindNamedView(const NamedTensorView* views, std::size_t count,
                                               const std::string& name) const {
  for (std::size_t index = 0; index < count; ++index) {
    if (views[index].name != nullptr && name == views[index].name) {
      return &views[index];
    }
  }
  return nullptr;
}

void AclStage::Run(const std::map<std::string, TensorView>& inputs,
                   const std::map<std::string, TensorView>& outputs) {
  std::vector<NamedTensorView> input_list;
  std::vector<NamedTensorView> output_list;
  input_list.reserve(inputs.size());
  output_list.reserve(outputs.size());
  for (const auto& [name, view] : inputs) {
    input_list.push_back(NamedTensorView{name.c_str(), view});
  }
  for (const auto& [name, view] : outputs) {
    output_list.push_back(NamedTensorView{name.c_str(), view});
  }
  RunNamed(input_list.data(), input_list.size(), output_list.data(), output_list.size());
}

void AclStage::Run(std::initializer_list<NamedTensorView> inputs,
                   std::initializer_list<NamedTensorView> outputs) {
  RunNamed(inputs.begin(), inputs.size(), outputs.begin(), outputs.size());
}

void AclStage::RunNamed(const NamedTensorView* inputs, std::size_t input_count,
                        const NamedTensorView* outputs, std::size_t output_count) {
  MLVC_PROFILE_RANGE_FUNCTION();
  CheckAcl(aclrtSetCurrentContext(runtime_->context()), "aclrtSetCurrentContext");
  for (std::size_t index = 0; index < record_.inputs.size(); ++index) {
    const TensorSpec& spec = record_.inputs[index];
    const NamedTensorView* input = FindNamedView(inputs, input_count, spec.name);
    Check(input != nullptr,
          "missing input tensor for ACL stage " + record_.name + ": " + spec.name);
    CheckTensorCompatible(spec, input->view, record_.name);
    const std::size_t bytes = input->view.bytes();
    Check(bytes == input_buffers_[index].bytes, "ACL input byte size does not match OM desc");
    if (input->view.location() == MemoryLocation::kAcl) {
      CheckAcl(aclUpdateDataBuffer(input_data_buffers_[index], input->view.data(), bytes),
               "aclUpdateDataBuffer input " + record_.name + "." + spec.name);
    } else {
      CheckAcl(aclUpdateDataBuffer(input_data_buffers_[index], input_buffers_[index].data,
                                   input_buffers_[index].bytes),
               "aclUpdateDataBuffer input staging " + record_.name + "." + spec.name);
      CheckAcl(aclrtMemcpy(input_buffers_[index].data, input_buffers_[index].bytes,
                           input->view.data(), bytes, InputCopyKind(input->view.location())),
               "aclrtMemcpy H2D " + record_.name + "." + spec.name);
    }
  }

  for (std::size_t index = 0; index < record_.outputs.size(); ++index) {
    const TensorSpec& spec = record_.outputs[index];
    const NamedTensorView* output = FindNamedView(outputs, output_count, spec.name);
    Check(output != nullptr,
          "missing output tensor for ACL stage " + record_.name + ": " + spec.name);
    CheckTensorCompatible(spec, output->view, record_.name);
    const std::size_t bytes = output->view.bytes();
    Check(bytes == output_buffers_[index].bytes, "ACL output byte size does not match OM desc");
    if (output->view.location() == MemoryLocation::kAcl) {
      CheckAcl(aclUpdateDataBuffer(output_data_buffers_[index], output->view.data(), bytes),
               "aclUpdateDataBuffer output " + record_.name + "." + spec.name);
    } else {
      CheckAcl(aclUpdateDataBuffer(output_data_buffers_[index], output_buffers_[index].data,
                                   output_buffers_[index].bytes),
               "aclUpdateDataBuffer output staging " + record_.name + "." + spec.name);
    }
  }

  CheckAcl(aclmdlExecute(model_id_, input_dataset_, output_dataset_),
           "aclmdlExecute " + record_.name);

  for (std::size_t index = 0; index < record_.outputs.size(); ++index) {
    const TensorSpec& spec = record_.outputs[index];
    const NamedTensorView* output = FindNamedView(outputs, output_count, spec.name);
    Check(output != nullptr,
          "missing output tensor for ACL stage " + record_.name + ": " + spec.name);
    if (output->view.location() != MemoryLocation::kAcl) {
      const std::size_t bytes = output->view.bytes();
      CheckAcl(aclrtMemcpy(output->view.data(), bytes, output_buffers_[index].data, bytes,
                           OutputCopyKind(output->view.location())),
               "aclrtMemcpy D2H " + record_.name + "." + spec.name);
    }
  }
}

AclModelSet::AclModelSet(AclRuntime* runtime, ModelManifest manifest)
    : manifest_(std::move(manifest)) {
  ValidateAclManifestForAclRuntime(manifest_);
  for (ModelRecord& record : manifest_.models()) {
    const std::filesystem::path model_path =
        record.model.is_absolute() ? record.model : manifest_.directory() / record.model;
    if (!record.replaces.empty() && !std::filesystem::exists(model_path)) {
      continue;
    }
    auto stage = std::make_unique<AclStage>(runtime, record, model_path);
    record.inputs = stage->record().inputs;
    record.outputs = stage->record().outputs;
    stages_[record.name] = std::move(stage);
  }
}

bool AclModelSet::HasStage(const std::string& name) const {
  return stages_.find(name) != stages_.end();
}

AclStage& AclModelSet::GetStage(const std::string& name) {
  auto it = stages_.find(name);
  Check(it != stages_.end(), "unknown ACL stage: " + name);
  return *it->second;
}

AclStage& AclModelSet::GetStage(std::string_view name) {
  for (auto& [stage_name, stage] : stages_) {
    if (stage_name == name) {
      return *stage;
    }
  }
  throw Error("unknown ACL stage: " + std::string(name));
}

}  // namespace mlvc
