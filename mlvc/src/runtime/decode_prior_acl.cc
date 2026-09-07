#include "mlvc/runtime/decode_prior_acl.h"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <dlfcn.h>

#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "mlvc/core/buffer.h"
#include "mlvc/core/status.h"

namespace mlvc {
namespace {

using ExecuteFn = aclnnStatus (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
using SinglePartWorkspaceFn = aclnnStatus (*)(const aclTensor*, int64_t, int64_t, int64_t, int64_t,
                                              const aclTensor*, uint64_t*, aclOpExecutor**);
using RestoreYWorkspaceFn = aclnnStatus (*)(const aclTensor*, const aclTensor*, int64_t, int64_t,
                                            int64_t, int64_t, const aclTensor*, uint64_t*,
                                            aclOpExecutor**);
using ApplyQuantWorkspaceFn = aclnnStatus (*)(const aclTensor*, const aclTensor*, int64_t, int64_t,
                                              const aclTensor*, uint64_t*, aclOpExecutor**);
using Int8ToFp16WorkspaceFn = aclnnStatus (*)(const aclTensor*, int64_t, int64_t, const aclTensor*,
                                              uint64_t*, aclOpExecutor**);

struct AclTensorDeleter {
  void operator()(const aclTensor* tensor) const {
    if (tensor != nullptr) {
      (void)aclDestroyTensor(tensor);
    }
  }
};

using TensorPtr = std::unique_ptr<const aclTensor, AclTensorDeleter>;

struct DecodePriorAclApi {
  void* handle = nullptr;
  SinglePartWorkspaceFn single_part_workspace = nullptr;
  ExecuteFn single_part_execute = nullptr;
  RestoreYWorkspaceFn restore_y_workspace = nullptr;
  ExecuteFn restore_y_execute = nullptr;
  ApplyQuantWorkspaceFn apply_quant_workspace = nullptr;
  ExecuteFn apply_quant_execute = nullptr;
  Int8ToFp16WorkspaceFn int8_to_fp16_workspace = nullptr;
  ExecuteFn int8_to_fp16_execute = nullptr;
  bool attempted = false;
  std::string error;
};

DecodePriorAclApi& ApiState() {
  static DecodePriorAclApi state;
  return state;
}

std::mutex& ApiMutex() {
  static std::mutex mutex;
  return mutex;
}

aclDataType AclDataTypeFor(DataType dtype) {
  switch (dtype) {
    case DataType::kFloat16:
      return ACL_FLOAT16;
    case DataType::kFloat32:
      return ACL_FLOAT;
    case DataType::kInt8:
      return ACL_INT8;
    case DataType::kInt16:
      return ACL_INT16;
    case DataType::kUInt8:
      return ACL_UINT8;
    case DataType::kInt32:
      return ACL_INT32;
  }
  throw Error("unsupported ACL dtype");
}

TensorPtr MakeAclTensor(const int64_t* shape, std::size_t rank, DataType dtype, void* data) {
  Check(shape != nullptr || rank == 0, "ACL tensor shape is required");
  std::vector<int64_t> strides(rank, 1);
  for (std::size_t i = rank; i > 1; --i) {
    strides[i - 2] = strides[i - 1] * shape[i - 1];
  }
  aclTensor* tensor =
      aclCreateTensor(shape, static_cast<uint64_t>(rank), AclDataTypeFor(dtype), strides.data(), 0,
                      ACL_FORMAT_ND, shape, static_cast<uint64_t>(rank), data);
  Check(tensor != nullptr, "aclCreateTensor failed");
  return TensorPtr(tensor);
}

void CheckAclStatus(aclError status, const char* operation) {
  if (status != ACL_ERROR_NONE) {
    throw Error(std::string(operation) + " failed: ret=" + std::to_string(status));
  }
}

void CheckAclnnStatus(aclnnStatus status, const char* operation) {
  if (status != OK) {
    throw Error(std::string(operation) + " failed: ret=" + std::to_string(status));
  }
}

std::vector<std::filesystem::path> CandidateLibraries() {
  std::vector<std::filesystem::path> paths;
  const char* env_path = std::getenv("MLVC_PRIOR_OPAPI_LIB");
  if (env_path != nullptr && *env_path != '\0') {
    paths.emplace_back(env_path);
  }
  const char* repo_root = std::getenv("MLVC_ACL_REPO_ROOT");
  if (repo_root != nullptr && *repo_root != '\0') {
    paths.emplace_back(
        std::filesystem::path(repo_root) /
        "output/custom_opp/mlvc_prior_ops/vendors/ulbvc/op_api/lib/libcust_opapi.so");
  }
  paths.emplace_back("libcust_opapi.so");
  return paths;
}

template <typename Fn>
Fn LoadSymbol(void* handle, const char* name) {
  return reinterpret_cast<Fn>(dlsym(handle, name));
}

bool LoadApi() {
  std::lock_guard<std::mutex> lock(ApiMutex());
  DecodePriorAclApi& state = ApiState();
  if (state.attempted) {
    return state.single_part_workspace != nullptr && state.single_part_execute != nullptr &&
           state.restore_y_workspace != nullptr && state.restore_y_execute != nullptr &&
           state.apply_quant_workspace != nullptr && state.apply_quant_execute != nullptr &&
           state.int8_to_fp16_workspace != nullptr && state.int8_to_fp16_execute != nullptr;
  }
  state.attempted = true;
  for (const std::filesystem::path& candidate : CandidateLibraries()) {
    void* handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      continue;
    }
    auto* single_part_workspace =
        LoadSymbol<SinglePartWorkspaceFn>(handle, "aclnnMlvcSinglePartGetWorkspaceSize");
    auto* single_part_execute = LoadSymbol<ExecuteFn>(handle, "aclnnMlvcSinglePart");
    auto* restore_y_workspace =
        LoadSymbol<RestoreYWorkspaceFn>(handle, "aclnnMlvcRestoreYGetWorkspaceSize");
    auto* restore_y_execute = LoadSymbol<ExecuteFn>(handle, "aclnnMlvcRestoreY");
    auto* apply_quant_workspace =
        LoadSymbol<ApplyQuantWorkspaceFn>(handle, "aclnnMlvcApplyChannelQuantStepGetWorkspaceSize");
    auto* apply_quant_execute = LoadSymbol<ExecuteFn>(handle, "aclnnMlvcApplyChannelQuantStep");
    auto* int8_to_fp16_workspace =
        LoadSymbol<Int8ToFp16WorkspaceFn>(handle, "aclnnMlvcInt8ToFp16GetWorkspaceSize");
    auto* int8_to_fp16_execute = LoadSymbol<ExecuteFn>(handle, "aclnnMlvcInt8ToFp16");
    if (single_part_workspace != nullptr && single_part_execute != nullptr &&
        restore_y_workspace != nullptr && restore_y_execute != nullptr &&
        apply_quant_workspace != nullptr && apply_quant_execute != nullptr &&
        int8_to_fp16_workspace != nullptr && int8_to_fp16_execute != nullptr) {
      state.handle = handle;
      state.single_part_workspace = single_part_workspace;
      state.single_part_execute = single_part_execute;
      state.restore_y_workspace = restore_y_workspace;
      state.restore_y_execute = restore_y_execute;
      state.apply_quant_workspace = apply_quant_workspace;
      state.apply_quant_execute = apply_quant_execute;
      state.int8_to_fp16_workspace = int8_to_fp16_workspace;
      state.int8_to_fp16_execute = int8_to_fp16_execute;
      state.error.clear();
      return true;
    }
    dlclose(handle);
  }
  state.error = "decode-prior ACLNN custom op library was not found";
  return false;
}

DecodePriorAclResult ExecutePrepared(const char* name, std::size_t elements,
                                     uint64_t workspace_size, aclOpExecutor* executor,
                                     ExecuteFn execute, void* stream, Profiler* profiler,
                                     std::chrono::steady_clock::time_point begin) {
  AclBuffer workspace;
  if (workspace_size != 0) {
    workspace.Allocate(workspace_size);
  }
  aclrtStream acl_stream = static_cast<aclrtStream>(stream);
  CheckAclnnStatus(execute(workspace.data(), workspace_size, executor, acl_stream), name);
  CheckAclStatus(aclrtSynchronizeStream(acl_stream), name);
  const auto end = std::chrono::steady_clock::now();
  if (profiler != nullptr) {
    profiler->AddEventWithArgs(std::string("acl_prior.") + name, "acl_prior", "acl_prior",
                               profiler->StartMs(begin), profiler->DurationMs(begin, end),
                               {Profiler::Arg("kernel", name),
                                Profiler::Arg("elements", static_cast<uint64_t>(elements))});
  }
  return DecodePriorAclResult{elements,
                              profiler != nullptr ? profiler->DurationMs(begin, end) : 0.0};
}

void CheckParts(int parts, int mask_index, int channels, int height, int width) {
  Check(parts == 2 || parts == 4, "decode prior parts must be 2 or 4");
  Check(mask_index >= 0 && mask_index < parts, "decode prior mask index is out of range");
  Check(channels > 0 && channels % parts == 0, "decode prior channel count mismatch");
  Check(height > 0 && width > 0, "decode prior spatial shape must be positive");
}

}  // namespace

bool DecodePriorAclAvailable() { return LoadApi(); }

DecodePriorAclResult DecodePriorSinglePartAcl(const void* full, DataType dtype, int parts,
                                              int mask_index, int channels, int height, int width,
                                              void* part, void* stream, Profiler* profiler) {
  Check(full != nullptr, "single_part ACL input is required");
  Check(part != nullptr, "single_part ACL output is required");
  Check(dtype == DataType::kFloat16 || dtype == DataType::kFloat32,
        "single_part ACL expects fp16 or fp32");
  CheckParts(parts, mask_index, channels, height, width);
  Check(LoadApi(), "single_part ACL custom op is not available");
  const int part_channels = channels / parts;
  const int64_t full_shape[4] = {1, channels, height, width};
  const int64_t part_shape[4] = {1, part_channels, height, width};
  TensorPtr full_tensor = MakeAclTensor(full_shape, 4, dtype, const_cast<void*>(full));
  TensorPtr part_tensor = MakeAclTensor(part_shape, 4, dtype, part);

  constexpr int64_t kTileElements = 1024;
  constexpr int64_t kBlockDim = 1;
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  DecodePriorAclApi& api = ApiState();
  const auto begin = std::chrono::steady_clock::now();
  CheckAclnnStatus(
      api.single_part_workspace(full_tensor.get(), mask_index, parts, kTileElements, kBlockDim,
                                part_tensor.get(), &workspace_size, &executor),
      "aclnnMlvcSinglePartGetWorkspaceSize");
  return ExecutePrepared("single_part", static_cast<std::size_t>(part_channels) * height * width,
                         workspace_size, executor, api.single_part_execute, stream, profiler,
                         begin);
}

DecodePriorAclResult DecodePriorRestoreYAcl(const int8_t* y_symbols, const void* means_fp16,
                                            int parts, int mask_index, int channels, int height,
                                            int width, void* output_fp16, void* stream,
                                            Profiler* profiler) {
  Check(y_symbols != nullptr, "restore_y ACL symbols input is required");
  Check(means_fp16 != nullptr, "restore_y ACL means input is required");
  Check(output_fp16 != nullptr, "restore_y ACL output is required");
  CheckParts(parts, mask_index, channels, height, width);
  Check(LoadApi(), "restore_y ACL custom op is not available");
  const int part_channels = channels / parts;
  const int64_t part_shape[4] = {1, part_channels, height, width};
  const int64_t full_shape[4] = {1, channels, height, width};
  TensorPtr symbols_tensor =
      MakeAclTensor(part_shape, 4, DataType::kInt8, const_cast<int8_t*>(y_symbols));
  TensorPtr means_tensor =
      MakeAclTensor(full_shape, 4, DataType::kFloat16, const_cast<void*>(means_fp16));
  TensorPtr output_tensor = MakeAclTensor(full_shape, 4, DataType::kFloat16, output_fp16);

  constexpr int64_t kTileElements = 1024;
  constexpr int64_t kBlockDim = 1;
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  DecodePriorAclApi& api = ApiState();
  const auto begin = std::chrono::steady_clock::now();
  CheckAclnnStatus(api.restore_y_workspace(symbols_tensor.get(), means_tensor.get(), mask_index,
                                           parts, kTileElements, kBlockDim, output_tensor.get(),
                                           &workspace_size, &executor),
                   "aclnnMlvcRestoreYGetWorkspaceSize");
  return ExecutePrepared("restore_y", static_cast<std::size_t>(part_channels) * height * width,
                         workspace_size, executor, api.restore_y_execute, stream, profiler, begin);
}

DecodePriorAclResult DecodePriorApplyChannelQuantStepAcl(
    const void* quant_step_fp16, int quant_channels, const void* values_fp16, int channels,
    int height, int width, void* values_out_fp16, void* stream, Profiler* profiler) {
  Check(quant_step_fp16 != nullptr, "apply_channel_quant_step ACL quant input is required");
  Check(values_fp16 != nullptr, "apply_channel_quant_step ACL values input is required");
  Check(values_out_fp16 != nullptr, "apply_channel_quant_step ACL output is required");
  Check(channels > 0 && height > 0 && width > 0,
        "apply_channel_quant_step ACL shape must be positive");
  Check(quant_channels == 1 || quant_channels == channels,
        "apply_channel_quant_step ACL quant channel mismatch");
  Check(LoadApi(), "apply_channel_quant_step ACL custom op is not available");
  const int64_t quant_shape[4] = {1, quant_channels, height, width};
  const int64_t value_shape[4] = {1, channels, height, width};
  TensorPtr quant_tensor =
      MakeAclTensor(quant_shape, 4, DataType::kFloat16, const_cast<void*>(quant_step_fp16));
  TensorPtr value_tensor =
      MakeAclTensor(value_shape, 4, DataType::kFloat16, const_cast<void*>(values_fp16));
  TensorPtr output_tensor = MakeAclTensor(value_shape, 4, DataType::kFloat16, values_out_fp16);

  constexpr int64_t kTileElements = 1024;
  constexpr int64_t kBlockDim = 1;
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  DecodePriorAclApi& api = ApiState();
  const auto begin = std::chrono::steady_clock::now();
  CheckAclnnStatus(
      api.apply_quant_workspace(quant_tensor.get(), value_tensor.get(), kTileElements, kBlockDim,
                                output_tensor.get(), &workspace_size, &executor),
      "aclnnMlvcApplyChannelQuantStepGetWorkspaceSize");
  return ExecutePrepared("apply_channel_quant_step",
                         static_cast<std::size_t>(channels) * height * width, workspace_size,
                         executor, api.apply_quant_execute, stream, profiler, begin);
}

DecodePriorAclResult DecodePriorInt8ToFp16Acl(const int8_t* symbols, const int64_t* shape,
                                              std::size_t rank, std::size_t count,
                                              void* output_fp16, void* stream, Profiler* profiler) {
  Check(symbols != nullptr, "int8_to_fp16 ACL symbols input is required");
  Check(output_fp16 != nullptr, "int8_to_fp16 ACL output is required");
  Check(shape != nullptr && rank > 0, "int8_to_fp16 ACL shape is required");
  Check(LoadApi(), "int8_to_fp16 ACL custom op is not available");
  TensorPtr symbols_tensor =
      MakeAclTensor(shape, rank, DataType::kInt8, const_cast<int8_t*>(symbols));
  TensorPtr output_tensor = MakeAclTensor(shape, rank, DataType::kFloat16, output_fp16);

  constexpr int64_t kTileElements = 1024;
  constexpr int64_t kBlockDim = 1;
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  DecodePriorAclApi& api = ApiState();
  const auto begin = std::chrono::steady_clock::now();
  CheckAclnnStatus(api.int8_to_fp16_workspace(symbols_tensor.get(), kTileElements, kBlockDim,
                                              output_tensor.get(), &workspace_size, &executor),
                   "aclnnMlvcInt8ToFp16GetWorkspaceSize");
  return ExecutePrepared("int8_to_fp16", count, workspace_size, executor, api.int8_to_fp16_execute,
                         stream, profiler, begin);
}

}  // namespace mlvc
