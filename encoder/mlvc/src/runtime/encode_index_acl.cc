#include "mlvc/runtime/encode_index_acl.h"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "mlvc/core/buffer.h"
#include "mlvc/core/status.h"
#include "mlvc/entropy/entropy_codec.h"

namespace mlvc {
namespace {

using GetWorkspaceFn = aclnnStatus (*)(const aclTensor*, const aclTensor*, const aclTensor*, double,
                                       int64_t, int64_t, const aclTensor*, const aclTensor*,
                                       uint64_t*, aclOpExecutor**);
using ExecuteFn = aclnnStatus (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);

struct AclTensorDeleter {
  void operator()(const aclTensor* tensor) const {
    if (tensor != nullptr) {
      (void)aclDestroyTensor(tensor);
    }
  }
};

using TensorPtr = std::unique_ptr<const aclTensor, AclTensorDeleter>;

struct EncodeIndexAclApi {
  void* handle = nullptr;
  GetWorkspaceFn get_workspace = nullptr;
  ExecuteFn execute = nullptr;
  bool attempted = false;
  std::string error;
};

EncodeIndexAclApi& ApiState() {
  static EncodeIndexAclApi state;
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

TensorPtr MakeAclTensor(const int64_t* shape, std::size_t rank, DataType dtype, void* data,
                        aclFormat format = ACL_FORMAT_ND) {
  Check(shape != nullptr || rank == 0, "ACL tensor shape is required");
  std::vector<int64_t> strides(rank, 1);
  for (std::size_t i = rank; i > 1; --i) {
    strides[i - 2] = strides[i - 1] * shape[i - 1];
  }
  aclTensor* tensor =
      aclCreateTensor(shape, static_cast<uint64_t>(rank), AclDataTypeFor(dtype), strides.data(), 0,
                      format, shape, static_cast<uint64_t>(rank), data);
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

float HalfBitsToFloat(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
  uint32_t exponent = (value >> 10) & 0x1fu;
  uint32_t mantissa = value & 0x03ffu;
  uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ffu;
      const uint32_t exp32 = exponent + (127 - 15);
      bits = sign | (exp32 << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    const uint32_t exp32 = exponent + (127 - 15);
    bits = sign | (exp32 << 23) | (mantissa << 13);
  }
  float output = 0.0f;
  std::memcpy(&output, &bits, sizeof(output));
  return output;
}

uint8_t ScaleIndexHost(float scale) {
  const float log_scale_min = std::log(kScaleMin);
  const float log_scale_max = std::log(kScaleMax);
  const float log_scale_step = (log_scale_max - log_scale_min) / (kScaleLevel - 1);
  const float log_step_recip = 1.0f / log_scale_step;
  const float clipped = std::min(std::max(scale, kScaleMin), kScaleMax);
  return static_cast<uint8_t>((std::log(clipped) - log_scale_min) * log_step_recip);
}

AclBuffer& Fp16ScaleIndexLookup() {
  static AclBuffer* lookup = new AclBuffer();
  static std::once_flag once;
  std::call_once(once, [] {
    std::vector<uint8_t> host_lookup(65536);
    for (std::size_t bits = 0; bits < host_lookup.size(); ++bits) {
      host_lookup[bits] = ScaleIndexHost(HalfBitsToFloat(static_cast<uint16_t>(bits)));
    }
    lookup->Allocate(host_lookup.size());
    CheckAclStatus(aclrtMemcpy(lookup->data(), lookup->bytes(), host_lookup.data(),
                               host_lookup.size(), ACL_MEMCPY_HOST_TO_DEVICE),
                   "aclrtMemcpy fp16 scale-index lookup");
  });
  return *lookup;
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

bool LoadApi() {
  std::lock_guard<std::mutex> lock(ApiMutex());
  EncodeIndexAclApi& state = ApiState();
  if (state.attempted) {
    return state.get_workspace != nullptr && state.execute != nullptr;
  }
  state.attempted = true;
  for (const std::filesystem::path& candidate : CandidateLibraries()) {
    void* handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      continue;
    }
    auto* get_workspace = reinterpret_cast<GetWorkspaceFn>(
        dlsym(handle, "aclnnMlvcBuildIndexEncodeGetWorkspaceSize"));
    auto* execute = reinterpret_cast<ExecuteFn>(dlsym(handle, "aclnnMlvcBuildIndexEncode"));
    if (get_workspace != nullptr && execute != nullptr) {
      state.handle = handle;
      state.get_workspace = get_workspace;
      state.execute = execute;
      state.error.clear();
      return true;
    }
    dlclose(handle);
  }
  state.error = "aclnnMlvcBuildIndexEncode custom op library was not found";
  return false;
}

void RecordKernelEvent(Profiler* profiler, std::chrono::steady_clock::time_point begin,
                       std::chrono::steady_clock::time_point end, std::size_t elements,
                       DataType symbol_dtype, DataType scale_dtype) {
  if (profiler == nullptr) {
    return;
  }
  profiler->AddEventWithArgs("acl_entropy.build_index_enc", "acl_entropy", "acl_entropy",
                             profiler->StartMs(begin), profiler->DurationMs(begin, end),
                             {Profiler::Arg("kernel", "MlvcBuildIndexEncode"),
                              Profiler::Arg("elements", static_cast<uint64_t>(elements)),
                              Profiler::Arg("symbol_dtype", DataTypeName(symbol_dtype)),
                              Profiler::Arg("scale_dtype", DataTypeName(scale_dtype))});
}

}  // namespace

bool EncodeIndexAclAvailable() { return LoadApi(); }

EncodeIndexAclResult BuildIndexEncodeAcl(const void* symbols, DataType symbol_dtype,
                                         const void* scales, DataType scale_dtype,
                                         const int64_t* shape, std::size_t rank, std::size_t count,
                                         float force_zero_thres, int16_t* combined,
                                         uint8_t* keep_mask, void* stream, Profiler* profiler) {
  Check(symbols != nullptr, "build_index_enc ACL symbols input is required");
  Check(scales != nullptr, "build_index_enc ACL scales input is required");
  Check(combined != nullptr, "build_index_enc ACL combined output is required");
  Check(keep_mask != nullptr, "build_index_enc ACL keep mask output is required");
  Check(symbol_dtype == DataType::kInt8 || symbol_dtype == DataType::kFloat16,
        "build_index_enc ACL symbols must be int8 or fp16");
  Check(scale_dtype == DataType::kFloat16, "build_index_enc ACL currently requires fp16 scales");
  Check(rank > 0 && shape != nullptr, "build_index_enc ACL tensor shape is required");
  Check(LoadApi(), "build_index_enc ACL custom op is not available");

  AclBuffer& lookup = Fp16ScaleIndexLookup();
  const int64_t lookup_shape[1] = {65536};
  TensorPtr symbol_tensor = MakeAclTensor(shape, rank, symbol_dtype, const_cast<void*>(symbols));
  TensorPtr scale_tensor = MakeAclTensor(shape, rank, scale_dtype, const_cast<void*>(scales));
  TensorPtr lookup_tensor =
      MakeAclTensor(lookup_shape, 1, DataType::kUInt8, lookup.data(), ACL_FORMAT_ND);
  TensorPtr combined_tensor = MakeAclTensor(shape, rank, DataType::kInt16, combined);
  TensorPtr keep_tensor = MakeAclTensor(shape, rank, DataType::kUInt8, keep_mask);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  constexpr int64_t kTileElements = 1024;
  constexpr int64_t kBlockDim = 1;
  EncodeIndexAclApi& api = ApiState();
  const auto begin = std::chrono::steady_clock::now();
  CheckAclnnStatus(
      api.get_workspace(symbol_tensor.get(), scale_tensor.get(), lookup_tensor.get(),
                        static_cast<double>(force_zero_thres), kTileElements, kBlockDim,
                        combined_tensor.get(), keep_tensor.get(), &workspace_size, &executor),
      "aclnnMlvcBuildIndexEncodeGetWorkspaceSize");
  AclBuffer workspace;
  if (workspace_size != 0) {
    workspace.Allocate(workspace_size);
  }
  aclrtStream acl_stream = static_cast<aclrtStream>(stream);
  CheckAclnnStatus(api.execute(workspace.data(), workspace_size, executor, acl_stream),
                   "aclnnMlvcBuildIndexEncode");
  CheckAclStatus(aclrtSynchronizeStream(acl_stream), "aclrtSynchronizeStream build_index_enc");
  const auto end = std::chrono::steady_clock::now();
  RecordKernelEvent(profiler, begin, end, count, symbol_dtype, scale_dtype);
  return EncodeIndexAclResult{count, profiler != nullptr ? profiler->DurationMs(begin, end) : 0.0};
}

}  // namespace mlvc
