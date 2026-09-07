#include "mlvc/runtime/fp16_yuv444_to_nv12_acl.h"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <dlfcn.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "mlvc/core/buffer.h"
#include "mlvc/core/status.h"
#include "mlvc/runtime/acl_runtime.h"

namespace mlvc {
namespace {

using WorkspaceFn = aclnnStatus (*)(const aclTensor*, int64_t, int64_t, int64_t, int64_t,
                                    int64_t, const aclTensor*, uint64_t*, aclOpExecutor**);
using ExecuteFn = aclnnStatus (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);

struct AclTensorDeleter {
  void operator()(const aclTensor* tensor) const {
    if (tensor != nullptr) (void)aclDestroyTensor(tensor);
  }
};
using TensorPtr = std::unique_ptr<const aclTensor, AclTensorDeleter>;

struct Api {
  void* library = nullptr;
  WorkspaceFn workspace = nullptr;
  ExecuteFn execute = nullptr;
  bool attempted = false;
};

Api& State() {
  static Api state;
  return state;
}

std::mutex& StateMutex() {
  static std::mutex mutex;
  return mutex;
}

TensorPtr MakeTensor(const int64_t* shape, std::size_t rank, aclDataType dtype, void* data) {
  std::vector<int64_t> strides(rank, 1);
  for (std::size_t i = rank; i > 1; --i) strides[i - 2] = strides[i - 1] * shape[i - 1];
  aclTensor* tensor = aclCreateTensor(shape, static_cast<uint64_t>(rank), dtype, strides.data(), 0,
                                      ACL_FORMAT_ND, shape, static_cast<uint64_t>(rank), data);
  Check(tensor != nullptr, "aclCreateTensor for FP16 YUV444-to-NV12 failed");
  return TensorPtr(tensor);
}

bool Load() {
  std::lock_guard<std::mutex> lock(StateMutex());
  Api& state = State();
  if (state.attempted) return state.workspace != nullptr && state.execute != nullptr;
  state.attempted = true;
  std::vector<std::filesystem::path> candidates;
  if (const char* explicit_path = std::getenv("MLVC_VIDEO_OPAPI_LIB")) {
    if (*explicit_path != '\0') candidates.emplace_back(explicit_path);
  }
  if (const char* prior_path = std::getenv("MLVC_PRIOR_OPAPI_LIB")) {
    if (*prior_path != '\0') candidates.emplace_back(prior_path);
  }
  if (const char* root = std::getenv("MLVC_ACL_REPO_ROOT")) {
    if (*root != '\0') {
      candidates.emplace_back(std::filesystem::path(root) /
                              "output/custom_opp/mlvc_prior_ops/vendors/mlvc/op_api/lib/"
                              "libcust_opapi.so");
    }
  }
  candidates.emplace_back("libcust_opapi.so");
  for (const auto& candidate : candidates) {
    void* library = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) continue;
    auto workspace = reinterpret_cast<WorkspaceFn>(
        dlsym(library, "aclnnMlvcFp16Yuv444ToNv12GetWorkspaceSize"));
    auto execute = reinterpret_cast<ExecuteFn>(
        dlsym(library, "aclnnMlvcFp16Yuv444ToNv12"));
    if (workspace != nullptr && execute != nullptr) {
      state.library = library;
      state.workspace = workspace;
      state.execute = execute;
      return true;
    }
    dlclose(library);
  }
  return false;
}

}  // namespace

bool Fp16Yuv444ToNv12AclAvailable() { return Load(); }

void Fp16Yuv444ToNv12Acl(const void* input_fp16, const TensorShape& input_shape,
                         const io::Nv12Layout& layout, void* output_nv12,
                         void* stream, Profiler* profiler) {
  Fp16Yuv444ToNv12Acl(input_fp16, input_shape, layout, output_nv12, stream,
                      nullptr, nullptr, profiler);
}

void Fp16Yuv444ToNv12Acl(const void* input_fp16, const TensorShape& input_shape,
                         const io::Nv12Layout& layout, void* output_nv12,
                         void* stream, aclrtEvent start_event,
                         aclrtEvent ready_event, Profiler* profiler) {
  Check(input_fp16 != nullptr && output_nv12 != nullptr && stream != nullptr,
        "device FP16 YUV444-to-NV12 arguments must not be null");
  Check(input_shape.rank() == 4 && input_shape.dim(0) == 1 && input_shape.dim(1) == 3,
        "device FP16 YUV444-to-NV12 expects [1,3,H,W]");
  Check(layout.width > 0 && layout.height > 0 && layout.width % 2 == 0 &&
            layout.height % 2 == 0 && layout.width_stride >= layout.width &&
            layout.height_stride >= layout.height,
        "invalid device NV12 layout");
  Check(Load(), "FP16 YUV444-to-NV12 ACL custom op is unavailable");

  const int64_t output_shape[2] = {layout.height_stride * 3LL / 2LL, layout.width_stride};
  TensorPtr input = MakeTensor(input_shape.dims().data(), input_shape.rank(), ACL_FLOAT16,
                               const_cast<void*>(input_fp16));
  TensorPtr output = MakeTensor(output_shape, 2, ACL_UINT8, output_nv12);
  constexpr int64_t kBlockDim = 32;
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  Api& api = State();
  const auto begin = std::chrono::steady_clock::now();
  const aclnnStatus prepare = api.workspace(
      input.get(), layout.width, layout.height, layout.width_stride, layout.height_stride,
      kBlockDim, output.get(), &workspace_size, &executor);
  Check(prepare == OK, "aclnnMlvcFp16Yuv444ToNv12GetWorkspaceSize failed: ret=" +
                           std::to_string(prepare));
  thread_local AclBuffer workspace;
  if (workspace.bytes() < workspace_size) workspace.Allocate(workspace_size);
  if (start_event != nullptr) {
    CheckAcl(aclrtRecordEvent(start_event, static_cast<aclrtStream>(stream)),
             "aclrtRecordEvent FP16 YUV444-to-NV12 start");
  }
  const aclnnStatus execute = api.execute(workspace.data(), workspace_size, executor,
                                          static_cast<aclrtStream>(stream));
  Check(execute == OK,
        "aclnnMlvcFp16Yuv444ToNv12 failed: ret=" + std::to_string(execute));
  if (ready_event != nullptr) {
    CheckAcl(aclrtRecordEvent(ready_event, static_cast<aclrtStream>(stream)),
             "aclrtRecordEvent FP16 YUV444-to-NV12 ready");
  }
  const auto end = std::chrono::steady_clock::now();
  if (profiler != nullptr) {
    profiler->AddEventWithArgs(
        "acl_video.fp16_yuv444_to_nv12.enqueue", "acl_video", "acl_video",
        profiler->StartMs(begin), profiler->DurationMs(begin, end),
        {Profiler::Arg("width", layout.width), Profiler::Arg("height", layout.height),
         Profiler::Arg("output_bytes", static_cast<uint64_t>(io::Nv12BufferSize(layout)))});
  }
}

}  // namespace mlvc
