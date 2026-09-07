#ifndef MLVC_APPLICATION_RUNTIME_MLVC_CODEC_RUNTIME_H_
#define MLVC_APPLICATION_RUNTIME_MLVC_CODEC_RUNTIME_H_

#include <mlvc/codec/detail/stage/stage_workspace.h>
#include <mlvc/entropy/sidecar.h>
#include <mlvc/runtime/model_manifest.h>
#include <mlvc/runtime/stage_runtime.h>

#include <filesystem>

namespace mlvc::app {

// Owns the model-facing resources shared by one encode or decode pipeline.
// The object is intentionally non-copyable: StageModelSet and its workspace
// contain device resources whose lifetime must remain tied to the ACL runtime.
class MlvcCodecRuntime {
 public:
  MlvcCodecRuntime(const std::filesystem::path& manifest_path, int device,
                   mlvc::codec::StageOutputBindingMode binding_mode =
                       mlvc::codec::StageOutputBindingMode::kCpu);
  ~MlvcCodecRuntime() = default;

  MlvcCodecRuntime(const MlvcCodecRuntime&) = delete;
  MlvcCodecRuntime& operator=(const MlvcCodecRuntime&) = delete;
  MlvcCodecRuntime(MlvcCodecRuntime&&) = delete;
  MlvcCodecRuntime& operator=(MlvcCodecRuntime&&) = delete;

  mlvc::StageRuntime& runtime() { return runtime_; }
  const mlvc::StageRuntime& runtime() const { return runtime_; }
  mlvc::StageModelSet& models() { return models_; }
  const mlvc::StageModelSet& models() const { return models_; }
  const mlvc::RuntimeSidecar& sidecar() const { return sidecar_; }
  mlvc::codec::StageOutputWorkspace& stage_output_workspace() { return workspace_; }
  const mlvc::codec::StageOutputWorkspace& stage_output_workspace() const { return workspace_; }

  const mlvc::ModelManifest& manifest() const { return models_.manifest(); }
  const std::filesystem::path& manifest_path() const { return models_.manifest().path(); }
  int device() const { return runtime_.device_id(); }

 private:
  static std::filesystem::path ResolveSidecarPath(const mlvc::ModelManifest& manifest);

  // Declaration order is significant. Models use runtime_, sidecar uses the
  // manifest owned by models_, and workspace uses models_. Destruction occurs
  // in the reverse order, preserving those dependencies.
  mlvc::StageRuntime runtime_;
  mlvc::StageModelSet models_;
  mlvc::RuntimeSidecar sidecar_;
  mlvc::codec::StageOutputWorkspace workspace_;
};

}  // namespace mlvc::app

#endif  // MLVC_APPLICATION_RUNTIME_MLVC_CODEC_RUNTIME_H_
