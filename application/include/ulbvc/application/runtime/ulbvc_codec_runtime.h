#ifndef ULBVC_APPLICATION_RUNTIME_ULBVC_CODEC_RUNTIME_H_
#define ULBVC_APPLICATION_RUNTIME_ULBVC_CODEC_RUNTIME_H_

#include <ulbvc/codec/detail/stage/stage_workspace.h>
#include <ulbvc/entropy/sidecar.h>
#include <ulbvc/runtime/model_manifest.h>
#include <ulbvc/runtime/stage_runtime.h>

#include <filesystem>

namespace ulbvc::app {

// Owns the model-facing resources shared by one encode or decode pipeline.
// The object is intentionally non-copyable: StageModelSet and its workspace
// contain device resources whose lifetime must remain tied to the ACL runtime.
class UlbvcCodecRuntime {
 public:
  UlbvcCodecRuntime(
      const std::filesystem::path& manifest_path, int device,
      ulbvc::codec::StageOutputBindingMode binding_mode = ulbvc::codec::StageOutputBindingMode::kCpu);
  ~UlbvcCodecRuntime() = default;

  UlbvcCodecRuntime(const UlbvcCodecRuntime&) = delete;
  UlbvcCodecRuntime& operator=(const UlbvcCodecRuntime&) = delete;
  UlbvcCodecRuntime(UlbvcCodecRuntime&&) = delete;
  UlbvcCodecRuntime& operator=(UlbvcCodecRuntime&&) = delete;

  ulbvc::StageRuntime& runtime() { return runtime_; }
  const ulbvc::StageRuntime& runtime() const { return runtime_; }
  ulbvc::StageModelSet& models() { return models_; }
  const ulbvc::StageModelSet& models() const { return models_; }
  const ulbvc::RuntimeSidecar& sidecar() const { return sidecar_; }
  ulbvc::codec::StageOutputWorkspace& stage_output_workspace() { return workspace_; }
  const ulbvc::codec::StageOutputWorkspace& stage_output_workspace() const { return workspace_; }

  const ulbvc::ModelManifest& manifest() const { return models_.manifest(); }
  const std::filesystem::path& manifest_path() const { return models_.manifest().path(); }
  int device() const { return runtime_.device_id(); }

 private:
  static ulbvc::ModelManifest LoadVerifiedManifest(
      const std::filesystem::path& manifest_path);

  // Declaration order is significant. Models use runtime_, QP metadata comes
  // from the model directory, and workspace uses models_. Destruction occurs
  // in the reverse order, preserving those dependencies.
  ulbvc::StageRuntime runtime_;
  ulbvc::StageModelSet models_;
  ulbvc::RuntimeSidecar sidecar_;
  ulbvc::codec::StageOutputWorkspace workspace_;
};

}  // namespace ulbvc::app

#endif  // ULBVC_APPLICATION_RUNTIME_ULBVC_CODEC_RUNTIME_H_
