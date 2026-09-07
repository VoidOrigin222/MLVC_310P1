#include <mlvc/application/runtime/mlvc_codec_runtime.h>

#include <filesystem>
#include <string>

#include "mlvc/core/status.h"

namespace mlvc::app {

std::filesystem::path MlvcCodecRuntime::ResolveSidecarPath(
    const mlvc::ModelManifest& manifest) {
  mlvc::Check(!manifest.sidecar().file.empty(),
              "model manifest does not specify a runtime sidecar");
  const std::filesystem::path sidecar_path = manifest.sidecar().file.is_absolute()
                                                 ? manifest.sidecar().file
                                                 : manifest.directory() / manifest.sidecar().file;
  mlvc::Check(std::filesystem::exists(sidecar_path),
              "runtime sidecar does not exist: " + sidecar_path.string());
  return sidecar_path;
}

MlvcCodecRuntime::MlvcCodecRuntime(
    const std::filesystem::path& manifest_path, int device,
    mlvc::codec::StageOutputBindingMode binding_mode)
    : runtime_(device),
      models_(&runtime_, mlvc::ModelManifest::Load(manifest_path)),
      sidecar_(mlvc::RuntimeSidecar::Load(ResolveSidecarPath(models_.manifest()))),
      workspace_(&models_, binding_mode) {}

}  // namespace mlvc::app
