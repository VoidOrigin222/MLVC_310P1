#include <mlvc/codec/detail/frame/reference.h>

#include <utility>

#include <mlvc/codec/detail/stage/stage_runner.h>
#include <mlvc/codec/detail/stage/stage_runtime_state.h>
#include "mlvc/core/status.h"
#include "mlvc/framework/profile_range.h"

namespace mlvc::codec {

ReferenceFeatureBinding ReferenceFeature(mlvc::StageModelSet* models, const ReferenceState& state,
                                         TensorData* scratch, mlvc::Profiler* profiler) {
  MLVC_PROFILE_RANGE_FUNCTION();
  mlvc::Check(scratch != nullptr, "reference feature scratch is required");
  if (state.feature_handle.has_value()) {
    MLVC_PROFILE_RANGE("ReferenceFeature.from_feature_handle");
    RunOutput adaptor =
        RunStage(models, "p_reference_feature_adaptor",
                 {StageInput{"reference_feature_in", nullptr, &*state.feature_handle}}, profiler);
    const mlvc::TensorHandle* handle = adaptor.Handle("reference_feature");
    if (handle != nullptr && handle->acl_valid()) {
      return ReferenceFeatureBinding{nullptr, handle};
    }
  }
  if (state.feature.has_value()) {
    MLVC_PROFILE_RANGE("ReferenceFeature.from_feature");
    RunOutput adaptor = RunStage(models, "p_reference_feature_adaptor",
                                 {{"reference_feature_in", &*state.feature}}, profiler);
    const mlvc::TensorHandle* handle = adaptor.Handle("reference_feature");
    if (handle != nullptr && handle->acl_valid()) {
      return ReferenceFeatureBinding{nullptr, handle};
    }
    {
      MLVC_PROFILE_RANGE("ReferenceFeature.clone.reference_feature");
      CloneTensorInto(adaptor.At("reference_feature"), scratch);
    }
    return ReferenceFeatureBinding{scratch, nullptr};
  }
  if (state.frame_handle.has_value()) {
    MLVC_PROFILE_RANGE("ReferenceFeature.from_frame_handle");
    RunOutput adaptor =
        RunStage(models, "p_reference_frame_adaptor",
                 {StageInput{"reference_frame", nullptr, &*state.frame_handle}}, profiler);
    const mlvc::TensorHandle* handle = adaptor.Handle("reference_feature");
    if (handle != nullptr && handle->acl_valid()) {
      return ReferenceFeatureBinding{nullptr, handle};
    }
  }
  mlvc::Check(state.frame.has_value(), "missing P-frame reference");
  MLVC_PROFILE_RANGE("ReferenceFeature.from_frame");
  RunOutput adaptor =
      RunStage(models, "p_reference_frame_adaptor", {{"reference_frame", &*state.frame}}, profiler);
  const mlvc::TensorHandle* handle = adaptor.Handle("reference_feature");
  if (handle != nullptr && handle->acl_valid()) {
    return ReferenceFeatureBinding{nullptr, handle};
  }
  {
    MLVC_PROFILE_RANGE("ReferenceFeature.clone.reference_feature");
    CloneTensorInto(adaptor.At("reference_feature"), scratch);
  }
  return ReferenceFeatureBinding{scratch, nullptr};
}

void MaterializeReferenceFrame(mlvc::StageModelSet* models, QScaleCache* q_scales, int last_qp,
                               ReferenceState* state, TensorData* scratch,
                               mlvc::Profiler* profiler) {
  MLVC_PROFILE_RANGE_FUNCTION();
  mlvc::Check(state != nullptr, "reference state is required");
  mlvc::Check(q_scales != nullptr, "q scale cache is required");
  mlvc::Check(scratch != nullptr, "materialized reference scratch is required");
  if (state->frame.has_value()) {
    state->ResetFeature();
    return;
  }
  if (state->frame_handle.has_value()) {
    state->ResetFeature();
    return;
  }
  mlvc::Check(state->feature.has_value() || state->feature_handle.has_value(),
              "cannot materialize an empty reference");
  const TensorData& reconstruction_q_step = q_scales->PReconstruction(last_qp);
  RunOutput reconstruction;
  if (state->feature_handle.has_value()) {
    reconstruction = RunStage(models, "p_reconstruction",
                              {StageInput{"decoded_feature", nullptr, &*state->feature_handle},
                               TensorOrHandleInput("reconstruction_q_step", reconstruction_q_step,
                                                   q_scales->PReconstructionHandle(last_qp))},
                              profiler);
  } else {
    reconstruction = RunStage(models, "p_reconstruction",
                              {TensorInput("decoded_feature", *state->feature),
                               TensorOrHandleInput("reconstruction_q_step", reconstruction_q_step,
                                                   q_scales->PReconstructionHandle(last_qp))},
                              profiler);
  }
  mlvc::TensorHandle frame_handle;
  const bool cloned_frame_handle =
      CloneRunOutputHandle(reconstruction, "x_hat", &frame_handle, "reference_frame", profiler);
  if (cloned_frame_handle) {
    state->frame_handle.emplace(std::move(frame_handle));
    state->frame.reset();
  } else {
    state->frame_handle.reset();
    CloneTensorInto(reconstruction.At("x_hat"), scratch);
    state->frame.emplace(*scratch);
  }
  state->ResetFeature();
}

}  // namespace mlvc::codec
