#include <ulbvc/codec/detail/frame/reference.h>
#include <ulbvc/codec/detail/stage/stage_runner.h>
#include <ulbvc/codec/detail/stage/stage_runtime_state.h>

#include <utility>

#include "ulbvc/core/status.h"
#include "ulbvc/framework/profile_range.h"

namespace ulbvc::codec {

ReferenceFeatureBinding ReferenceFeature(ulbvc::StageModelSet* models, const ReferenceState& state,
                                         TensorData* scratch, ulbvc::Profiler* profiler) {
  ULBVC_PROFILE_RANGE_FUNCTION();
  ulbvc::Check(scratch != nullptr, "reference feature scratch is required");
  if (state.feature_handle.has_value()) {
    ULBVC_PROFILE_RANGE("ReferenceFeature.from_feature_handle");
    RunOutput adaptor =
        RunStage(models, "p_reference_feature_adaptor",
                 {StageInput{"reference_feature_in", nullptr, &*state.feature_handle}}, profiler);
    const ulbvc::TensorHandle* handle = adaptor.Handle("reference_feature");
    if (handle != nullptr && handle->acl_valid()) {
      return ReferenceFeatureBinding{nullptr, handle};
    }
  }
  if (state.feature.has_value()) {
    ULBVC_PROFILE_RANGE("ReferenceFeature.from_feature");
    RunOutput adaptor = RunStage(models, "p_reference_feature_adaptor",
                                 {{"reference_feature_in", &*state.feature}}, profiler);
    const ulbvc::TensorHandle* handle = adaptor.Handle("reference_feature");
    if (handle != nullptr && handle->acl_valid()) {
      return ReferenceFeatureBinding{nullptr, handle};
    }
    {
      ULBVC_PROFILE_RANGE("ReferenceFeature.clone.reference_feature");
      CloneTensorInto(adaptor.At("reference_feature"), scratch);
    }
    return ReferenceFeatureBinding{scratch, nullptr};
  }
  if (state.frame_handle.has_value()) {
    ULBVC_PROFILE_RANGE("ReferenceFeature.from_frame_handle");
    RunOutput adaptor =
        RunStage(models, "p_reference_frame_adaptor",
                 {StageInput{"reference_frame", nullptr, &*state.frame_handle}}, profiler);
    const ulbvc::TensorHandle* handle = adaptor.Handle("reference_feature");
    if (handle != nullptr && handle->acl_valid()) {
      return ReferenceFeatureBinding{nullptr, handle};
    }
  }
  ulbvc::Check(state.frame.has_value(), "missing P-frame reference");
  ULBVC_PROFILE_RANGE("ReferenceFeature.from_frame");
  RunOutput adaptor =
      RunStage(models, "p_reference_frame_adaptor", {{"reference_frame", &*state.frame}}, profiler);
  const ulbvc::TensorHandle* handle = adaptor.Handle("reference_feature");
  if (handle != nullptr && handle->acl_valid()) {
    return ReferenceFeatureBinding{nullptr, handle};
  }
  {
    ULBVC_PROFILE_RANGE("ReferenceFeature.clone.reference_feature");
    CloneTensorInto(adaptor.At("reference_feature"), scratch);
  }
  return ReferenceFeatureBinding{scratch, nullptr};
}

void MaterializeReferenceFrame(ulbvc::StageModelSet* models, QScaleCache* q_scales, int last_qp,
                               ReferenceState* state, TensorData* scratch,
                               ulbvc::Profiler* profiler) {
  ULBVC_PROFILE_RANGE_FUNCTION();
  ulbvc::Check(state != nullptr, "reference state is required");
  ulbvc::Check(q_scales != nullptr, "q scale cache is required");
  ulbvc::Check(scratch != nullptr, "materialized reference scratch is required");
  if (state->frame.has_value()) {
    state->ResetFeature();
    return;
  }
  if (state->frame_handle.has_value()) {
    state->ResetFeature();
    return;
  }
  ulbvc::Check(state->feature.has_value() || state->feature_handle.has_value(),
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
  ulbvc::TensorHandle frame_handle;
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

}  // namespace ulbvc::codec
