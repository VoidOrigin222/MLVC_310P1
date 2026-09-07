#include <mlvc/codec/detail/stage/stage_output_policy.h>

#include <string_view>

#include <mlvc/codec/detail/stage/constants.h>
#include "mlvc/core/status.h"

namespace mlvc::codec {

bool IsEntropyOnlyStageOutput(std::string_view stage_name, std::string_view tensor_name) {
  if ((stage_name == "i_hyper_encoder" || stage_name == "p_hyper_encoder" ||
       stage_name == "i_hyper_fused" || stage_name == "p_hyper_fused") &&
      tensor_name == "z_symbols") {
    return true;
  }
  if (stage_name == "i_spatial_prior") {
    for (int part = 0; part < 4; ++part) {
      if (tensor_name == kYResidualPartNames[part] || tensor_name == kScalePartNames[part]) {
        return true;
      }
    }
  }
  if (stage_name == "p_spatial_prior") {
    for (int part = 0; part < 2; ++part) {
      if (tensor_name == kYResidualPartNames[part] || tensor_name == kScalePartNames[part]) {
        return true;
      }
    }
  }
  return false;
}

bool IsQScaleInput(std::string_view input_name) {
  return input_name == "encoder_q_step" || input_name == "decoder_q_step" ||
         input_name == "feature_q_step" || input_name == "reconstruction_q_step";
}

const char* MaterializationReasonName(MaterializationReason reason) {
  switch (reason) {
    case MaterializationReason::kEntropyEncode:
      return "entropy_encode";
    case MaterializationReason::kEntropyDecode:
      return "entropy_decode";
    case MaterializationReason::kVideoWrite:
      return "video_write";
    case MaterializationReason::kMadCheck:
      return "mad_check";
    case MaterializationReason::kDebugDump:
      return "debug_dump";
    case MaterializationReason::kFallbackCpuStage:
      return "fallback_cpu_stage";
    case MaterializationReason::kAclOnlyDownstream:
      return "acl_only_downstream";
  }
  return "fallback_cpu_stage";
}

MaterializationReason StageOutputMaterializationReason(std::string_view stage_name,
                                                       std::string_view tensor_name) {
  if (IsEntropyOnlyStageOutput(stage_name, tensor_name)) {
    return MaterializationReason::kEntropyEncode;
  }
  if (stage_name == "i_spatial_prior_decode_init" ||
      stage_name == "i_spatial_prior_decode_step_1" ||
      stage_name == "i_spatial_prior_decode_step_2" ||
      stage_name == "i_spatial_prior_decode_step_3" ||
      stage_name == "p_spatial_prior_decode_init" || stage_name == "p_spatial_prior_decode_step") {
    if (tensor_name == "decoder_y_quant_step" || tensor_name == "scale_params" ||
        tensor_name == "mean_params") {
      return MaterializationReason::kEntropyDecode;
    }
  }
  if (stage_name == "i_decoder" || stage_name == "p_reconstruction") {
    if (tensor_name == "x_hat") {
      return MaterializationReason::kVideoWrite;
    }
  }
  if (IsAsyncEncodeAclOnlyStageOutput(stage_name, tensor_name) ||
      IsAsyncDecodeAclOnlyStageOutput(stage_name, tensor_name)) {
    return MaterializationReason::kAclOnlyDownstream;
  }
  return MaterializationReason::kFallbackCpuStage;
}

const char* StageOutputCpuMirrorSkipReason(
    std::string_view stage_name, const mlvc::TensorSpec& tensor, MaterializationReason reason,
    bool skip_entropy_cpu_mirror, bool skip_encode_acl_only_cpu_mirror,
    bool skip_decode_acl_only_cpu_mirror, bool decode_prior_acl_available) {
  if (skip_entropy_cpu_mirror && reason == MaterializationReason::kEntropyEncode) {
    return "deferred_pinned_copy";
  }
  if (skip_encode_acl_only_cpu_mirror && reason == MaterializationReason::kAclOnlyDownstream) {
    return "acl_only_downstream";
  }
  if (skip_decode_acl_only_cpu_mirror && reason == MaterializationReason::kAclOnlyDownstream) {
    return "acl_only_downstream";
  }
  if (skip_decode_acl_only_cpu_mirror && reason == MaterializationReason::kEntropyDecode &&
      tensor.name == "scale_params" && tensor.dtype == mlvc::DataType::kFloat16 &&
      decode_prior_acl_available) {
    return "acl_prior_single_part";
  }
  (void)stage_name;
  return nullptr;
}

bool IsAsyncEncodeAclOnlyStageOutput(std::string_view stage_name, std::string_view tensor_name) {
  if ((stage_name == "p_reference_feature_adaptor" || stage_name == "p_reference_frame_adaptor") &&
      tensor_name == "reference_feature") {
    return true;
  }
  if (stage_name == "i_encoder" && tensor_name == "y_latent") {
    return true;
  }
  if (stage_name == "i_hyper_fused" && tensor_name == "quantized_z") {
    return true;
  }
  if (stage_name == "i_hyper_encoder" && tensor_name == "quantized_z") {
    return true;
  }
  if (stage_name == "i_hyper_fused" && tensor_name == "fused_prior_params") {
    return true;
  }
  if (stage_name == "i_hyper_decoder_prior" && tensor_name == "fused_prior_params") {
    return true;
  }
  if (stage_name == "i_spatial_prior" && tensor_name == "reconstructed_y_latent") {
    return true;
  }
  if (stage_name == "p_reference_context" &&
      (tensor_name == "reference_context" || tensor_name == "temporal_prior_context")) {
    return true;
  }
  if (stage_name == "p_analysis_encoder" && tensor_name == "y_latent") {
    return true;
  }
  if (stage_name == "p_hyper_fused" && tensor_name == "quantized_z") {
    return true;
  }
  if (stage_name == "p_hyper_encoder" && tensor_name == "quantized_z") {
    return true;
  }
  if (stage_name == "p_hyper_fused" && tensor_name == "fused_prior_params") {
    return true;
  }
  if (stage_name == "p_hyper_temporal_prior" && tensor_name == "fused_prior_params") {
    return true;
  }
  if (stage_name == "p_spatial_prior" && tensor_name == "reconstructed_y_latent") {
    return true;
  }
  if (stage_name == "p_synthesis_decoder" && tensor_name == "decoded_feature") {
    return true;
  }
  if (stage_name == "p_reconstruction" && tensor_name == "x_hat") {
    return true;
  }
  return false;
}

bool IsAsyncDecodeAclOnlyStageOutput(std::string_view stage_name, std::string_view tensor_name) {
  if (stage_name == "MLVCDecoder" &&
      (tensor_name == "x_hat" || tensor_name == "feature")) {
    return true;
  }
  if ((stage_name == "p_reference_feature_adaptor" || stage_name == "p_reference_frame_adaptor") &&
      tensor_name == "reference_feature") {
    return true;
  }
  if (stage_name == "i_hyper_fused" && tensor_name == "fused_prior_params") {
    return true;
  }
  if (stage_name == "i_hyper_decoder_prior" && tensor_name == "fused_prior_params") {
    return true;
  }
  if (stage_name == "i_spatial_prior_decode_init" && tensor_name == "reduced_prior_params") {
    return true;
  }
  if (stage_name == "p_reference_context" &&
      (tensor_name == "reference_context" || tensor_name == "temporal_prior_context")) {
    return true;
  }
  if (stage_name == "p_hyper_fused" && tensor_name == "fused_prior_params") {
    return true;
  }
  if (stage_name == "p_hyper_temporal_prior" && tensor_name == "fused_prior_params") {
    return true;
  }
  return false;
}

}  // namespace mlvc::codec
