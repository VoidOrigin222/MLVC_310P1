#include <ulbvc/codec/detail/stage/stage_output_policy.h>

#include <iostream>
#include <string>

#include "ulbvc/core/status.h"

namespace {

ulbvc::TensorSpec Spec(std::string name, ulbvc::DataType dtype) {
  ulbvc::TensorSpec spec;
  spec.name = std::move(name);
  spec.dtype = dtype;
  spec.shape = {1, 4, 2, 2};
  return spec;
}

void CheckSkip(std::string_view stage_name, const ulbvc::TensorSpec& spec, bool skip_entropy,
               bool skip_encode_acl_only, bool skip_decode_acl_only,
               bool decode_prior_acl_available, const char* expected,
               bool force_decode_video_output_cpu_mirror = false) {
  const ulbvc::codec::MaterializationReason reason =
      ulbvc::codec::StageOutputMaterializationReason(stage_name, spec.name);
  const char* actual = ulbvc::codec::StageOutputCpuMirrorSkipReason(
      stage_name, spec, reason, skip_entropy, skip_encode_acl_only, skip_decode_acl_only,
      decode_prior_acl_available, force_decode_video_output_cpu_mirror);
  if (expected == nullptr) {
    ulbvc::Check(actual == nullptr,
                "expected no CPU mirror skip for " + std::string(stage_name) + "." + spec.name);
  } else {
    ulbvc::Check(
        actual != nullptr && std::string(actual) == expected,
        "unexpected CPU mirror skip reason for " + std::string(stage_name) + "." + spec.name);
  }
}

}  // namespace

int main() {
  try {
    CheckSkip("i_hyper_encoder", Spec("z_symbols", ulbvc::DataType::kFloat16), true, false, false,
              false, "deferred_pinned_copy");
    CheckSkip("p_reference_context", Spec("reference_context", ulbvc::DataType::kFloat16), false,
              false, true, false, "acl_only_downstream");
    CheckSkip("i_spatial_prior_decode_init", Spec("scale_params", ulbvc::DataType::kFloat16), false,
              false, true, true, "acl_prior_single_part");
    CheckSkip("i_spatial_prior_decode_step_2", Spec("scale_params", ulbvc::DataType::kFloat16),
              false, false, true, true, "acl_prior_single_part");
    CheckSkip("p_spatial_prior_decode_step", Spec("scale_params", ulbvc::DataType::kFloat16), false,
              false, true, true, "acl_prior_single_part");
    CheckSkip("i_spatial_prior_decode_init", Spec("scale_params", ulbvc::DataType::kFloat16), false,
              false, true, false, nullptr);
    CheckSkip("i_spatial_prior_decode_init", Spec("scale_params", ulbvc::DataType::kFloat32), false,
              false, true, true, nullptr);
    CheckSkip("i_spatial_prior_decode_init", Spec("mean_params", ulbvc::DataType::kFloat16), false,
              false, true, true, nullptr);
    CheckSkip("i_spatial_prior_decode_init", Spec("decoder_y_quant_step", ulbvc::DataType::kFloat16),
              false, false, true, true, nullptr);
    CheckSkip("ULBVCDecoder", Spec("x_hat", ulbvc::DataType::kFloat16), false, false, false, true,
              nullptr);
    CheckSkip("ULBVCDecoder", Spec("feature", ulbvc::DataType::kFloat16), false, false, false, true,
              nullptr);
    CheckSkip("ULBVCDecoder", Spec("x_hat", ulbvc::DataType::kFloat16), false, false, true, true,
              "acl_only_downstream");
    CheckSkip("ULBVCDecoder", Spec("feature", ulbvc::DataType::kFloat16), false, false, true, true,
              "acl_only_downstream");
    CheckSkip("ULBVCDecoder", Spec("x_hat", ulbvc::DataType::kFloat16), false, false, true, true,
              nullptr, true);
    std::cout << "stage_output_policy status=ok\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "check_stage_output_policy failed: " << error.what() << "\n";
    return 1;
  }
}
