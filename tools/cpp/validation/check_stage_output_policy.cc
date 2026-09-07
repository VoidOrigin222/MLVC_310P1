#include <iostream>
#include <string>

#include <mlvc/codec/detail/stage/stage_output_policy.h>
#include "mlvc/core/status.h"

namespace {

mlvc::TensorSpec Spec(std::string name, mlvc::DataType dtype) {
  mlvc::TensorSpec spec;
  spec.name = std::move(name);
  spec.dtype = dtype;
  spec.shape = {1, 4, 2, 2};
  return spec;
}

void CheckSkip(std::string_view stage_name, const mlvc::TensorSpec& spec, bool skip_entropy,
               bool skip_encode_acl_only, bool skip_decode_acl_only,
               bool decode_prior_acl_available, const char* expected) {
  const mlvc::codec::MaterializationReason reason =
      mlvc::codec::StageOutputMaterializationReason(stage_name, spec.name);
  const char* actual = mlvc::codec::StageOutputCpuMirrorSkipReason(
      stage_name, spec, reason, skip_entropy, skip_encode_acl_only, skip_decode_acl_only,
      decode_prior_acl_available);
  if (expected == nullptr) {
    mlvc::Check(actual == nullptr,
                "expected no CPU mirror skip for " + std::string(stage_name) + "." + spec.name);
  } else {
    mlvc::Check(
        actual != nullptr && std::string(actual) == expected,
        "unexpected CPU mirror skip reason for " + std::string(stage_name) + "." + spec.name);
  }
}

}  // namespace

int main() {
  try {
    CheckSkip("i_hyper_encoder", Spec("z_symbols", mlvc::DataType::kFloat16), true, false, false,
              false, "deferred_pinned_copy");
    CheckSkip("p_reference_context", Spec("reference_context", mlvc::DataType::kFloat16), false,
              false, true, false, "acl_only_downstream");
    CheckSkip("i_spatial_prior_decode_init", Spec("scale_params", mlvc::DataType::kFloat16), false,
              false, true, true, "acl_prior_single_part");
    CheckSkip("i_spatial_prior_decode_step_2", Spec("scale_params", mlvc::DataType::kFloat16),
              false, false, true, true, "acl_prior_single_part");
    CheckSkip("p_spatial_prior_decode_step", Spec("scale_params", mlvc::DataType::kFloat16), false,
              false, true, true, "acl_prior_single_part");
    CheckSkip("i_spatial_prior_decode_init", Spec("scale_params", mlvc::DataType::kFloat16), false,
              false, true, false, nullptr);
    CheckSkip("i_spatial_prior_decode_init", Spec("scale_params", mlvc::DataType::kFloat32), false,
              false, true, true, nullptr);
    CheckSkip("i_spatial_prior_decode_init", Spec("mean_params", mlvc::DataType::kFloat16), false,
              false, true, true, nullptr);
    CheckSkip("i_spatial_prior_decode_init", Spec("decoder_y_quant_step", mlvc::DataType::kFloat16),
              false, false, true, true, nullptr);
    CheckSkip("MLVCDecoder", Spec("x_hat", mlvc::DataType::kFloat16), false, false,
              false, true, nullptr);
    CheckSkip("MLVCDecoder", Spec("feature", mlvc::DataType::kFloat16), false, false,
              false, true, nullptr);
    CheckSkip("MLVCDecoder", Spec("x_hat", mlvc::DataType::kFloat16), false, false,
              true, true, "acl_only_downstream");
    CheckSkip("MLVCDecoder", Spec("feature", mlvc::DataType::kFloat16), false, false,
              true, true, "acl_only_downstream");
    std::cout << "stage_output_policy status=ok\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "check_stage_output_policy failed: " << error.what() << "\n";
    return 1;
  }
}
