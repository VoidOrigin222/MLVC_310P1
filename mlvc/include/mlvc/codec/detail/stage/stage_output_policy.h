#ifndef MLVC_CODEC_DETAIL_STAGE_OUTPUT_POLICY_H_
#define MLVC_CODEC_DETAIL_STAGE_OUTPUT_POLICY_H_

#include <mlvc/runtime/model_manifest.h>

#include <string_view>

namespace mlvc::codec {

enum class MaterializationReason {
  kEntropyEncode,
  kEntropyDecode,
  kVideoWrite,
  kMadCheck,
  kDebugDump,
  kFallbackCpuStage,
  kAclOnlyDownstream,
};

const char* MaterializationReasonName(MaterializationReason reason);
MaterializationReason StageOutputMaterializationReason(std::string_view stage_name,
                                                       std::string_view tensor_name);
const char* StageOutputCpuMirrorSkipReason(
    std::string_view stage_name, const mlvc::TensorSpec& tensor, MaterializationReason reason,
    bool skip_entropy_cpu_mirror, bool skip_encode_acl_only_cpu_mirror,
    bool skip_decode_acl_only_cpu_mirror, bool decode_prior_acl_available);
bool IsEntropyOnlyStageOutput(std::string_view stage_name, std::string_view tensor_name);
bool IsQScaleInput(std::string_view input_name);
bool IsAsyncEncodeAclOnlyStageOutput(std::string_view stage_name, std::string_view tensor_name);
bool IsAsyncDecodeAclOnlyStageOutput(std::string_view stage_name, std::string_view tensor_name);

}  // namespace mlvc::codec

#endif  // MLVC_CODEC_DETAIL_STAGE_OUTPUT_POLICY_H_
