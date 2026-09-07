#ifndef MLVC_CODEC_EXECUTION_PROFILE_H_
#define MLVC_CODEC_EXECUTION_PROFILE_H_

#include <string_view>

namespace mlvc::codec {

constexpr std::string_view kCodecExecutionProfile = "pipeline-v1";
constexpr std::string_view kCodecExecutionMode = "async";
constexpr std::string_view kCodecStageOutputBinding = "acl-mirror";
constexpr std::string_view kCodecScheduler = "dag";

inline bool IsCodecExecutionProfile(std::string_view profile) {
  return profile.empty() || profile == kCodecExecutionProfile;
}

}  // namespace mlvc::codec

#endif  // MLVC_CODEC_EXECUTION_PROFILE_H_
