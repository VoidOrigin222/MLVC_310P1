#ifndef ULBVC_CODEC_EXECUTION_PROFILE_H_
#define ULBVC_CODEC_EXECUTION_PROFILE_H_

#include <string_view>

namespace ulbvc::codec {

constexpr std::string_view kCodecExecutionProfile = "pipeline-v1";
constexpr std::string_view kCodecExecutionMode = "async";
constexpr std::string_view kCodecStageOutputBinding = "acl-device-with-cpu-mirror";
constexpr std::string_view kCodecScheduler = "dag";

inline bool IsCodecExecutionProfile(std::string_view profile) {
  return profile.empty() || profile == kCodecExecutionProfile;
}

}  // namespace ulbvc::codec

#endif  // ULBVC_CODEC_EXECUTION_PROFILE_H_
