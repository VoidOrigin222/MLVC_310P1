#ifndef ULBVC_CODEC_ULBVC_ENTROPY_H_
#define ULBVC_CODEC_ULBVC_ENTROPY_H_

#include <ulbvc/codec/tensor_data.h>

#include <cstdint>
#include <string_view>
#include <vector>

namespace ulbvc::codec {

// dmc61sbr_mini_perceptual uses four repeated y-scale channels per z channel.
constexpr int kUlbvcChannelRepeat = 4;

enum class UlbvcFrameType : uint8_t {
  kIFrame = 0,
  kPFrame = 1,
  kLtrRecovery = 2,
};

bool IsUlbvcIFrame(int frame_index, int gop, int reset_interval);
bool ShouldResetReferenceFeature(int frame_index, int gop, int reset_interval);
int UlbvcResetCycleIndex(int frame_index, int reset_interval);
bool ShouldUseLtrFeatures(int cycle_index, int ltr_start_idx, int ltr_period);
bool ShouldSaveLtrFeatures(int cycle_index, int ltr_start_idx, int ltr_period);
void ValidateForcedLtrConfiguration(int gop, int ltr_start_idx, int ltr_period,
                                    int recovery_frame, int reference_frame);
UlbvcFrameType UlbvcFrameTypeForFrame(int frame_index, int gop, int reset_interval, int ltr_start_idx,
                                    int ltr_period);
std::string_view UlbvcFrameTypeName(UlbvcFrameType frame_type);
std::string_view UlbvcEntropyPrefix(bool is_i_frame);
bool UlbvcUseTwoParts(int width, int height);

void BuildUlbvcScaleIndexesFromZRaw(const TensorData& z_raw, int y_channels, int y_height,
                                   int y_width, int channel_repeat, std::vector<uint8_t>* scales_0,
                                   std::vector<uint8_t>* scales_1);
void BuildUlbvcCombinedYIndexes(const std::vector<int8_t>& symbols,
                               const std::vector<uint8_t>& scales, std::vector<int16_t>* indexes);

}  // namespace ulbvc::codec

#endif  // ULBVC_CODEC_ULBVC_ENTROPY_H_
