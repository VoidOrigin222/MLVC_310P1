#ifndef MLVC_CODEC_MLVC_ENTROPY_H_
#define MLVC_CODEC_MLVC_ENTROPY_H_

#include <mlvc/codec/tensor_data.h>

#include <cstdint>
#include <string_view>
#include <vector>

namespace mlvc::codec {

// dmc61sbr_mini_perceptual uses four repeated y-scale channels per z channel.
constexpr int kMlvcChannelRepeat = 4;

enum class MlvcFrameType : uint8_t {
  kIFrame = 0,
  kPFrame = 1,
  kLtrRecovery = 2,
};

bool IsMlvcIFrame(int frame_index, int gop, int reset_interval);
bool ShouldResetReferenceFeature(int frame_index, int gop, int reset_interval);
int MlvcResetCycleIndex(int frame_index, int reset_interval);
bool ShouldUseLtrFeatures(int cycle_index, int ltr_start_idx, int ltr_period);
bool ShouldSaveLtrFeatures(int cycle_index, int ltr_start_idx, int ltr_period);
MlvcFrameType MlvcFrameTypeForFrame(int frame_index, int gop, int reset_interval, int ltr_start_idx,
                                    int ltr_period);
std::string_view MlvcFrameTypeName(MlvcFrameType frame_type);
std::string_view MlvcEntropyPrefix(bool is_i_frame);
bool MlvcUseTwoParts(int width, int height);

void BuildMlvcScaleIndexesFromZRaw(const TensorData& z_raw, int y_channels, int y_height,
                                   int y_width, int channel_repeat, std::vector<uint8_t>* scales_0,
                                   std::vector<uint8_t>* scales_1);
void BuildMlvcCombinedYIndexes(const std::vector<int8_t>& symbols,
                               const std::vector<uint8_t>& scales, std::vector<int16_t>* indexes);

}  // namespace mlvc::codec

#endif  // MLVC_CODEC_MLVC_ENTROPY_H_
