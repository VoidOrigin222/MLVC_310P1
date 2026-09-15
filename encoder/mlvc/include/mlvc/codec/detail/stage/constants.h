#ifndef MLVC_CODEC_DETAIL_CONSTANTS_H_
#define MLVC_CODEC_DETAIL_CONSTANTS_H_

#include <array>
#include <cstddef>

namespace mlvc::codec {

constexpr int kBaseQp = 17;
constexpr int kIndexMap[8] = {0, 1, 0, 2, 0, 2, 0, 2};
constexpr std::size_t kMaxStageInputs = 5;
constexpr std::size_t kMaxStageOutputs = 9;
constexpr std::array<const char*, 4> kYResidualPartNames = {
    "y_residual_symbols_part_0",
    "y_residual_symbols_part_1",
    "y_residual_symbols_part_2",
    "y_residual_symbols_part_3",
};
constexpr std::array<const char*, 4> kScalePartNames = {
    "scale_params_part_0",
    "scale_params_part_1",
    "scale_params_part_2",
    "scale_params_part_3",
};
constexpr std::array<const char*, 3> kISpatialPriorDecodeStepNames = {
    "i_spatial_prior_decode_step_1",
    "i_spatial_prior_decode_step_2",
    "i_spatial_prior_decode_step_3",
};

}  // namespace mlvc::codec

#endif  // MLVC_CODEC_DETAIL_CONSTANTS_H_
