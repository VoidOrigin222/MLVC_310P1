#ifndef ULBVC_CODEC_DETAIL_PRIOR_LAYOUT_H_
#define ULBVC_CODEC_DETAIL_PRIOR_LAYOUT_H_

#include <cstdint>
#include <vector>

namespace ulbvc::codec {

void SinglePart4x(const std::vector<float>& full, int mask_index, int channels, int height,
                  int width, std::vector<float>* part);
void SinglePart2x(const std::vector<float>& full, int mask_index, int channels, int height,
                  int width, std::vector<float>* part);
void RestoreY4x(const std::vector<int8_t>& y_symbols, const std::vector<float>& means,
                int mask_index, int channels, int height, int width, std::vector<float>* output);
void RestoreY2x(const std::vector<int8_t>& y_symbols, const std::vector<float>& means,
                int mask_index, int channels, int height, int width, std::vector<float>* output);

}  // namespace ulbvc::codec

#endif  // ULBVC_CODEC_DETAIL_PRIOR_LAYOUT_H_
