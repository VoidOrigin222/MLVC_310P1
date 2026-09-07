#ifndef MLVC_FP16_YUV444_TO_NV12_TILING_H
#define MLVC_FP16_YUV444_TO_NV12_TILING_H

#include <cstdint>

struct MlvcFp16Yuv444ToNv12TilingData {
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t visibleWidth;
    uint32_t visibleHeight;
    uint32_t widthStride;
    uint32_t heightStride;
    uint32_t outputRows;
    uint32_t rowsPerCore;
    uint32_t tailCoreRows;
};

#endif  // MLVC_FP16_YUV444_TO_NV12_TILING_H
