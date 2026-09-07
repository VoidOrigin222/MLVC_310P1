#ifndef MLVC_DECODE_PRIOR_TILING_H
#define MLVC_DECODE_PRIOR_TILING_H

#include <cstdint>

struct MlvcSinglePartTilingData {
    uint32_t totalElements;
    uint32_t elementsPerCore;
    uint32_t tailCoreElements;
    uint32_t tileElements;
    uint32_t channels;
    uint32_t height;
    uint32_t width;
    uint32_t partChannels;
    uint32_t maskIndex;
    uint32_t parts;
    uint32_t dataType;
};

struct MlvcRestoreYTilingData {
    uint32_t totalPartElements;
    uint32_t totalFullElements;
    uint32_t elementsPerCore;
    uint32_t tailCoreElements;
    uint32_t tileElements;
    uint32_t channels;
    uint32_t height;
    uint32_t width;
    uint32_t partChannels;
    uint32_t maskIndex;
    uint32_t parts;
};

struct MlvcApplyChannelQuantStepTilingData {
    uint32_t totalElements;
    uint32_t elementsPerCore;
    uint32_t tailCoreElements;
    uint32_t tileElements;
    uint32_t channels;
    uint32_t height;
    uint32_t width;
    uint32_t quantChannels;
};

struct MlvcInt8ToFp16TilingData {
    uint32_t totalElements;
    uint32_t elementsPerCore;
    uint32_t tailCoreElements;
    uint32_t tileElements;
};

#endif // MLVC_DECODE_PRIOR_TILING_H
