#ifndef MLVC_BUILD_INDEX_ENCODE_TILING_H
#define MLVC_BUILD_INDEX_ENCODE_TILING_H

#include <cstdint>

struct MlvcBuildIndexEncodeTilingData {
    uint32_t totalElements;
    uint32_t elementsPerCore;
    uint32_t tailCoreElements;
    uint32_t tileElements;
    uint32_t symbolType;
    float forceZeroThres;
};

#endif // MLVC_BUILD_INDEX_ENCODE_TILING_H
