#ifndef ULBVC_BUILD_INDEX_ENCODE_TILING_H
#define ULBVC_BUILD_INDEX_ENCODE_TILING_H

#include <cstdint>

struct UlbvcBuildIndexEncodeTilingData {
    uint32_t totalElements;
    uint32_t elementsPerCore;
    uint32_t tailCoreElements;
    uint32_t tileElements;
    uint32_t symbolType;
    float forceZeroThres;
};

#endif // ULBVC_BUILD_INDEX_ENCODE_TILING_H
