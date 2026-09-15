#include "kernel_operator.h"
#include "mlvc_build_index_encode_tiling.h"

using namespace AscendC;

namespace {
constexpr float kScaleMin = 0.11F;
constexpr float kScaleMax = 16.0F;
}

class KernelMlvcBuildIndexEncode {
public:
    __aicore__ inline KernelMlvcBuildIndexEncode() {}

    __aicore__ inline void Init(GM_ADDR symbols, GM_ADDR scales, GM_ADDR lookup,
                                GM_ADDR combined, GM_ADDR keepMask,
                                const MlvcBuildIndexEncodeTilingData& tilingData)
    {
        totalElements_ = tilingData.totalElements;
        elementsPerCore_ = tilingData.elementsPerCore;
        tailCoreElements_ = tilingData.tailCoreElements;
        tileElements_ = tilingData.tileElements;
        symbolType_ = tilingData.symbolType;
        forceZeroThres_ = tilingData.forceZeroThres;

        const uint32_t blockIdx = GetBlockIdx();
        coreOffset_ = blockIdx * elementsPerCore_;
        coreElements_ = elementsPerCore_;
        if (blockIdx == GetBlockNum() - 1) {
            coreElements_ = tailCoreElements_;
        }
        if (coreOffset_ >= totalElements_) {
            coreElements_ = 0;
        }
        if (coreOffset_ + coreElements_ > totalElements_) {
            coreElements_ = totalElements_ - coreOffset_;
        }

        symbolFp16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(symbols) + coreOffset_, coreElements_);
        symbolInt8Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(symbols) + coreOffset_, coreElements_);
        scaleFp16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(scales) + coreOffset_, coreElements_);
        scaleBitsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(scales) + coreOffset_, coreElements_);
        lookupGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(lookup), 65536);
        combinedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t*>(combined) + coreOffset_, coreElements_);
        keepMaskGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(keepMask) + coreOffset_, coreElements_);
    }

    __aicore__ inline void Process()
    {
        const uint32_t tileCount = (coreElements_ + tileElements_ - 1) / tileElements_;
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t offset = tile * tileElements_;
            const uint32_t count = (coreElements_ - offset) < tileElements_
                                       ? (coreElements_ - offset)
                                       : tileElements_;
            ComputeTile(offset, count);
        }
    }

private:
    __aicore__ inline void ComputeTile(uint32_t offset, uint32_t count)
    {
        for (uint32_t i = 0; i < count; ++i) {
            bool keep = true;
            const uint32_t index = offset + i;
            const int32_t symbol = SymbolValue(index);
            const uint8_t scaleIndex = ScaleIndex(index, &keep);
            combinedGm_.SetValue(index,
                                 static_cast<int16_t>((symbol << 8) + static_cast<int32_t>(scaleIndex)));
            keepMaskGm_.SetValue(index, keep ? uint8_t{1} : uint8_t{0});
        }
    }

    __aicore__ inline int32_t SymbolValue(uint32_t index) const
    {
        if (symbolType_ == 0U) {
            const half value = symbolFp16Gm_.GetValue(index);
            const float asFloat = static_cast<float>(value);
            return asFloat >= 0.0F ? static_cast<int32_t>(asFloat + 0.5F)
                                   : static_cast<int32_t>(asFloat - 0.5F);
        }
        return static_cast<int32_t>(symbolInt8Gm_.GetValue(index));
    }

    __aicore__ inline uint8_t ScaleIndex(uint32_t index, bool* keep) const
    {
        const half scaleHalf = scaleFp16Gm_.GetValue(index);
        const float scale = static_cast<float>(scaleHalf);
        const float clipped = scale < kScaleMin ? kScaleMin : (scale > kScaleMax ? kScaleMax : scale);
        *keep = forceZeroThres_ < 0.0F || clipped > forceZeroThres_;
        return lookupGm_.GetValue(static_cast<uint32_t>(scaleBitsGm_.GetValue(index)));
    }

    GlobalTensor<half> symbolFp16Gm_;
    GlobalTensor<int8_t> symbolInt8Gm_;
    GlobalTensor<half> scaleFp16Gm_;
    GlobalTensor<uint16_t> scaleBitsGm_;
    GlobalTensor<uint8_t> lookupGm_;
    GlobalTensor<int16_t> combinedGm_;
    GlobalTensor<uint8_t> keepMaskGm_;
    uint32_t totalElements_ = 0;
    uint32_t elementsPerCore_ = 0;
    uint32_t tailCoreElements_ = 0;
    uint32_t tileElements_ = 0;
    uint32_t symbolType_ = 0;
    uint32_t coreOffset_ = 0;
    uint32_t coreElements_ = 0;
    float forceZeroThres_ = -1.0F;
};

extern "C" __global__ __aicore__ void mlvc_build_index_encode(
    GM_ADDR symbols, GM_ADDR scales, GM_ADDR fp16ScaleIndexLookup,
    GM_ADDR combined, GM_ADDR keepMask, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MlvcBuildIndexEncodeTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelMlvcBuildIndexEncode op;
    op.Init(symbols, scales, fp16ScaleIndexLookup, combined, keepMask, tilingData);
    op.Process();
}
