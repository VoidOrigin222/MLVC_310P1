#ifndef MLVC_DECODE_PRIOR_KERNELS_H
#define MLVC_DECODE_PRIOR_KERNELS_H

#include "kernel_operator.h"
#include "mlvc_decode_prior_tiling.h"

using namespace AscendC;

namespace {

constexpr uint32_t kDataTypeFp16 = 0U;
constexpr uint32_t kDataTypeFp32 = 1U;

__aicore__ inline uint32_t MicroMask4(uint32_t maskIndex, uint32_t chunk, uint32_t y, uint32_t x)
{
    const uint32_t selector = ((maskIndex & 3U) << 2) | (chunk & 3U);
    const uint32_t parity = ((y & 1U) << 1) | (x & 1U);
    if (selector == 0U) {
        return parity == 0U ? 1U : 0U;
    }
    if (selector == 1U) {
        return parity == 1U ? 1U : 0U;
    }
    if (selector == 2U) {
        return parity == 2U ? 1U : 0U;
    }
    if (selector == 3U) {
        return parity == 3U ? 1U : 0U;
    }
    if (selector == 4U) {
        return parity == 3U ? 1U : 0U;
    }
    if (selector == 5U) {
        return parity == 2U ? 1U : 0U;
    }
    if (selector == 6U) {
        return parity == 1U ? 1U : 0U;
    }
    if (selector == 7U) {
        return parity == 0U ? 1U : 0U;
    }
    if (selector == 8U) {
        return parity == 2U ? 1U : 0U;
    }
    if (selector == 9U) {
        return parity == 3U ? 1U : 0U;
    }
    if (selector == 10U) {
        return parity == 0U ? 1U : 0U;
    }
    if (selector == 11U) {
        return parity == 1U ? 1U : 0U;
    }
    if (selector == 12U) {
        return parity == 1U ? 1U : 0U;
    }
    if (selector == 13U) {
        return parity == 0U ? 1U : 0U;
    }
    if (selector == 14U) {
        return parity == 3U ? 1U : 0U;
    }
    return parity == 2U ? 1U : 0U;
}

__aicore__ inline uint32_t MicroMask2(uint32_t maskIndex, uint32_t chunk, uint32_t y, uint32_t x)
{
    const uint32_t parity = ((y & 1U) << 1) | (x & 1U);
    if ((maskIndex & 1U) == 0U) {
        if ((chunk & 1U) == 0U) {
            return (parity == 0U || parity == 3U) ? 1U : 0U;
        }
        return (parity == 1U || parity == 2U) ? 1U : 0U;
    }
    if ((chunk & 1U) == 0U) {
        return (parity == 1U || parity == 2U) ? 1U : 0U;
    }
    return (parity == 0U || parity == 3U) ? 1U : 0U;
}

__aicore__ inline uint32_t SelectedChunk(uint32_t parts, uint32_t maskIndex, uint32_t y, uint32_t x)
{
    if (parts == 4U) {
        for (uint32_t chunk = 0; chunk < 4U; ++chunk) {
            if (MicroMask4(maskIndex, chunk, y, x) != 0U) {
                return chunk;
            }
        }
        return 0U;
    }
    return MicroMask2(maskIndex, 0U, y, x) != 0U ? 0U : 1U;
}

__aicore__ inline float HalfToFloat(half value)
{
    return static_cast<float>(value);
}

} // namespace

class KernelMlvcSinglePart {
public:
    __aicore__ inline KernelMlvcSinglePart() {}

    __aicore__ inline void Init(GM_ADDR full, GM_ADDR part, const MlvcSinglePartTilingData& tilingData)
    {
        totalElements_ = tilingData.totalElements;
        elementsPerCore_ = tilingData.elementsPerCore;
        tailCoreElements_ = tilingData.tailCoreElements;
        tileElements_ = tilingData.tileElements;
        channels_ = tilingData.channels;
        height_ = tilingData.height;
        width_ = tilingData.width;
        partChannels_ = tilingData.partChannels;
        maskIndex_ = tilingData.maskIndex;
        parts_ = tilingData.parts;
        dataType_ = tilingData.dataType;

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

        fullFp16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(full), channels_ * height_ * width_);
        partFp16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(part), totalElements_);
        fullFp32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(full), channels_ * height_ * width_);
        partFp32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(part), totalElements_);
    }

    __aicore__ inline void Process()
    {
        const uint32_t tileCount = (coreElements_ + tileElements_ - 1U) / tileElements_;
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t offset = tile * tileElements_;
            const uint32_t count = (coreElements_ - offset) < tileElements_ ? (coreElements_ - offset)
                                                                            : tileElements_;
            ComputeTile(offset, count);
        }
    }

private:
    __aicore__ inline void ComputeTile(uint32_t offset, uint32_t count)
    {
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t linear = coreOffset_ + offset + i;
            const uint32_t x = linear % width_;
            const uint32_t y = (linear / width_) % height_;
            const uint32_t c = linear / (height_ * width_);
            const uint32_t chunk = SelectedChunk(parts_, maskIndex_, y, x);
            const uint32_t sourceIndex = ((chunk * partChannels_ + c) * height_ + y) * width_ + x;
            if (dataType_ == kDataTypeFp32) {
                partFp32Gm_.SetValue(linear, fullFp32Gm_.GetValue(sourceIndex));
            } else {
                partFp16Gm_.SetValue(linear, fullFp16Gm_.GetValue(sourceIndex));
            }
        }
    }

    GlobalTensor<half> fullFp16Gm_;
    GlobalTensor<half> partFp16Gm_;
    GlobalTensor<float> fullFp32Gm_;
    GlobalTensor<float> partFp32Gm_;
    uint32_t totalElements_ = 0;
    uint32_t elementsPerCore_ = 0;
    uint32_t tailCoreElements_ = 0;
    uint32_t tileElements_ = 0;
    uint32_t channels_ = 0;
    uint32_t height_ = 0;
    uint32_t width_ = 0;
    uint32_t partChannels_ = 0;
    uint32_t maskIndex_ = 0;
    uint32_t parts_ = 0;
    uint32_t dataType_ = 0;
    uint32_t coreOffset_ = 0;
    uint32_t coreElements_ = 0;
};

class KernelMlvcRestoreY {
public:
    __aicore__ inline KernelMlvcRestoreY() {}

    __aicore__ inline void Init(GM_ADDR symbols, GM_ADDR means, GM_ADDR output,
                                const MlvcRestoreYTilingData& tilingData)
    {
        totalPartElements_ = tilingData.totalPartElements;
        totalFullElements_ = tilingData.totalFullElements;
        elementsPerCore_ = tilingData.elementsPerCore;
        tailCoreElements_ = tilingData.tailCoreElements;
        tileElements_ = tilingData.tileElements;
        channels_ = tilingData.channels;
        height_ = tilingData.height;
        width_ = tilingData.width;
        partChannels_ = tilingData.partChannels;
        maskIndex_ = tilingData.maskIndex;
        parts_ = tilingData.parts;

        const uint32_t blockIdx = GetBlockIdx();
        coreOffset_ = blockIdx * elementsPerCore_;
        coreElements_ = elementsPerCore_;
        if (blockIdx == GetBlockNum() - 1) {
            coreElements_ = tailCoreElements_;
        }
        if (coreOffset_ >= totalPartElements_) {
            coreElements_ = 0;
        }
        if (coreOffset_ + coreElements_ > totalPartElements_) {
            coreElements_ = totalPartElements_ - coreOffset_;
        }

        symbolsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(symbols), totalPartElements_);
        meansGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(means), totalFullElements_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output), totalFullElements_);
    }

    __aicore__ inline void Process()
    {
        const uint32_t tileCount = (coreElements_ + tileElements_ - 1U) / tileElements_;
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t offset = tile * tileElements_;
            const uint32_t count = (coreElements_ - offset) < tileElements_ ? (coreElements_ - offset)
                                                                            : tileElements_;
            ComputeTile(offset, count);
        }
    }

private:
    __aicore__ inline void ComputeTile(uint32_t offset, uint32_t count)
    {
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t linear = coreOffset_ + offset + i;
            const uint32_t x = linear % width_;
            const uint32_t y = (linear / width_) % height_;
            const uint32_t c = linear / (height_ * width_);
            const uint32_t chunk = SelectedChunk(parts_, maskIndex_, y, x);
            const uint32_t targetIndex = ((chunk * partChannels_ + c) * height_ + y) * width_ + x;
            const float value = static_cast<float>(symbolsGm_.GetValue(linear)) +
                                HalfToFloat(meansGm_.GetValue(targetIndex));
            outputGm_.SetValue(targetIndex, static_cast<half>(value));
        }
    }

    GlobalTensor<int8_t> symbolsGm_;
    GlobalTensor<half> meansGm_;
    GlobalTensor<half> outputGm_;
    uint32_t totalPartElements_ = 0;
    uint32_t totalFullElements_ = 0;
    uint32_t elementsPerCore_ = 0;
    uint32_t tailCoreElements_ = 0;
    uint32_t tileElements_ = 0;
    uint32_t channels_ = 0;
    uint32_t height_ = 0;
    uint32_t width_ = 0;
    uint32_t partChannels_ = 0;
    uint32_t maskIndex_ = 0;
    uint32_t parts_ = 0;
    uint32_t coreOffset_ = 0;
    uint32_t coreElements_ = 0;
};

class KernelMlvcApplyChannelQuantStep {
public:
    __aicore__ inline KernelMlvcApplyChannelQuantStep() {}

    __aicore__ inline void Init(GM_ADDR quantStep, GM_ADDR values, GM_ADDR valuesOut,
                                const MlvcApplyChannelQuantStepTilingData& tilingData)
    {
        totalElements_ = tilingData.totalElements;
        elementsPerCore_ = tilingData.elementsPerCore;
        tailCoreElements_ = tilingData.tailCoreElements;
        tileElements_ = tilingData.tileElements;
        channels_ = tilingData.channels;
        height_ = tilingData.height;
        width_ = tilingData.width;
        quantChannels_ = tilingData.quantChannels;

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

        quantStepGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(quantStep),
                                     quantChannels_ * height_ * width_);
        valuesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(values), totalElements_);
        valuesOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(valuesOut), totalElements_);
    }

    __aicore__ inline void Process()
    {
        const uint32_t tileCount = (coreElements_ + tileElements_ - 1U) / tileElements_;
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t offset = tile * tileElements_;
            const uint32_t count = (coreElements_ - offset) < tileElements_ ? (coreElements_ - offset)
                                                                            : tileElements_;
            ComputeTile(offset, count);
        }
    }

private:
    __aicore__ inline void ComputeTile(uint32_t offset, uint32_t count)
    {
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t linear = coreOffset_ + offset + i;
            const uint32_t x = linear % width_;
            const uint32_t y = (linear / width_) % height_;
            const uint32_t c = linear / (height_ * width_);
            const uint32_t quantChannel = quantChannels_ == 1U ? 0U : c;
            const uint32_t quantIndex = (quantChannel * height_ + y) * width_ + x;
            const float value = HalfToFloat(valuesGm_.GetValue(linear)) *
                                HalfToFloat(quantStepGm_.GetValue(quantIndex));
            valuesOutGm_.SetValue(linear, static_cast<half>(value));
        }
    }

    GlobalTensor<half> quantStepGm_;
    GlobalTensor<half> valuesGm_;
    GlobalTensor<half> valuesOutGm_;
    uint32_t totalElements_ = 0;
    uint32_t elementsPerCore_ = 0;
    uint32_t tailCoreElements_ = 0;
    uint32_t tileElements_ = 0;
    uint32_t channels_ = 0;
    uint32_t height_ = 0;
    uint32_t width_ = 0;
    uint32_t quantChannels_ = 0;
    uint32_t coreOffset_ = 0;
    uint32_t coreElements_ = 0;
};

class KernelMlvcInt8ToFp16 {
public:
    __aicore__ inline KernelMlvcInt8ToFp16() {}

    __aicore__ inline void Init(GM_ADDR symbols, GM_ADDR output, const MlvcInt8ToFp16TilingData& tilingData)
    {
        totalElements_ = tilingData.totalElements;
        elementsPerCore_ = tilingData.elementsPerCore;
        tailCoreElements_ = tilingData.tailCoreElements;
        tileElements_ = tilingData.tileElements;

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

        symbolsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(symbols), totalElements_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output), totalElements_);
    }

    __aicore__ inline void Process()
    {
        const uint32_t tileCount = (coreElements_ + tileElements_ - 1U) / tileElements_;
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t offset = tile * tileElements_;
            const uint32_t count = (coreElements_ - offset) < tileElements_ ? (coreElements_ - offset)
                                                                            : tileElements_;
            ComputeTile(offset, count);
        }
    }

private:
    __aicore__ inline void ComputeTile(uint32_t offset, uint32_t count)
    {
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t linear = coreOffset_ + offset + i;
            outputGm_.SetValue(linear, static_cast<half>(static_cast<float>(symbolsGm_.GetValue(linear))));
        }
    }

    GlobalTensor<int8_t> symbolsGm_;
    GlobalTensor<half> outputGm_;
    uint32_t totalElements_ = 0;
    uint32_t elementsPerCore_ = 0;
    uint32_t tailCoreElements_ = 0;
    uint32_t tileElements_ = 0;
    uint32_t coreOffset_ = 0;
    uint32_t coreElements_ = 0;
};

#endif // MLVC_DECODE_PRIOR_KERNELS_H
