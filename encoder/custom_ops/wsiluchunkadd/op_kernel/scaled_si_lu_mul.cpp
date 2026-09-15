#include "kernel_operator.h"
#include "scaled_si_lu_mul_tiling.h"

using namespace AscendC;

constexpr int32_t kBufferNum = 2;

template <typename T>
class KernelScaledSiLUMul {
public:
    __aicore__ inline KernelScaledSiLUMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ScaledSiLUMulTilingData& tilingData)
    {
        totalElements_ = tilingData.totalElements;
        elementsPerCore_ = tilingData.elementsPerCore;
        tailCoreElements_ = tilingData.tailCoreElements;
        tileElements_ = tilingData.tileElements;
        computeMode_ = tilingData.computeMode;

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

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + coreOffset_, coreElements_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + coreOffset_, coreElements_);

        pipe_.InitBuffer(xQueue_, kBufferNum, tileElements_ * sizeof(T));
        pipe_.InitBuffer(outQueue_, kBufferNum, tileElements_ * sizeof(T));
        pipe_.InitBuffer(expBuf_, tileElements_ * sizeof(T));
        pipe_.InitBuffer(denomBuf_, tileElements_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        const uint32_t tileCount = (coreElements_ + tileElements_ - 1) / tileElements_;
        if (tileCount == 0) {
            return;
        }

        CopyIn(0);
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            if (tile + 1 < tileCount) {
                CopyIn(tile + 1);
            }
            Compute(TileElementCount(tile));
            CopyOut(TileOffset(tile), TileElementCount(tile));
        }
    }

private:
    __aicore__ inline uint32_t TileOffset(uint32_t tile) const
    {
        return tile * tileElements_;
    }

    __aicore__ inline uint32_t TileElementCount(uint32_t tile) const
    {
        const uint32_t offset = TileOffset(tile);
        const uint32_t remaining = coreElements_ - offset;
        return remaining < tileElements_ ? remaining : tileElements_;
    }

    __aicore__ inline void CopyIn(uint32_t tile)
    {
        const uint32_t offset = TileOffset(tile);
        const uint32_t count = TileElementCount(tile);
        LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
        DataCopy(xLocal, xGm_[offset], count);
        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<T> xLocal = xQueue_.DeQue<T>();
        LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();
        LocalTensor<T> expLocal = expBuf_.Get<T>();
        LocalTensor<T> denomLocal = denomBuf_.Get<T>();

        Muls(denomLocal, xLocal, static_cast<T>(-4.0), count);
        PipeBarrier<PIPE_V>();
        Exp(expLocal, denomLocal, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Adds(denomLocal, expLocal, static_cast<T>(1.0), count);
        PipeBarrier<PIPE_V>();
        if (computeMode_ == 1U) {
            Div(outLocal, xLocal, denomLocal, static_cast<int32_t>(count));
        } else {
            Duplicate(expLocal, static_cast<T>(1.0), static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
            Div(outLocal, expLocal, denomLocal, static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
            Mul(outLocal, outLocal, xLocal, count);
        }

        outQueue_.EnQue(outLocal);
        xQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        LocalTensor<T> outLocal = outQueue_.DeQue<T>();
        DataCopy(yGm_[offset], outLocal, count);
        outQueue_.FreeTensor(outLocal);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, kBufferNum> xQueue_;
    TQue<QuePosition::VECOUT, kBufferNum> outQueue_;
    TBuf<QuePosition::VECCALC> expBuf_;
    TBuf<QuePosition::VECCALC> denomBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    uint32_t totalElements_ = 0;
    uint32_t elementsPerCore_ = 0;
    uint32_t tailCoreElements_ = 0;
    uint32_t tileElements_ = 0;
    uint32_t computeMode_ = 0;
    uint32_t coreOffset_ = 0;
    uint32_t coreElements_ = 0;
};

extern "C" __global__ __aicore__ void scaled_si_lu_mul(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ScaledSiLUMulTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.dataType == 1U) {
        KernelScaledSiLUMul<float> op;
        op.Init(x, y, tilingData);
        op.Process();
    } else {
        KernelScaledSiLUMul<half> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
}
