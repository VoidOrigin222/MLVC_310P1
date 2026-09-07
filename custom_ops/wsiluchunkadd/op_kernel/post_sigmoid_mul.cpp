#include "kernel_operator.h"
#include "post_sigmoid_mul_tiling.h"

using namespace AscendC;

constexpr int32_t kBufferNum = 2;

template <typename T>
class KernelPostSigmoidMul {
public:
    __aicore__ inline KernelPostSigmoidMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR sigmoidOut, GM_ADDR y,
                                const PostSigmoidMulTilingData& tilingData)
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

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + coreOffset_, coreElements_);
        sigmoidGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(sigmoidOut) + coreOffset_, coreElements_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + coreOffset_, coreElements_);

        pipe_.InitBuffer(xQueue_, kBufferNum, tileElements_ * sizeof(T));
        pipe_.InitBuffer(sigmoidQueue_, kBufferNum, tileElements_ * sizeof(T));
        pipe_.InitBuffer(outQueue_, kBufferNum, tileElements_ * sizeof(T));
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
        LocalTensor<T> sigmoidLocal = sigmoidQueue_.AllocTensor<T>();
        DataCopy(xLocal, xGm_[offset], count);
        DataCopy(sigmoidLocal, sigmoidGm_[offset], count);
        xQueue_.EnQue(xLocal);
        sigmoidQueue_.EnQue(sigmoidLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<T> xLocal = xQueue_.DeQue<T>();
        LocalTensor<T> sigmoidLocal = sigmoidQueue_.DeQue<T>();
        LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();

        Mul(outLocal, xLocal, sigmoidLocal, count);

        outQueue_.EnQue(outLocal);
        xQueue_.FreeTensor(xLocal);
        sigmoidQueue_.FreeTensor(sigmoidLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        LocalTensor<T> outLocal = outQueue_.DeQue<T>();
        DataCopy(yGm_[offset], outLocal, count);
        outQueue_.FreeTensor(outLocal);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, kBufferNum> xQueue_;
    TQue<QuePosition::VECIN, kBufferNum> sigmoidQueue_;
    TQue<QuePosition::VECOUT, kBufferNum> outQueue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> sigmoidGm_;
    GlobalTensor<T> yGm_;
    uint32_t totalElements_ = 0;
    uint32_t elementsPerCore_ = 0;
    uint32_t tailCoreElements_ = 0;
    uint32_t tileElements_ = 0;
    uint32_t coreOffset_ = 0;
    uint32_t coreElements_ = 0;
};

extern "C" __global__ __aicore__ void post_sigmoid_mul(GM_ADDR x, GM_ADDR sigmoidOut, GM_ADDR y,
                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(PostSigmoidMulTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.dataType == 1U) {
        KernelPostSigmoidMul<float> op;
        op.Init(x, sigmoidOut, y, tilingData);
        op.Process();
    } else {
        KernelPostSigmoidMul<half> op;
        op.Init(x, sigmoidOut, y, tilingData);
        op.Process();
    }
}
