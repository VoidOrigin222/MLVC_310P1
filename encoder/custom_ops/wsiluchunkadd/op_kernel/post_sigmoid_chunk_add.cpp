#include "kernel_operator.h"
#include "post_sigmoid_chunk_add_tiling.h"

using namespace AscendC;

constexpr int32_t kBufferNum = 2;

template <typename T>
class KernelPostSigmoidChunkAdd {
public:
    __aicore__ inline KernelPostSigmoidChunkAdd() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR sigmoidOut, GM_ADDR y,
                                const PostSigmoidChunkAddTilingData& tilingData)
    {
        totalOutputElements_ = tilingData.totalOutputElements;
        elementsPerCore_ = tilingData.elementsPerCore;
        tailCoreElements_ = tilingData.tailCoreElements;
        tileElements_ = tilingData.tileElements;
        inputHalfOffset_ = tilingData.inputHalfOffset;
        barrierMode_ = tilingData.barrierMode;

        const uint32_t blockIdx = GetBlockIdx();
        coreOffset_ = blockIdx * elementsPerCore_;
        coreElements_ = elementsPerCore_;
        if (blockIdx == GetBlockNum() - 1) {
            coreElements_ = tailCoreElements_;
        }
        if (coreOffset_ >= totalOutputElements_) {
            coreElements_ = 0;
        }
        if (coreOffset_ + coreElements_ > totalOutputElements_) {
            coreElements_ = totalOutputElements_ - coreOffset_;
        }

        x0Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + coreOffset_, coreElements_);
        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + inputHalfOffset_ + coreOffset_, coreElements_);
        sigmoid0Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(sigmoidOut) + coreOffset_, coreElements_);
        sigmoid1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(sigmoidOut) + inputHalfOffset_ + coreOffset_,
                                    coreElements_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + coreOffset_, coreElements_);

        pipe_.InitBuffer(x0Queue_, kBufferNum, tileElements_ * sizeof(T));
        pipe_.InitBuffer(x1Queue_, kBufferNum, tileElements_ * sizeof(T));
        pipe_.InitBuffer(sigmoid0Queue_, kBufferNum, tileElements_ * sizeof(T));
        pipe_.InitBuffer(sigmoid1Queue_, kBufferNum, tileElements_ * sizeof(T));
        pipe_.InitBuffer(outQueue_, kBufferNum, tileElements_ * sizeof(T));
        pipe_.InitBuffer(termBuf_, tileElements_ * sizeof(T));
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
        LocalTensor<T> x0Local = x0Queue_.AllocTensor<T>();
        LocalTensor<T> x1Local = x1Queue_.AllocTensor<T>();
        LocalTensor<T> sigmoid0Local = sigmoid0Queue_.AllocTensor<T>();
        LocalTensor<T> sigmoid1Local = sigmoid1Queue_.AllocTensor<T>();
        DataCopy(x0Local, x0Gm_[offset], count);
        DataCopy(x1Local, x1Gm_[offset], count);
        DataCopy(sigmoid0Local, sigmoid0Gm_[offset], count);
        DataCopy(sigmoid1Local, sigmoid1Gm_[offset], count);
        x0Queue_.EnQue(x0Local);
        x1Queue_.EnQue(x1Local);
        sigmoid0Queue_.EnQue(sigmoid0Local);
        sigmoid1Queue_.EnQue(sigmoid1Local);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<T> x0Local = x0Queue_.DeQue<T>();
        LocalTensor<T> x1Local = x1Queue_.DeQue<T>();
        LocalTensor<T> sigmoid0Local = sigmoid0Queue_.DeQue<T>();
        LocalTensor<T> sigmoid1Local = sigmoid1Queue_.DeQue<T>();
        LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();
        LocalTensor<T> termLocal = termBuf_.Get<T>();

        Mul(outLocal, x0Local, sigmoid0Local, count);
        if (barrierMode_ == 0U) {
            PipeBarrier<PIPE_V>();
        }
        Mul(termLocal, x1Local, sigmoid1Local, count);
        PipeBarrier<PIPE_V>();
        Add(outLocal, outLocal, termLocal, count);

        outQueue_.EnQue(outLocal);
        x0Queue_.FreeTensor(x0Local);
        x1Queue_.FreeTensor(x1Local);
        sigmoid0Queue_.FreeTensor(sigmoid0Local);
        sigmoid1Queue_.FreeTensor(sigmoid1Local);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        LocalTensor<T> outLocal = outQueue_.DeQue<T>();
        DataCopy(yGm_[offset], outLocal, count);
        outQueue_.FreeTensor(outLocal);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, kBufferNum> x0Queue_;
    TQue<QuePosition::VECIN, kBufferNum> x1Queue_;
    TQue<QuePosition::VECIN, kBufferNum> sigmoid0Queue_;
    TQue<QuePosition::VECIN, kBufferNum> sigmoid1Queue_;
    TQue<QuePosition::VECOUT, kBufferNum> outQueue_;
    TBuf<QuePosition::VECCALC> termBuf_;
    GlobalTensor<T> x0Gm_;
    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> sigmoid0Gm_;
    GlobalTensor<T> sigmoid1Gm_;
    GlobalTensor<T> yGm_;
    uint32_t totalOutputElements_ = 0;
    uint32_t elementsPerCore_ = 0;
    uint32_t tailCoreElements_ = 0;
    uint32_t tileElements_ = 0;
    uint32_t inputHalfOffset_ = 0;
    uint32_t barrierMode_ = 0;
    uint32_t coreOffset_ = 0;
    uint32_t coreElements_ = 0;
};

extern "C" __global__ __aicore__ void post_sigmoid_chunk_add(GM_ADDR x, GM_ADDR sigmoidOut, GM_ADDR y,
                                                              GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(PostSigmoidChunkAddTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.dataType == 1U) {
        KernelPostSigmoidChunkAdd<float> op;
        op.Init(x, sigmoidOut, y, tilingData);
        op.Process();
    } else {
        KernelPostSigmoidChunkAdd<half> op;
        op.Init(x, sigmoidOut, y, tilingData);
        op.Process();
    }
}
