#include "kernel_operator.h"
#include "mlvc_fp16_yuv444_to_nv12_tiling.h"

using namespace AscendC;

class KernelMlvcFp16Yuv444ToNv12 {
public:
    __aicore__ inline KernelMlvcFp16Yuv444ToNv12() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                                const MlvcFp16Yuv444ToNv12TilingData& tiling)
    {
        inputHeight_ = tiling.inputHeight;
        inputWidth_ = tiling.inputWidth;
        visibleWidth_ = tiling.visibleWidth;
        visibleHeight_ = tiling.visibleHeight;
        widthStride_ = tiling.widthStride;
        heightStride_ = tiling.heightStride;
        outputRows_ = tiling.outputRows;
        planeElements_ = inputHeight_ * inputWidth_;

        const uint32_t blockIdx = GetBlockIdx();
        coreRowOffset_ = blockIdx * tiling.rowsPerCore;
        coreRows_ = blockIdx == GetBlockNum() - 1 ? tiling.tailCoreRows
                                                  : tiling.rowsPerCore;
        if (coreRowOffset_ >= outputRows_) {
            coreRows_ = 0;
        } else if (coreRowOffset_ + coreRows_ > outputRows_) {
            coreRows_ = outputRows_ - coreRowOffset_;
        }

        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input),
                                 3U * planeElements_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output),
                                  outputRows_ * widthStride_);
        pipe_.InitBuffer(inputQueue_, 2, inputWidth_ * sizeof(half));
        pipe_.InitBuffer(outputQueue_, 1, widthStride_ * sizeof(uint8_t));
        pipe_.InitBuffer(uBuffer_, visibleWidth_ * sizeof(uint8_t));
        pipe_.InitBuffer(vBuffer_, visibleWidth_ * sizeof(uint8_t));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t localRow = 0; localRow < coreRows_; ++localRow) {
            const uint32_t outputRow = coreRowOffset_ + localRow;
            if (outputRow < heightStride_) {
                ProcessYRow(outputRow);
            } else {
                ProcessUvRow(outputRow - heightStride_, outputRow);
            }
        }
    }

private:
    __aicore__ inline void CopyInputRow(uint32_t plane, uint32_t row)
    {
        LocalTensor<half> input = inputQueue_.AllocTensor<half>();
        DataCopy(input, inputGm_[plane * planeElements_ + row * inputWidth_],
                 visibleWidth_);
        inputQueue_.EnQue(input);
    }

    __aicore__ inline uint8_t ToByte(float value) const
    {
        const float clamped = value < 0.0F ? 0.0F : (value > 1.0F ? 1.0F : value);
        const int32_t rounded =
            Cast<float, int32_t, RoundMode::CAST_RINT>(clamped * 255.0F);
        return static_cast<uint8_t>(rounded);
    }

    __aicore__ inline void ProcessYRow(uint32_t outputRow)
    {
        LocalTensor<uint8_t> output = outputQueue_.AllocTensor<uint8_t>();
        LocalTensor<uint32_t> outputWords = output.ReinterpretCast<uint32_t>();
        Duplicate(outputWords, uint32_t{0}, widthStride_ / sizeof(uint32_t));
        PipeBarrier<PIPE_V>();
        if (outputRow < visibleHeight_) {
            CopyInputRow(0, outputRow);
            LocalTensor<half> input = inputQueue_.DeQue<half>();
            for (uint32_t x = 0; x < visibleWidth_; ++x) {
                output.SetValue(x, ToByte(static_cast<float>(input.GetValue(x))));
            }
            inputQueue_.FreeTensor(input);
        }
        outputQueue_.EnQue(output);
        LocalTensor<uint8_t> ready = outputQueue_.DeQue<uint8_t>();
        DataCopy(outputGm_[outputRow * widthStride_], ready, widthStride_);
        outputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void AveragePlaneRow(uint32_t plane, uint32_t inputRow,
                                           const LocalTensor<uint8_t>& bytes)
    {
        CopyInputRow(plane, inputRow);
        CopyInputRow(plane, inputRow + 1U);
        LocalTensor<half> row0 = inputQueue_.DeQue<half>();
        LocalTensor<half> row1 = inputQueue_.DeQue<half>();
        for (uint32_t x = 0; x < visibleWidth_ / 2U; ++x) {
            const uint32_t sourceX = 2U * x;
            const float average =
                (static_cast<float>(row0.GetValue(sourceX)) +
                 static_cast<float>(row0.GetValue(sourceX + 1U)) +
                 static_cast<float>(row1.GetValue(sourceX)) +
                 static_cast<float>(row1.GetValue(sourceX + 1U))) * 0.25F;
            bytes.SetValue(x, ToByte(average));
        }
        inputQueue_.FreeTensor(row0);
        inputQueue_.FreeTensor(row1);
    }

    __aicore__ inline void ProcessUvRow(uint32_t uvRow, uint32_t outputRow)
    {
        LocalTensor<uint8_t> output = outputQueue_.AllocTensor<uint8_t>();
        LocalTensor<uint32_t> outputWords = output.ReinterpretCast<uint32_t>();
        Duplicate(outputWords, uint32_t{0}, widthStride_ / sizeof(uint32_t));
        PipeBarrier<PIPE_V>();
        if (uvRow < visibleHeight_ / 2U) {
            LocalTensor<uint8_t> u = uBuffer_.Get<uint8_t>();
            LocalTensor<uint8_t> v = vBuffer_.Get<uint8_t>();
            AveragePlaneRow(1, 2U * uvRow, u);
            AveragePlaneRow(2, 2U * uvRow, v);
            for (uint32_t x = 0; x < visibleWidth_ / 2U; ++x) {
                output.SetValue(2U * x, u.GetValue(x));
                output.SetValue(2U * x + 1U, v.GetValue(x));
            }
        }
        outputQueue_.EnQue(output);
        LocalTensor<uint8_t> ready = outputQueue_.DeQue<uint8_t>();
        DataCopy(outputGm_[outputRow * widthStride_], ready, widthStride_);
        outputQueue_.FreeTensor(ready);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<QuePosition::VECCALC> uBuffer_;
    TBuf<QuePosition::VECCALC> vBuffer_;
    GlobalTensor<half> inputGm_;
    GlobalTensor<uint8_t> outputGm_;
    uint32_t inputHeight_ = 0;
    uint32_t inputWidth_ = 0;
    uint32_t visibleWidth_ = 0;
    uint32_t visibleHeight_ = 0;
    uint32_t widthStride_ = 0;
    uint32_t heightStride_ = 0;
    uint32_t outputRows_ = 0;
    uint32_t planeElements_ = 0;
    uint32_t coreRowOffset_ = 0;
    uint32_t coreRows_ = 0;
};

extern "C" __global__ __aicore__ void mlvc_fp16_yuv444_to_nv12(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MlvcFp16Yuv444ToNv12TilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelMlvcFp16Yuv444ToNv12 op;
    op.Init(input, output, tilingData);
    op.Process();
}
