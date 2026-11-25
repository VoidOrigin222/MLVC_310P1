#include "ulbvc_decode_prior_kernels.h"

extern "C" __global__ __aicore__ void ulbvc_apply_channel_quant_step(
    GM_ADDR quantStep, GM_ADDR values, GM_ADDR valuesOut, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(UlbvcApplyChannelQuantStepTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelUlbvcApplyChannelQuantStep op;
    op.Init(quantStep, values, valuesOut, tilingData);
    op.Process();
}
