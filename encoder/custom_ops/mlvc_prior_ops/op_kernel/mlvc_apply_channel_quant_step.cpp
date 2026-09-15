#include "mlvc_decode_prior_kernels.h"

extern "C" __global__ __aicore__ void mlvc_apply_channel_quant_step(
    GM_ADDR quantStep, GM_ADDR values, GM_ADDR valuesOut, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MlvcApplyChannelQuantStepTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelMlvcApplyChannelQuantStep op;
    op.Init(quantStep, values, valuesOut, tilingData);
    op.Process();
}
