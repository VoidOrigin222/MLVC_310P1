#include "mlvc_decode_prior_kernels.h"

extern "C" __global__ __aicore__ void mlvc_single_part(
    GM_ADDR full, GM_ADDR part, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MlvcSinglePartTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelMlvcSinglePart op;
    op.Init(full, part, tilingData);
    op.Process();
}
