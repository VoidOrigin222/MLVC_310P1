#include "ulbvc_decode_prior_kernels.h"

extern "C" __global__ __aicore__ void ulbvc_single_part(
    GM_ADDR full, GM_ADDR part, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(UlbvcSinglePartTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelUlbvcSinglePart op;
    op.Init(full, part, tilingData);
    op.Process();
}
