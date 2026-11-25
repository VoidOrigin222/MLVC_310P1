#include "ulbvc_decode_prior_kernels.h"

extern "C" __global__ __aicore__ void ulbvc_restore_y(
    GM_ADDR ySymbols, GM_ADDR means, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(UlbvcRestoreYTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelUlbvcRestoreY op;
    op.Init(ySymbols, means, output, tilingData);
    op.Process();
}
