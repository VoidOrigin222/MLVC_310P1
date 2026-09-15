#include "mlvc_decode_prior_kernels.h"

extern "C" __global__ __aicore__ void mlvc_int8_to_fp16(
    GM_ADDR symbols, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MlvcInt8ToFp16TilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelMlvcInt8ToFp16 op;
    op.Init(symbols, output, tilingData);
    op.Process();
}
