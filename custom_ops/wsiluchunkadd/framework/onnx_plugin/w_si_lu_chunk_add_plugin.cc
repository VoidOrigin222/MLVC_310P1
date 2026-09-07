/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * ------------------------------------------------------------------------- */

#include "register/register.h"
#include <vector>

namespace domi {
Status ParseParamNoop(const ge::Operator& op_src, ge::Operator& op_dest)
{
    return SUCCESS;
}

static const std::vector<ge::AscendString> kPostSigmoidChunkAddOriginOpTypes = {
    ge::AscendString("PostSigmoidChunkAdd"),
    ge::AscendString("ai.onnx::17::PostSigmoidChunkAdd"),
};

static const std::vector<ge::AscendString> kPostSigmoidMulOriginOpTypes = {
    ge::AscendString("PostSigmoidMul"),
    ge::AscendString("ai.onnx::17::PostSigmoidMul"),
};

static const std::vector<ge::AscendString> kScaledSiLUMulOriginOpTypes = {
    ge::AscendString("ScaledSiLUMul"),
    ge::AscendString("ai.onnx::17::ScaledSiLUMul"),
};

REGISTER_CUSTOM_OP("PostSigmoidChunkAdd")
    .FrameworkType(ONNX)
    .OriginOpType(kPostSigmoidChunkAddOriginOpTypes)
    .ParseParamsByOperatorFn(ParseParamNoop);

REGISTER_CUSTOM_OP("PostSigmoidMul")
    .FrameworkType(ONNX)
    .OriginOpType(kPostSigmoidMulOriginOpTypes)
    .ParseParamsByOperatorFn(ParseParamNoop);

REGISTER_CUSTOM_OP("ScaledSiLUMul")
    .FrameworkType(ONNX)
    .OriginOpType(kScaledSiLUMulOriginOpTypes)
    .ParseParamsByOperatorFn(ParseParamNoop);
} // namespace domi
