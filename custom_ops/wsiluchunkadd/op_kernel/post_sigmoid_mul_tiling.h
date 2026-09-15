/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the
 * Mulan PSL v2.
 * ------------------------------------------------------------------------- */

#ifndef POST_SIGMOID_MUL_TILING_H
#define POST_SIGMOID_MUL_TILING_H

#include <cstdint>

struct PostSigmoidMulTilingData {
    uint32_t totalElements;
    uint32_t elementsPerCore;
    uint32_t tailCoreElements;
    uint32_t tileElements;
    uint32_t dataType;
};

#endif // POST_SIGMOID_MUL_TILING_H
