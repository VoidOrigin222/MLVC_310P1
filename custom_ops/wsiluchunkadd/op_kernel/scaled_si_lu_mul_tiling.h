/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the
 * Mulan PSL v2.
 * ------------------------------------------------------------------------- */

#ifndef SCALED_SI_LU_MUL_TILING_H
#define SCALED_SI_LU_MUL_TILING_H

#include <cstdint>

struct ScaledSiLUMulTilingData {
    uint32_t totalElements;
    uint32_t elementsPerCore;
    uint32_t tailCoreElements;
    uint32_t tileElements;
    uint32_t dataType;
    uint32_t computeMode;
};

#endif // SCALED_SI_LU_MUL_TILING_H
