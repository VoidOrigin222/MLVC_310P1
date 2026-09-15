#include "../op_kernel/mlvc_fp16_yuv444_to_nv12_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr size_t kWidthAttr = 0;
constexpr size_t kHeightAttr = 1;
constexpr size_t kWidthStrideAttr = 2;
constexpr size_t kHeightStrideAttr = 3;
constexpr size_t kBlockDimAttr = 4;
constexpr uint32_t kDefaultBlockDim = 32;

uint32_t Attr(gert::TilingContext* context, size_t index, uint32_t fallback)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr || attrs->GetAttrNum() <= index) {
        return fallback;
    }
    const int64_t* value = attrs->GetAttrPointer<int64_t>(index);
    return value != nullptr && *value > 0 && *value <= static_cast<int64_t>(UINT32_MAX)
               ? static_cast<uint32_t>(*value)
               : fallback;
}

uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1U) / divisor;
}
}  // namespace

namespace optiling {
static ge::graphStatus MlvcFp16Yuv444ToNv12TilingFunc(gert::TilingContext* context)
{
    auto* tiling = context->GetTilingData<MlvcFp16Yuv444ToNv12TilingData>();
    const gert::StorageShape* input = context->GetInputShape(0);
    const gert::StorageShape* output = context->GetOutputShape(0);
    const auto* inputDesc = context->GetInputDesc(0);
    const auto* outputDesc = context->GetOutputDesc(0);
    if (tiling == nullptr || input == nullptr || output == nullptr || inputDesc == nullptr ||
        outputDesc == nullptr || input->GetStorageShape().GetDimNum() != 4 ||
        output->GetStorageShape().GetDimNum() != 2 ||
        inputDesc->GetDataType() != ge::DT_FLOAT16 ||
        outputDesc->GetDataType() != ge::DT_UINT8) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t batch = static_cast<uint32_t>(input->GetStorageShape().GetDim(0));
    const uint32_t channels = static_cast<uint32_t>(input->GetStorageShape().GetDim(1));
    const uint32_t inputHeight = static_cast<uint32_t>(input->GetStorageShape().GetDim(2));
    const uint32_t inputWidth = static_cast<uint32_t>(input->GetStorageShape().GetDim(3));
    const uint32_t width = Attr(context, kWidthAttr, 0);
    const uint32_t height = Attr(context, kHeightAttr, 0);
    const uint32_t widthStride = Attr(context, kWidthStrideAttr, 0);
    const uint32_t heightStride = Attr(context, kHeightStrideAttr, 0);
    const uint32_t blockDim = Attr(context, kBlockDimAttr, kDefaultBlockDim);
    if (batch != 1U || channels != 3U || width == 0U || height == 0U ||
        (width & 1U) != 0U || (height & 1U) != 0U || width > inputWidth ||
        height > inputHeight || widthStride < width || heightStride < height ||
        (width & 15U) != 0U || (widthStride & 31U) != 0U ||
        (heightStride & 1U) != 0U || blockDim == 0U ||
        output->GetStorageShape().GetDim(0) != static_cast<int64_t>(heightStride * 3U / 2U) ||
        output->GetStorageShape().GetDim(1) != static_cast<int64_t>(widthStride)) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t outputBytes64 =
        static_cast<uint64_t>(widthStride) * heightStride * 3U / 2U;
    if (outputBytes64 > static_cast<uint64_t>(UINT32_MAX)) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t outputRows = heightStride * 3U / 2U;
    const uint32_t activeBlockDim = blockDim < outputRows ? blockDim : outputRows;
    const uint32_t rowsPerCore = CeilDiv(outputRows, activeBlockDim);
    tiling->inputHeight = inputHeight;
    tiling->inputWidth = inputWidth;
    tiling->visibleWidth = width;
    tiling->visibleHeight = height;
    tiling->widthStride = widthStride;
    tiling->heightStride = heightStride;
    tiling->outputRows = outputRows;
    tiling->rowsPerCore = rowsPerCore;
    tiling->tailCoreRows = outputRows - rowsPerCore * (activeBlockDim - 1U);
    context->SetBlockDim(activeBlockDim);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShapeMlvcFp16Yuv444ToNv12(gert::InferShapeContext* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    gert::Shape* output = context->GetOutputShape(0);
    if (attrs == nullptr || output == nullptr || attrs->GetAttrNum() <= kHeightStrideAttr) {
        return GRAPH_FAILED;
    }
    const int64_t* widthStride = attrs->GetAttrPointer<int64_t>(kWidthStrideAttr);
    const int64_t* heightStride = attrs->GetAttrPointer<int64_t>(kHeightStrideAttr);
    if (widthStride == nullptr || heightStride == nullptr || *widthStride <= 0 ||
        *heightStride <= 0 || (*heightStride & 1) != 0) {
        return GRAPH_FAILED;
    }
    output->SetDimNum(2);
    output->SetDim(0, *heightStride * 3 / 2);
    output->SetDim(1, *widthStride);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMlvcFp16Yuv444ToNv12(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_UINT8);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class MlvcFp16Yuv444ToNv12 : public OpDef {
public:
    explicit MlvcFp16Yuv444ToNv12(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("width").AttrType(REQUIRED).Int();
        this->Attr("height").AttrType(REQUIRED).Int();
        this->Attr("width_stride").AttrType(REQUIRED).Int();
        this->Attr("height_stride").AttrType(REQUIRED).Int();
        this->Attr("block_dim").AttrType(OPTIONAL).Int(kDefaultBlockDim);
        this->SetInferShape(ge::InferShapeMlvcFp16Yuv444ToNv12)
            .SetInferDataType(ge::InferDataTypeMlvcFp16Yuv444ToNv12);
        this->AICore().SetTiling(optiling::MlvcFp16Yuv444ToNv12TilingFunc);
        this->AICore().AddConfig("ascend310p");
    }
};

OP_ADD(MlvcFp16Yuv444ToNv12);
}  // namespace ops
