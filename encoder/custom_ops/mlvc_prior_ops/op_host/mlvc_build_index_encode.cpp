#include "../op_kernel/mlvc_build_index_encode_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t kDefaultBlockDim = 8;
constexpr uint32_t kDefaultTileElements = 1024;
constexpr size_t kForceZeroThresAttrIndex = 0;
constexpr size_t kTileElementsAttrIndex = 1;
constexpr size_t kBlockDimAttrIndex = 2;

uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1) / divisor;
}

uint32_t SelectTileElements(gert::TilingContext* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() > kTileElementsAttrIndex) {
        const int64_t* tileAttr = attrs->GetAttrPointer<int64_t>(kTileElementsAttrIndex);
        if (tileAttr != nullptr && *tileAttr >= 64 && *tileAttr <= 8192) {
            return static_cast<uint32_t>(*tileAttr);
        }
    }
    return kDefaultTileElements;
}

uint32_t SelectBlockDim(gert::TilingContext* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() > kBlockDimAttrIndex) {
        const int64_t* blockDimAttr = attrs->GetAttrPointer<int64_t>(kBlockDimAttrIndex);
        if (blockDimAttr != nullptr &&
            (*blockDimAttr == 1 || *blockDimAttr == 2 || *blockDimAttr == 4 ||
             *blockDimAttr == 8 || *blockDimAttr == 16 || *blockDimAttr == 24 ||
             *blockDimAttr == 32)) {
            return static_cast<uint32_t>(*blockDimAttr);
        }
    }
    return kDefaultBlockDim;
}

float SelectForceZeroThres(gert::TilingContext* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() > kForceZeroThresAttrIndex) {
        const float* attr = attrs->GetAttrPointer<float>(kForceZeroThresAttrIndex);
        if (attr != nullptr) {
            return *attr;
        }
    }
    return -1.0F;
}

uint64_t ElementCount(const gert::StorageShape* shape)
{
    uint64_t elements = 1;
    for (int i = 0; i < shape->GetStorageShape().GetDimNum(); ++i) {
        elements *= static_cast<uint64_t>(shape->GetStorageShape().GetDim(i));
    }
    return elements;
}
}

namespace optiling {
static ge::graphStatus MlvcBuildIndexEncodeTilingFunc(gert::TilingContext* context)
{
    auto* tiling = context->GetTilingData<MlvcBuildIndexEncodeTilingData>();
    const gert::StorageShape* symbolsShape = context->GetInputShape(0);
    const gert::StorageShape* scalesShape = context->GetInputShape(1);
    const gert::StorageShape* lookupShape = context->GetInputShape(2);
    const gert::StorageShape* combinedShape = context->GetOutputShape(0);
    const gert::StorageShape* keepShape = context->GetOutputShape(1);
    if (tiling == nullptr || symbolsShape == nullptr || scalesShape == nullptr ||
        lookupShape == nullptr || combinedShape == nullptr || keepShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t elements = ElementCount(symbolsShape);
    if (elements == 0 || elements > static_cast<uint64_t>(UINT32_MAX) ||
        ElementCount(scalesShape) != elements || ElementCount(combinedShape) != elements ||
        ElementCount(keepShape) != elements || ElementCount(lookupShape) < 65536) {
        return ge::GRAPH_FAILED;
    }

    const auto* symbolDesc = context->GetInputDesc(0);
    const auto* scaleDesc = context->GetInputDesc(1);
    const auto* lookupDesc = context->GetInputDesc(2);
    const auto* combinedDesc = context->GetOutputDesc(0);
    const auto* keepDesc = context->GetOutputDesc(1);
    if (symbolDesc == nullptr || scaleDesc == nullptr || lookupDesc == nullptr ||
        combinedDesc == nullptr || keepDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto symbolType = symbolDesc->GetDataType();
    const auto scaleType = scaleDesc->GetDataType();
    if ((symbolType != ge::DT_FLOAT16 && symbolType != ge::DT_INT8) ||
        scaleType != ge::DT_FLOAT16 ||
        lookupDesc->GetDataType() != ge::DT_UINT8 ||
        combinedDesc->GetDataType() != ge::DT_INT16 ||
        keepDesc->GetDataType() != ge::DT_UINT8) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t blockDim = SelectBlockDim(context);
    const uint32_t total = static_cast<uint32_t>(elements);
    const uint32_t elementsPerCore = CeilDiv(total, blockDim);
    const uint32_t tailCoreElements = total > elementsPerCore * (blockDim - 1)
                                          ? total - elementsPerCore * (blockDim - 1)
                                          : 0;

    tiling->totalElements = total;
    tiling->elementsPerCore = elementsPerCore;
    tiling->tailCoreElements = tailCoreElements;
    tiling->tileElements = SelectTileElements(context);
    tiling->symbolType = symbolType == ge::DT_INT8 ? 1U : 0U;
    tiling->forceZeroThres = SelectForceZeroThres(context);

    context->SetBlockDim(blockDim);
    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShapeMlvcBuildIndexEncode(gert::InferShapeContext* context)
{
    const gert::Shape* symbolsShape = context->GetInputShape(0);
    const gert::Shape* scalesShape = context->GetInputShape(1);
    gert::Shape* combinedShape = context->GetOutputShape(0);
    gert::Shape* keepShape = context->GetOutputShape(1);
    if (symbolsShape == nullptr || scalesShape == nullptr || combinedShape == nullptr ||
        keepShape == nullptr || symbolsShape->GetDimNum() != scalesShape->GetDimNum()) {
        return GRAPH_FAILED;
    }
    for (int i = 0; i < symbolsShape->GetDimNum(); ++i) {
        if (symbolsShape->GetDim(i) != scalesShape->GetDim(i)) {
            return GRAPH_FAILED;
        }
    }
    *combinedShape = *symbolsShape;
    *keepShape = *symbolsShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMlvcBuildIndexEncode(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_INT16);
    context->SetOutputDataType(1, ge::DT_UINT8);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class MlvcBuildIndexEncode : public OpDef {
public:
    explicit MlvcBuildIndexEncode(const char* name) : OpDef(name)
    {
        this->Input("symbols")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("scales")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("fp16_scale_index_lookup")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("combined")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT16, ge::DT_INT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("keep_mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("force_zero_thres").AttrType(OPTIONAL).Float(-1.0F);
        this->Attr("tile_elements").AttrType(OPTIONAL).Int(kDefaultTileElements);
        this->Attr("block_dim").AttrType(OPTIONAL).Int(kDefaultBlockDim);

        this->SetInferShape(ge::InferShapeMlvcBuildIndexEncode)
            .SetInferDataType(ge::InferDataTypeMlvcBuildIndexEncode);

        this->AICore().SetTiling(optiling::MlvcBuildIndexEncodeTilingFunc);
        this->AICore().AddConfig("ascend310p");
    }
};

OP_ADD(MlvcBuildIndexEncode);
}
