#include "../op_kernel/mlvc_decode_prior_tiling.h"
#include "register/op_def_registry.h"

namespace {

constexpr uint32_t kDefaultBlockDim = 1;
constexpr uint32_t kDefaultTileElements = 1024;
constexpr size_t kMaskIndexAttrIndex = 0;
constexpr size_t kPartsAttrIndex = 1;
constexpr size_t kTileElementsAttrIndex = 2;
constexpr size_t kBlockDimAttrIndex = 3;
constexpr size_t kSimpleTileElementsAttrIndex = 0;
constexpr size_t kSimpleBlockDimAttrIndex = 1;

uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1) / divisor;
}

uint64_t ElementCount(const gert::StorageShape* shape)
{
    uint64_t elements = 1;
    for (int i = 0; i < shape->GetStorageShape().GetDimNum(); ++i) {
        elements *= static_cast<uint64_t>(shape->GetStorageShape().GetDim(i));
    }
    return elements;
}

uint32_t Dim(const gert::StorageShape* shape, int index)
{
    return static_cast<uint32_t>(shape->GetStorageShape().GetDim(index));
}

uint32_t SelectIntAttr(gert::TilingContext* context, size_t attrIndex, uint32_t fallback)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() > attrIndex) {
        const int64_t* attr = attrs->GetAttrPointer<int64_t>(attrIndex);
        if (attr != nullptr && *attr > 0 && *attr <= static_cast<int64_t>(UINT32_MAX)) {
            return static_cast<uint32_t>(*attr);
        }
    }
    return fallback;
}

uint32_t SelectTileElements(gert::TilingContext* context, size_t attrIndex)
{
    const uint32_t value = SelectIntAttr(context, attrIndex, kDefaultTileElements);
    return value >= 64 && value <= 8192 ? value : kDefaultTileElements;
}

uint32_t SelectBlockDim(gert::TilingContext* context, size_t attrIndex)
{
    const uint32_t value = SelectIntAttr(context, attrIndex, kDefaultBlockDim);
    return value == 1U || value == 2U || value == 4U || value == 8U || value == 16U ||
                   value == 24U || value == 32U
               ? value
               : kDefaultBlockDim;
}

uint32_t ElementsPerCore(uint32_t total, uint32_t blockDim)
{
    return CeilDiv(total, blockDim);
}

uint32_t TailCoreElements(uint32_t total, uint32_t elementsPerCore, uint32_t blockDim)
{
    return total > elementsPerCore * (blockDim - 1U) ? total - elementsPerCore * (blockDim - 1U) : 0U;
}

bool IsNchw4D(const gert::StorageShape* shape)
{
    return shape != nullptr && shape->GetStorageShape().GetDimNum() == 4;
}

bool SameShape(const gert::StorageShape* lhs, const gert::StorageShape* rhs)
{
    if (lhs == nullptr || rhs == nullptr ||
        lhs->GetStorageShape().GetDimNum() != rhs->GetStorageShape().GetDimNum()) {
        return false;
    }
    for (int i = 0; i < lhs->GetStorageShape().GetDimNum(); ++i) {
        if (lhs->GetStorageShape().GetDim(i) != rhs->GetStorageShape().GetDim(i)) {
            return false;
        }
    }
    return true;
}

bool SameGeShape(const gert::Shape* lhs, const gert::Shape* rhs)
{
    if (lhs == nullptr || rhs == nullptr || lhs->GetDimNum() != rhs->GetDimNum()) {
        return false;
    }
    for (int i = 0; i < lhs->GetDimNum(); ++i) {
        if (lhs->GetDim(i) != rhs->GetDim(i)) {
            return false;
        }
    }
    return true;
}

} // namespace

namespace optiling {

static ge::graphStatus MlvcSinglePartTilingFunc(gert::TilingContext* context)
{
    auto* tiling = context->GetTilingData<MlvcSinglePartTilingData>();
    const gert::StorageShape* fullShape = context->GetInputShape(0);
    const gert::StorageShape* partShape = context->GetOutputShape(0);
    if (tiling == nullptr || !IsNchw4D(fullShape) || !IsNchw4D(partShape)) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t parts = SelectIntAttr(context, kPartsAttrIndex, 4);
    const uint32_t maskIndex = SelectIntAttr(context, kMaskIndexAttrIndex, 0);
    const uint32_t channels = Dim(fullShape, 1);
    const uint32_t height = Dim(fullShape, 2);
    const uint32_t width = Dim(fullShape, 3);
    if ((parts != 2U && parts != 4U) || maskIndex >= parts || channels == 0U ||
        channels % parts != 0U || height == 0U || width == 0U ||
        Dim(partShape, 0) != Dim(fullShape, 0) || Dim(fullShape, 0) != 1U ||
        Dim(partShape, 1) != channels / parts || Dim(partShape, 2) != height ||
        Dim(partShape, 3) != width) {
        return ge::GRAPH_FAILED;
    }

    const auto* fullDesc = context->GetInputDesc(0);
    const auto* partDesc = context->GetOutputDesc(0);
    if (fullDesc == nullptr || partDesc == nullptr ||
        fullDesc->GetDataType() != partDesc->GetDataType() ||
        (fullDesc->GetDataType() != ge::DT_FLOAT16 && fullDesc->GetDataType() != ge::DT_FLOAT)) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t elements64 = ElementCount(partShape);
    if (elements64 == 0 || elements64 > static_cast<uint64_t>(UINT32_MAX)) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t total = static_cast<uint32_t>(elements64);
    const uint32_t blockDim = SelectBlockDim(context, kBlockDimAttrIndex);
    const uint32_t elementsPerCore = ElementsPerCore(total, blockDim);

    tiling->totalElements = total;
    tiling->elementsPerCore = elementsPerCore;
    tiling->tailCoreElements = TailCoreElements(total, elementsPerCore, blockDim);
    tiling->tileElements = SelectTileElements(context, kTileElementsAttrIndex);
    tiling->channels = channels;
    tiling->height = height;
    tiling->width = width;
    tiling->partChannels = channels / parts;
    tiling->maskIndex = maskIndex;
    tiling->parts = parts;
    tiling->dataType = fullDesc->GetDataType() == ge::DT_FLOAT ? 1U : 0U;

    context->SetBlockDim(blockDim);
    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MlvcRestoreYTilingFunc(gert::TilingContext* context)
{
    auto* tiling = context->GetTilingData<MlvcRestoreYTilingData>();
    const gert::StorageShape* symbolsShape = context->GetInputShape(0);
    const gert::StorageShape* meansShape = context->GetInputShape(1);
    const gert::StorageShape* outputShape = context->GetOutputShape(0);
    if (tiling == nullptr || !IsNchw4D(symbolsShape) || !IsNchw4D(meansShape) ||
        !IsNchw4D(outputShape) || !SameShape(meansShape, outputShape)) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t parts = SelectIntAttr(context, kPartsAttrIndex, 4);
    const uint32_t maskIndex = SelectIntAttr(context, kMaskIndexAttrIndex, 0);
    const uint32_t channels = Dim(meansShape, 1);
    const uint32_t height = Dim(meansShape, 2);
    const uint32_t width = Dim(meansShape, 3);
    if ((parts != 2U && parts != 4U) || maskIndex >= parts || channels == 0U ||
        channels % parts != 0U || height == 0U || width == 0U ||
        Dim(meansShape, 0) != 1U || Dim(symbolsShape, 0) != 1U ||
        Dim(symbolsShape, 1) != channels / parts || Dim(symbolsShape, 2) != height ||
        Dim(symbolsShape, 3) != width) {
        return ge::GRAPH_FAILED;
    }

    const auto* symbolsDesc = context->GetInputDesc(0);
    const auto* meansDesc = context->GetInputDesc(1);
    const auto* outputDesc = context->GetOutputDesc(0);
    if (symbolsDesc == nullptr || meansDesc == nullptr || outputDesc == nullptr ||
        symbolsDesc->GetDataType() != ge::DT_INT8 ||
        meansDesc->GetDataType() != ge::DT_FLOAT16 ||
        outputDesc->GetDataType() != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t partElements64 = ElementCount(symbolsShape);
    const uint64_t fullElements64 = ElementCount(meansShape);
    if (partElements64 == 0 || partElements64 > static_cast<uint64_t>(UINT32_MAX) ||
        fullElements64 == 0 || fullElements64 > static_cast<uint64_t>(UINT32_MAX)) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t total = static_cast<uint32_t>(partElements64);
    const uint32_t blockDim = SelectBlockDim(context, kBlockDimAttrIndex);
    const uint32_t elementsPerCore = ElementsPerCore(total, blockDim);

    tiling->totalPartElements = total;
    tiling->totalFullElements = static_cast<uint32_t>(fullElements64);
    tiling->elementsPerCore = elementsPerCore;
    tiling->tailCoreElements = TailCoreElements(total, elementsPerCore, blockDim);
    tiling->tileElements = SelectTileElements(context, kTileElementsAttrIndex);
    tiling->channels = channels;
    tiling->height = height;
    tiling->width = width;
    tiling->partChannels = channels / parts;
    tiling->maskIndex = maskIndex;
    tiling->parts = parts;

    context->SetBlockDim(blockDim);
    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MlvcApplyChannelQuantStepTilingFunc(gert::TilingContext* context)
{
    auto* tiling = context->GetTilingData<MlvcApplyChannelQuantStepTilingData>();
    const gert::StorageShape* quantShape = context->GetInputShape(0);
    const gert::StorageShape* valuesShape = context->GetInputShape(1);
    const gert::StorageShape* outputShape = context->GetOutputShape(0);
    if (tiling == nullptr || !IsNchw4D(quantShape) || !IsNchw4D(valuesShape) ||
        !IsNchw4D(outputShape) || !SameShape(valuesShape, outputShape)) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t channels = Dim(valuesShape, 1);
    const uint32_t height = Dim(valuesShape, 2);
    const uint32_t width = Dim(valuesShape, 3);
    const uint32_t quantChannels = Dim(quantShape, 1);
    if (Dim(valuesShape, 0) != 1U || Dim(quantShape, 0) != 1U ||
        channels == 0U || height == 0U || width == 0U ||
        (quantChannels != 1U && quantChannels != channels) ||
        Dim(quantShape, 2) != height || Dim(quantShape, 3) != width) {
        return ge::GRAPH_FAILED;
    }

    const auto* quantDesc = context->GetInputDesc(0);
    const auto* valuesDesc = context->GetInputDesc(1);
    const auto* outputDesc = context->GetOutputDesc(0);
    if (quantDesc == nullptr || valuesDesc == nullptr || outputDesc == nullptr ||
        quantDesc->GetDataType() != ge::DT_FLOAT16 ||
        valuesDesc->GetDataType() != ge::DT_FLOAT16 ||
        outputDesc->GetDataType() != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t elements64 = ElementCount(valuesShape);
    if (elements64 == 0 || elements64 > static_cast<uint64_t>(UINT32_MAX)) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t total = static_cast<uint32_t>(elements64);
    const uint32_t blockDim = SelectBlockDim(context, kSimpleBlockDimAttrIndex);
    const uint32_t elementsPerCore = ElementsPerCore(total, blockDim);

    tiling->totalElements = total;
    tiling->elementsPerCore = elementsPerCore;
    tiling->tailCoreElements = TailCoreElements(total, elementsPerCore, blockDim);
    tiling->tileElements = SelectTileElements(context, kSimpleTileElementsAttrIndex);
    tiling->channels = channels;
    tiling->height = height;
    tiling->width = width;
    tiling->quantChannels = quantChannels;

    context->SetBlockDim(blockDim);
    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MlvcInt8ToFp16TilingFunc(gert::TilingContext* context)
{
    auto* tiling = context->GetTilingData<MlvcInt8ToFp16TilingData>();
    const gert::StorageShape* symbolsShape = context->GetInputShape(0);
    const gert::StorageShape* outputShape = context->GetOutputShape(0);
    if (tiling == nullptr || symbolsShape == nullptr || outputShape == nullptr ||
        !SameShape(symbolsShape, outputShape)) {
        return ge::GRAPH_FAILED;
    }

    const auto* symbolsDesc = context->GetInputDesc(0);
    const auto* outputDesc = context->GetOutputDesc(0);
    if (symbolsDesc == nullptr || outputDesc == nullptr ||
        symbolsDesc->GetDataType() != ge::DT_INT8 ||
        outputDesc->GetDataType() != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t elements64 = ElementCount(symbolsShape);
    if (elements64 == 0 || elements64 > static_cast<uint64_t>(UINT32_MAX)) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t total = static_cast<uint32_t>(elements64);
    const uint32_t blockDim = SelectBlockDim(context, kSimpleBlockDimAttrIndex);
    const uint32_t elementsPerCore = ElementsPerCore(total, blockDim);

    tiling->totalElements = total;
    tiling->elementsPerCore = elementsPerCore;
    tiling->tailCoreElements = TailCoreElements(total, elementsPerCore, blockDim);
    tiling->tileElements = SelectTileElements(context, kSimpleTileElementsAttrIndex);

    context->SetBlockDim(blockDim);
    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static ge::graphStatus InferShapeMlvcSinglePart(gert::InferShapeContext* context)
{
    const gert::Shape* fullShape = context->GetInputShape(0);
    gert::Shape* partShape = context->GetOutputShape(0);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const int64_t* partsAttr = attrs != nullptr && attrs->GetAttrNum() > kPartsAttrIndex
                                   ? attrs->GetAttrPointer<int64_t>(kPartsAttrIndex)
                                   : nullptr;
    const int64_t parts = partsAttr != nullptr ? *partsAttr : 4;
    if (fullShape == nullptr || partShape == nullptr || fullShape->GetDimNum() != 4 ||
        (parts != 2 && parts != 4)) {
        return GRAPH_FAILED;
    }
    const int64_t channels = fullShape->GetDim(1);
    if (channels <= 0 || channels % parts != 0) {
        return GRAPH_FAILED;
    }
    *partShape = *fullShape;
    partShape->SetDim(1, channels / parts);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeMlvcRestoreY(gert::InferShapeContext* context)
{
    const gert::Shape* meansShape = context->GetInputShape(1);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (meansShape == nullptr || outputShape == nullptr || meansShape->GetDimNum() != 4) {
        return GRAPH_FAILED;
    }
    *outputShape = *meansShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeMlvcSameAsSecondInput(gert::InferShapeContext* context)
{
    const gert::Shape* valuesShape = context->GetInputShape(1);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (valuesShape == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }
    *outputShape = *valuesShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeMlvcSameAsFirstInput(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMlvcSinglePart(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMlvcFp16(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class MlvcSinglePart : public OpDef {
public:
    explicit MlvcSinglePart(const char* name) : OpDef(name)
    {
        this->Input("full")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("part")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("mask_index").AttrType(OPTIONAL).Int(0);
        this->Attr("parts").AttrType(OPTIONAL).Int(4);
        this->Attr("tile_elements").AttrType(OPTIONAL).Int(kDefaultTileElements);
        this->Attr("block_dim").AttrType(OPTIONAL).Int(kDefaultBlockDim);

        this->SetInferShape(ge::InferShapeMlvcSinglePart)
            .SetInferDataType(ge::InferDataTypeMlvcSinglePart);
        this->AICore().SetTiling(optiling::MlvcSinglePartTilingFunc);
        this->AICore().AddConfig("ascend310p");
    }
};

class MlvcRestoreY : public OpDef {
public:
    explicit MlvcRestoreY(const char* name) : OpDef(name)
    {
        this->Input("y_symbols")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("means")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("mask_index").AttrType(OPTIONAL).Int(0);
        this->Attr("parts").AttrType(OPTIONAL).Int(4);
        this->Attr("tile_elements").AttrType(OPTIONAL).Int(kDefaultTileElements);
        this->Attr("block_dim").AttrType(OPTIONAL).Int(kDefaultBlockDim);

        this->SetInferShape(ge::InferShapeMlvcRestoreY)
            .SetInferDataType(ge::InferDataTypeMlvcFp16);
        this->AICore().SetTiling(optiling::MlvcRestoreYTilingFunc);
        this->AICore().AddConfig("ascend310p");
    }
};

class MlvcApplyChannelQuantStep : public OpDef {
public:
    explicit MlvcApplyChannelQuantStep(const char* name) : OpDef(name)
    {
        this->Input("quant_step")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("values_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("tile_elements").AttrType(OPTIONAL).Int(kDefaultTileElements);
        this->Attr("block_dim").AttrType(OPTIONAL).Int(kDefaultBlockDim);

        this->SetInferShape(ge::InferShapeMlvcSameAsSecondInput)
            .SetInferDataType(ge::InferDataTypeMlvcFp16);
        this->AICore().SetTiling(optiling::MlvcApplyChannelQuantStepTilingFunc);
        this->AICore().AddConfig("ascend310p");
    }
};

class MlvcInt8ToFp16 : public OpDef {
public:
    explicit MlvcInt8ToFp16(const char* name) : OpDef(name)
    {
        this->Input("symbols")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("tile_elements").AttrType(OPTIONAL).Int(kDefaultTileElements);
        this->Attr("block_dim").AttrType(OPTIONAL).Int(kDefaultBlockDim);

        this->SetInferShape(ge::InferShapeMlvcSameAsFirstInput)
            .SetInferDataType(ge::InferDataTypeMlvcFp16);
        this->AICore().SetTiling(optiling::MlvcInt8ToFp16TilingFunc);
        this->AICore().AddConfig("ascend310p");
    }
};

OP_ADD(MlvcSinglePart);
OP_ADD(MlvcRestoreY);
OP_ADD(MlvcApplyChannelQuantStep);
OP_ADD(MlvcInt8ToFp16);

} // namespace ops
