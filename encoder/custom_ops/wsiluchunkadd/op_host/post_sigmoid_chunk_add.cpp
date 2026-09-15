
#include "../op_kernel/post_sigmoid_chunk_add_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t kBlockDim = 8;
constexpr uint32_t kDefaultTileElements = 4096;
constexpr uint32_t kDefaultBarrierMode = 0;
constexpr size_t kTileElementsAttrIndex = 1;
constexpr size_t kBlockDimAttrIndex = 2;
constexpr size_t kBarrierModeAttrIndex = 3;
constexpr int64_t kAxisC = 1;

uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1) / divisor;
}

uint32_t SelectTileElements(gert::TilingContext* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() > kTileElementsAttrIndex) {
        const int64_t* tileAttr = attrs->GetAttrPointer<int64_t>(kTileElementsAttrIndex);
        if (tileAttr != nullptr &&
            (*tileAttr == 2048 || *tileAttr == 4096 || *tileAttr == 8192 || *tileAttr == 16384)) {
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
            (*blockDimAttr == 8 || *blockDimAttr == 16 || *blockDimAttr == 24 || *blockDimAttr == 32)) {
            return static_cast<uint32_t>(*blockDimAttr);
        }
    }
    return kBlockDim;
}

uint32_t SelectBarrierMode(gert::TilingContext* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() > kBarrierModeAttrIndex) {
        const int64_t* barrierModeAttr = attrs->GetAttrPointer<int64_t>(kBarrierModeAttrIndex);
        if (barrierModeAttr != nullptr && (*barrierModeAttr == 0 || *barrierModeAttr == 1)) {
            return static_cast<uint32_t>(*barrierModeAttr);
        }
    }
    return kDefaultBarrierMode;
}
}

namespace optiling {
static ge::graphStatus PostSigmoidChunkAddTilingFunc(gert::TilingContext* context)
{
    auto* tiling = context->GetTilingData<PostSigmoidChunkAddTilingData>();
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* sigmoidShape = context->GetInputShape(1);
    const gert::StorageShape* yShape = context->GetOutputShape(0);
    if (tiling == nullptr || xShape == nullptr || sigmoidShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint64_t inputElements = 1;
    for (int i = 0; i < xShape->GetStorageShape().GetDimNum(); ++i) {
        inputElements *= static_cast<uint64_t>(xShape->GetStorageShape().GetDim(i));
    }
    uint64_t sigmoidElements = 1;
    for (int i = 0; i < sigmoidShape->GetStorageShape().GetDimNum(); ++i) {
        sigmoidElements *= static_cast<uint64_t>(sigmoidShape->GetStorageShape().GetDim(i));
    }
    uint64_t outputElements = 1;
    for (int i = 0; i < yShape->GetStorageShape().GetDimNum(); ++i) {
        outputElements *= static_cast<uint64_t>(yShape->GetStorageShape().GetDim(i));
    }
    if (inputElements != sigmoidElements || inputElements != outputElements * 2 || outputElements == 0 ||
        outputElements > static_cast<uint64_t>(UINT32_MAX)) {
        return ge::GRAPH_FAILED;
    }

    const auto* inputDesc = context->GetInputDesc(0);
    const auto* sigmoidDesc = context->GetInputDesc(1);
    if (inputDesc == nullptr || sigmoidDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto inputDataType = inputDesc->GetDataType();
    if (inputDataType != sigmoidDesc->GetDataType() ||
        (inputDataType != ge::DT_FLOAT16 && inputDataType != ge::DT_FLOAT)) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t blockDim = SelectBlockDim(context);
    const uint32_t total = static_cast<uint32_t>(outputElements);
    const uint32_t elementsPerCore = CeilDiv(total, blockDim);
    const uint32_t tailCoreElements = total > elementsPerCore * (blockDim - 1)
                                          ? total - elementsPerCore * (blockDim - 1)
                                          : 0;

    tiling->totalOutputElements = total;
    tiling->elementsPerCore = elementsPerCore;
    tiling->tailCoreElements = tailCoreElements;
    tiling->tileElements = SelectTileElements(context);
    tiling->inputHalfOffset = total;
    tiling->dataType = inputDataType == ge::DT_FLOAT ? 1U : 0U;
    tiling->barrierMode = SelectBarrierMode(context);

    context->SetBlockDim(blockDim);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShapePostSigmoidChunkAdd(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::Shape* sigmoidShape = context->GetInputShape(1);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (xShape == nullptr || sigmoidShape == nullptr || yShape == nullptr || xShape->GetDimNum() != 4 ||
        sigmoidShape->GetDimNum() != 4) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDimNum() != sigmoidShape->GetDimNum()) {
        return GRAPH_FAILED;
    }
    for (int i = 0; i < xShape->GetDimNum(); ++i) {
        if (xShape->GetDim(i) != sigmoidShape->GetDim(i)) {
            return GRAPH_FAILED;
        }
    }
    *yShape = *xShape;
    const int64_t channels = xShape->GetDim(kAxisC);
    if (channels <= 0 || channels % 2 != 0) {
        return GRAPH_FAILED;
    }
    yShape->SetDim(kAxisC, channels / 2);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypePostSigmoidChunkAdd(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    const auto sigmoidDataType = context->GetInputDataType(1);
    if (inputDataType != sigmoidDataType) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class PostSigmoidChunkAdd : public OpDef {
public:
    explicit PostSigmoidChunkAdd(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW});
        this->Input("sigmoid_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW});
        this->Attr("axis").AttrType(OPTIONAL).Int(1);
        this->Attr("tile_elements").AttrType(OPTIONAL).Int(kDefaultTileElements);
        this->Attr("block_dim").AttrType(OPTIONAL).Int(kBlockDim);
        this->Attr("barrier_mode").AttrType(OPTIONAL).Int(kDefaultBarrierMode);

        this->SetInferShape(ge::InferShapePostSigmoidChunkAdd)
            .SetInferDataType(ge::InferDataTypePostSigmoidChunkAdd);

        this->AICore().SetTiling(optiling::PostSigmoidChunkAddTilingFunc);
        this->AICore().AddConfig("ascend310p");
    }
};

OP_ADD(PostSigmoidChunkAdd);
}
