
#include "../op_kernel/post_sigmoid_mul_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t kBlockDim = 8;
constexpr uint32_t kDefaultTileElements = 8192;
constexpr size_t kTileElementsAttrIndex = 0;
constexpr size_t kBlockDimAttrIndex = 1;

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

bool IsSupportedStorageShape(const gert::StorageShape* shape)
{
    if (shape == nullptr) {
        return false;
    }
    const int dimNum = shape->GetStorageShape().GetDimNum();
    return dimNum == 4 || dimNum == 5;
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

uint32_t SelectTileElements(gert::TilingContext* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() > kTileElementsAttrIndex) {
        const int64_t* tileAttr = attrs->GetAttrPointer<int64_t>(kTileElementsAttrIndex);
        if (tileAttr != nullptr &&
            (*tileAttr == 4096 || *tileAttr == 8192 || *tileAttr == 16384)) {
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
}

namespace optiling {
static ge::graphStatus PostSigmoidMulTilingFunc(gert::TilingContext* context)
{
    auto* tiling = context->GetTilingData<PostSigmoidMulTilingData>();
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* sigmoidShape = context->GetInputShape(1);
    const gert::StorageShape* yShape = context->GetOutputShape(0);
    if (tiling == nullptr || !IsSupportedStorageShape(xShape) || !IsSupportedStorageShape(sigmoidShape) ||
        !IsSupportedStorageShape(yShape) ||
        !SameShape(xShape, sigmoidShape) || !SameShape(xShape, yShape)) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t elements64 = ElementCount(xShape);
    if (elements64 == 0 || elements64 > static_cast<uint64_t>(UINT32_MAX)) {
        return ge::GRAPH_FAILED;
    }

    const auto* xDesc = context->GetInputDesc(0);
    const auto* sigmoidDesc = context->GetInputDesc(1);
    const auto* yDesc = context->GetOutputDesc(0);
    if (xDesc == nullptr || sigmoidDesc == nullptr || yDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto dataType = xDesc->GetDataType();
    if (dataType != sigmoidDesc->GetDataType() || dataType != yDesc->GetDataType() ||
        (dataType != ge::DT_FLOAT16 && dataType != ge::DT_FLOAT)) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t total = static_cast<uint32_t>(elements64);
    const uint32_t blockDim = SelectBlockDim(context);
    const uint32_t elementsPerCore = CeilDiv(total, blockDim);
    const uint32_t tailCoreElements = total > elementsPerCore * (blockDim - 1U)
                                          ? total - elementsPerCore * (blockDim - 1U)
                                          : 0U;

    tiling->totalElements = total;
    tiling->elementsPerCore = elementsPerCore;
    tiling->tailCoreElements = tailCoreElements;
    tiling->tileElements = SelectTileElements(context);
    tiling->dataType = dataType == ge::DT_FLOAT ? 1U : 0U;

    context->SetBlockDim(blockDim);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShapePostSigmoidMul(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::Shape* sigmoidShape = context->GetInputShape(1);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (xShape == nullptr || sigmoidShape == nullptr || yShape == nullptr ||
        xShape->GetDimNum() != 4 || sigmoidShape->GetDimNum() != 4 ||
        xShape->GetDimNum() != sigmoidShape->GetDimNum()) {
        return GRAPH_FAILED;
    }
    for (int i = 0; i < xShape->GetDimNum(); ++i) {
        if (xShape->GetDim(i) != sigmoidShape->GetDim(i)) {
            return GRAPH_FAILED;
        }
    }
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypePostSigmoidMul(gert::InferDataTypeContext* context)
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
class PostSigmoidMul : public OpDef {
public:
    explicit PostSigmoidMul(const char* name) : OpDef(name)
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
        this->Attr("tile_elements").AttrType(OPTIONAL).Int(kDefaultTileElements);
        this->Attr("block_dim").AttrType(OPTIONAL).Int(kBlockDim);

        this->SetInferShape(ge::InferShapePostSigmoidMul)
            .SetInferDataType(ge::InferDataTypePostSigmoidMul);

        this->AICore().SetTiling(optiling::PostSigmoidMulTilingFunc);
        this->AICore().AddConfig("ascend310p");
    }
};

OP_ADD(PostSigmoidMul);
}
