#include <mlvc/codec/translation_warp.h>

#include <mlvc/core/status.h>

#include <algorithm>
#include <cstring>
#include <limits>

namespace mlvc::codec {

TensorData ShiftTensorPreserveBoundary(const mlvc::TensorView& source, int kx, int ky) {
  Check(source.location() == mlvc::MemoryLocation::kCpu,
        "translation warp requires an explicitly materialized CPU tensor");
  const auto& dims = source.shape().dims();
  Check(dims.size() >= 2, "translation warp requires at least two spatial dimensions");
  for (const int64_t dim : dims) Check(dim > 0, "translation warp requires positive dimensions");
  const int64_t height = dims[dims.size() - 2];
  const int64_t width = dims.back();
  Check(static_cast<int64_t>(kx) > -width && static_cast<int64_t>(kx) < width &&
            static_cast<int64_t>(ky) > -height && static_cast<int64_t>(ky) < height,
        "feature shift exceeds feature shape");
  Check(source.data() != nullptr, "translation warp requires nonempty tensor storage");
  TensorData result = MakeTensor(dims, source.dtype());
  const auto* input = static_cast<const uint8_t*>(source.data());
  std::memcpy(result.bytes.data(), input, result.bytes.size());
  if (kx == 0 && ky == 0) return result;
  const std::size_t element_size = mlvc::ElementSize(source.dtype());
  const std::size_t row_bytes = static_cast<std::size_t>(width) * element_size;
  const std::size_t plane_bytes = static_cast<std::size_t>(height) * row_bytes;
  const int64_t src_x = std::max<int64_t>(-static_cast<int64_t>(kx), 0);
  const int64_t dst_x = std::max<int64_t>(kx, 0);
  const int64_t src_y = std::max<int64_t>(-static_cast<int64_t>(ky), 0);
  const int64_t dst_y = std::max<int64_t>(ky, 0);
  const int64_t copy_width = width - std::max<int64_t>(kx, -static_cast<int64_t>(kx));
  const int64_t copy_height = height - std::max<int64_t>(ky, -static_cast<int64_t>(ky));
  for (std::size_t plane = 0; plane < result.bytes.size() / plane_bytes; ++plane) {
    for (int64_t y = 0; y < copy_height; ++y) {
      const std::size_t src = plane * plane_bytes + static_cast<std::size_t>(src_y + y) * row_bytes +
                              static_cast<std::size_t>(src_x) * element_size;
      const std::size_t dst = plane * plane_bytes + static_cast<std::size_t>(dst_y + y) * row_bytes +
                              static_cast<std::size_t>(dst_x) * element_size;
      std::memcpy(result.bytes.data() + dst, input + src,
                  static_cast<std::size_t>(copy_width) * element_size);
    }
  }
  return result;
}

TensorData ShiftTensorPreserveBoundary(const TensorData& source, int kx, int ky) {
  Check(source.ByteSize() == source.shape.NumElements() * mlvc::ElementSize(source.dtype),
        "translation warp tensor byte count does not match shape");
  return ShiftTensorPreserveBoundary(source.View(), kx, ky);
}

std::pair<TensorData, TensorData> ShiftFeatureAndMemory(const TensorData& feature,
                                                      const TensorData& memory, int kx, int ky) {
  const auto& feature_dims = feature.shape.dims();
  const auto& memory_dims = memory.shape.dims();
  Check(feature_dims.size() >= 2 && memory_dims.size() >= 2 &&
            feature_dims.back() == memory_dims.back() &&
            feature_dims[feature_dims.size() - 2] == memory_dims[memory_dims.size() - 2],
        "feature and memory must share the spatial grid");
  return {ShiftTensorPreserveBoundary(feature, kx, ky),
          ShiftTensorPreserveBoundary(memory, kx, ky)};
}

void RequireTranslationWarpModels(const mlvc::ModelManifest& manifest) {
  const auto& encoder = manifest.GetModel("MLVCEncoder");
  const auto& decoder = manifest.GetModel("MLVCDecoder");
  Check(encoder.sha256 == "e43b78e55ff2d4cda220623e63aeab4c40e823025203c0ed4baca9146e5daa4d" &&
            decoder.sha256 == "040ba537a07864e53f1b54e288dae293fee91e1265151ae6d6c5c1141627045b",
        "translation warp requires the audited 1080p FP16 model pair; audit new models before enabling");
  for (const auto* model : {&encoder, &decoder}) {
    const auto reference = std::find_if(model->inputs.begin(), model->inputs.end(),
        [](const mlvc::TensorSpec& spec) { return spec.name == "ref_feature"; });
    Check(reference != model->inputs.end() && reference->dtype == mlvc::DataType::kFloat16 &&
              reference->shape == std::vector<int64_t>({1, 96, 136, 240}),
          "translation warp requires the audited concatenated feature/memory grid");
  }
  Check(manifest.HasModel("MLVCReferenceFromFrame"),
        "translation warp requires MLVCReferenceFromFrame for Python GOP/reset semantics");
  const auto& reset = manifest.GetModel("MLVCReferenceFromFrame");
  Check(reset.sha256 == "91f68c4cf52286ff20f2ca1b80d453fa70d275925d9c80680b56c90874ab962f",
        "translation warp requires the audited MLVCReferenceFromFrame OM");
  Check(reset.inputs.size() == 1 && reset.inputs[0].name == "ref_frame" &&
            reset.inputs[0].dtype == mlvc::DataType::kFloat16 &&
            reset.inputs[0].shape == std::vector<int64_t>({1, 3, 1088, 1920}) &&
            reset.outputs.size() == 1 && reset.outputs[0].name == "ref_feature" &&
            reset.outputs[0].dtype == mlvc::DataType::kFloat16 &&
            reset.outputs[0].shape == std::vector<int64_t>({1, 96, 136, 240}),
        "translation warp frame adaptor has an unexpected tensor contract");
}

}  // namespace mlvc::codec
