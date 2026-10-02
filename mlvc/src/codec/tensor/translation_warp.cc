#include <mlvc/codec/translation_warp.h>

#include <mlvc/core/status.h>

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <limits>

namespace mlvc::codec {

namespace {

bool Overlaps(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) {
  if (a_bytes == 0 || b_bytes == 0) return false;
  const auto a_address = reinterpret_cast<uintptr_t>(a);
  const auto b_address = reinterpret_cast<uintptr_t>(b);
  return a_address <= b_address ? b_address - a_address < a_bytes
                                : a_address - b_address < b_bytes;
}

}  // namespace

void ShiftTensorPreserveBoundaryInto(const mlvc::TensorView& source, int kx, int ky,
                                    TensorData* output) {
  Check(output != nullptr, "translation warp output is required");
  Check(source.location() == mlvc::MemoryLocation::kCpu,
        "translation warp requires an explicitly materialized CPU tensor");
  // A borrowed source view may refer to output->shape itself.
  const auto dims = source.shape().dims();
  Check(dims.size() >= 2, "translation warp requires at least two spatial dimensions");
  const auto dtype = source.dtype();
  const std::size_t element_size = mlvc::ElementSize(dtype);
  std::size_t byte_count = element_size;
  for (const int64_t dim : dims) {
    Check(dim > 0, "translation warp requires positive dimensions");
    Check(static_cast<uint64_t>(dim) <= std::numeric_limits<std::size_t>::max() / byte_count,
          "translation warp tensor byte count overflows");
    byte_count *= static_cast<std::size_t>(dim);
  }
  const int64_t height = dims[dims.size() - 2];
  const int64_t width = dims.back();
  Check(static_cast<int64_t>(kx) > -width && static_cast<int64_t>(kx) < width &&
            static_cast<int64_t>(ky) > -height && static_cast<int64_t>(ky) < height,
        "feature shift exceeds feature shape");
  Check(source.data() != nullptr, "translation warp requires nonempty tensor storage");
  const auto* input = static_cast<const uint8_t*>(source.data());
  std::vector<uint8_t> snapshot;
  if (Overlaps(input, byte_count, output->bytes.data(), output->bytes.capacity())) {
    snapshot.assign(input, input + byte_count);
    input = snapshot.data();
  }
  // Detaching an external output lease must not release a borrowed source.
  const auto source_lifetime = output->external_owner;
  output->ClearExternalBuffer();
  if (output->shape.dims() != dims) output->shape = mlvc::TensorShape(dims);
  output->dtype = dtype;
  output->bytes.resize(byte_count);
  auto* destination = output->bytes.data();
  if (kx == 0 && ky == 0) {
    std::memcpy(destination, input, byte_count);
    return;
  }
  const std::size_t row_bytes = static_cast<std::size_t>(width) * element_size;
  const std::size_t plane_bytes = static_cast<std::size_t>(height) * row_bytes;
  const int64_t src_x = std::max<int64_t>(-static_cast<int64_t>(kx), 0);
  const int64_t dst_x = std::max<int64_t>(kx, 0);
  const int64_t src_y = std::max<int64_t>(-static_cast<int64_t>(ky), 0);
  const int64_t dst_y = std::max<int64_t>(ky, 0);
  const int64_t copy_width = width - std::max<int64_t>(kx, -static_cast<int64_t>(kx));
  const int64_t copy_height = height - std::max<int64_t>(ky, -static_cast<int64_t>(ky));
  const std::size_t left_bytes = static_cast<std::size_t>(dst_x) * element_size;
  const std::size_t shifted_bytes = static_cast<std::size_t>(copy_width) * element_size;
  const std::size_t right_bytes = row_bytes - left_bytes - shifted_bytes;
  const std::size_t top_bytes = static_cast<std::size_t>(dst_y) * row_bytes;
  const std::size_t middle_bytes = static_cast<std::size_t>(copy_height) * row_bytes;
  const std::size_t bottom_offset = top_bytes + middle_bytes;
  for (std::size_t plane = 0; plane < byte_count / plane_bytes; ++plane) {
    const auto* plane_input = input + plane * plane_bytes;
    auto* plane_output = destination + plane * plane_bytes;
    if (top_bytes != 0) std::memcpy(plane_output, plane_input, top_bytes);
    if (bottom_offset < plane_bytes) {
      std::memcpy(plane_output + bottom_offset, plane_input + bottom_offset,
                  plane_bytes - bottom_offset);
    }
    // Pure vertical shifts form one contiguous rectangle per plane.
    if (kx == 0) {
      std::memcpy(plane_output + top_bytes,
                  plane_input + static_cast<std::size_t>(src_y) * row_bytes, middle_bytes);
      continue;
    }
    for (int64_t y = 0; y < copy_height; ++y) {
      const std::size_t destination_row = static_cast<std::size_t>(dst_y + y) * row_bytes;
      auto* row_output = plane_output + destination_row;
      const auto* original_row = plane_input + destination_row;
      const auto* shifted_row = plane_input + static_cast<std::size_t>(src_y + y) * row_bytes;
      if (left_bytes != 0) std::memcpy(row_output, original_row, left_bytes);
      std::memcpy(row_output + left_bytes,
                  shifted_row + static_cast<std::size_t>(src_x) * element_size, shifted_bytes);
      if (right_bytes != 0) {
        const std::size_t right_offset = left_bytes + shifted_bytes;
        std::memcpy(row_output + right_offset, original_row + right_offset, right_bytes);
      }
    }
  }
}

TensorData ShiftTensorPreserveBoundary(const mlvc::TensorView& source, int kx, int ky) {
  TensorData result;
  ShiftTensorPreserveBoundaryInto(source, kx, ky, &result);
  return result;
}

void ShiftTensorPreserveBoundaryInto(const TensorData& source, int kx, int ky, TensorData* output) {
  Check(source.ByteSize() == source.shape.NumElements() * mlvc::ElementSize(source.dtype),
        "translation warp tensor byte count does not match shape");
  ShiftTensorPreserveBoundaryInto(source.View(), kx, ky, output);
}

TensorData ShiftTensorPreserveBoundary(const TensorData& source, int kx, int ky) {
  TensorData result;
  ShiftTensorPreserveBoundaryInto(source, kx, ky, &result);
  return result;
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
