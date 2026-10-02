#include <mlvc/codec/translation_warp.h>
#include <mlvc/core/status.h>

#include <climits>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

namespace {

std::vector<uint8_t> PixelOracle(const mlvc::codec::TensorData& source, int kx, int ky) {
  const auto& dims = source.shape.dims();
  const int64_t height = dims[dims.size() - 2];
  const int64_t width = dims.back();
  const std::size_t element_size = mlvc::ElementSize(source.dtype);
  const std::size_t plane_elements = static_cast<std::size_t>(width * height);
  auto expected = source.bytes;
  for (std::size_t plane = 0; plane < source.Elements() / plane_elements; ++plane) {
    for (int64_t y = 0; y < height; ++y) {
      for (int64_t x = 0; x < width; ++x) {
        const int64_t sx = x - kx;
        const int64_t sy = y - ky;
        if (sx < 0 || sx >= width || sy < 0 || sy >= height) continue;
        const auto dst = plane * plane_elements + static_cast<std::size_t>(y * width + x);
        const auto src = plane * plane_elements + static_cast<std::size_t>(sy * width + sx);
        std::memcpy(expected.data() + dst * element_size,
                    source.bytes.data() + src * element_size, element_size);
      }
    }
  }
  return expected;
}

void CheckDirectCopiesAndAliases() {
  using mlvc::Check;
  using mlvc::codec::ShiftTensorPreserveBoundary;
  using mlvc::codec::ShiftTensorPreserveBoundaryInto;
  auto source = mlvc::codec::MakeTensor({2, 4, 5}, mlvc::DataType::kFloat16);
  const uint16_t patterns[] = {0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e01, 0x7c01, 0xfc01,
                             0xffff, 0x0001, 0x03ff, 0x0400, 0x3555, 0xb555};
  for (std::size_t index = 0; index < source.Elements(); ++index) {
    const auto bits = patterns[index % (sizeof(patterns) / sizeof(patterns[0]))];
    std::memcpy(source.bytes.data() + index * 2, &bits, sizeof(bits));
  }
  const auto original = source.bytes;
  mlvc::codec::TensorData scratch;
  ShiftTensorPreserveBoundaryInto(source, 0, 0, &scratch);
  auto* scratch_pointer = scratch.bytes.data();
  const auto scratch_capacity = scratch.bytes.capacity();
  for (int ky = -3; ky <= 3; ++ky) {
    for (int kx = -4; kx <= 4; ++kx) {
      const auto expected = PixelOracle(source, kx, ky);
      scratch.bytes.assign(scratch.bytes.size(), 0xa5);
      ShiftTensorPreserveBoundaryInto(source, kx, ky, &scratch);
      Check(scratch.bytes == expected && scratch.dtype == source.dtype &&
                scratch.shape.dims() == source.shape.dims(),
            "direct warp did not preserve every FP16 bit or boundary byte");
      Check(scratch.bytes.data() == scratch_pointer && scratch.bytes.capacity() == scratch_capacity,
            "same-size warp did not reuse scratch storage");
      Check(ShiftTensorPreserveBoundary(source, kx, ky).bytes == expected,
            "returning warp differs from independent pixel oracle");
      auto alias = source;
      ShiftTensorPreserveBoundaryInto(alias, kx, ky, &alias);
      Check(alias.bytes == expected, "exact source/output alias corrupted warp data");
      alias = source;
      const auto borrowed = alias.View();
      ShiftTensorPreserveBoundaryInto(borrowed, kx, ky, &alias);
      Check(alias.bytes == expected, "borrowed source/output alias corrupted warp data");
      Check(source.bytes == original, "direct warp modified nonalias source");
    }
  }
  const auto zero = ShiftTensorPreserveBoundary(source, 0, 0);
  Check(zero.bytes == original && zero.bytes.data() != source.bytes.data(),
        "zero shift must remain an independent full copy");

  // Source is a subview of output's owned bytes, and the result changes shape
  // and shrinks output. Validate the snapshot before metadata/storage changes.
  auto partial = source;
  auto subsource = mlvc::codec::MakeTensor({2, 2, 3}, mlvc::DataType::kFloat16);
  std::memcpy(subsource.bytes.data(), partial.bytes.data() + 8, subsource.bytes.size());
  const auto subview = mlvc::TensorView::Borrowed(partial.bytes.data() + 8, &subsource.shape,
                                                subsource.dtype, mlvc::MemoryLocation::kCpu);
  const auto partial_expected = PixelOracle(subsource, -2, 1);
  ShiftTensorPreserveBoundaryInto(subview, -2, 1, &partial);
  Check(partial.bytes == partial_expected && partial.shape.dims() == subsource.shape.dims(),
        "partially overlapping source/output with resize corrupted warp data");

  // The borrowed source's sole lifetime lease lives in output. It must remain
  // alive through detachment and destination allocation/copy, then release.
  auto owner = std::make_shared<std::vector<uint8_t>>(source.bytes);
  const std::weak_ptr<std::vector<uint8_t>> lifetime = owner;
  mlvc::codec::TensorData external_output;
  external_output.shape = source.shape;
  external_output.dtype = source.dtype;
  external_output.external_data = owner->data();
  external_output.external_bytes = owner->size();
  external_output.external_owner = owner;
  external_output.external_location = mlvc::MemoryLocation::kCpu;
  owner.reset();
  const auto external_view = external_output.View();
  ShiftTensorPreserveBoundaryInto(external_view, 1, -1, &external_output);
  Check(external_output.bytes == PixelOracle(source, 1, -1) &&
            !external_output.has_external_buffer() && lifetime.expired(),
        "external output detachment lost borrowed source lifetime or owned-result semantics");

  // One-pixel spatial dimensions still permit movement along the other axis.
  for (const auto& shape : {std::vector<int64_t>{3, 1, 5}, std::vector<int64_t>{3, 5, 1}}) {
    auto thin = mlvc::codec::MakeTensor(shape, mlvc::DataType::kUInt8);
    for (std::size_t i = 0; i < thin.bytes.size(); ++i) thin.bytes[i] = static_cast<uint8_t>(i);
    const int kx = shape.back() == 1 ? 0 : -4;
    const int ky = shape.back() == 1 ? 4 : 0;
    ShiftTensorPreserveBoundaryInto(thin, kx, ky, &scratch);
    Check(scratch.bytes == PixelOracle(thin, kx, ky), "thin spatial-grid boundary mismatch");
  }
}

}  // namespace

int main() {
  using mlvc::Check;
  using mlvc::codec::MakeTensor;
  using mlvc::codec::ShiftTensorPreserveBoundary;
  CheckDirectCopiesAndAliases();
  auto source = MakeTensor({2, 3, 4, 5}, mlvc::DataType::kInt32);
  for (int i = 0; i < 120; ++i) std::memcpy(source.bytes.data() + i * 4, &i, 4);
  const auto original = source.bytes;
  for (int ky = -3; ky <= 3; ++ky) {
    for (int kx = -4; kx <= 4; ++kx) {
      const auto shifted = ShiftTensorPreserveBoundary(source, kx, ky);
      for (int plane = 0; plane < 6; ++plane) {
        for (int y = 0; y < 4; ++y) {
          for (int x = 0; x < 5; ++x) {
            const int sx = x - kx, sy = y - ky;
            const int expected = plane * 20 +
                (sx >= 0 && sx < 5 && sy >= 0 && sy < 4 ? sy * 5 + sx : y * 5 + x);
            int actual = -1;
            std::memcpy(&actual, shifted.bytes.data() + (plane * 20 + y * 5 + x) * 4, 4);
            Check(actual == expected, "integer slice or retained boundary mismatch");
          }
        }
      }
      Check(source.bytes == original, "warp modified source storage");
    }
  }
  auto memory = MakeTensor({2, 1, 4, 5}, mlvc::DataType::kInt32);
  memory.bytes.assign(memory.bytes.size(), 0x5a);
  const auto pair = mlvc::codec::ShiftFeatureAndMemory(source, memory, 2, -1);
  Check(pair.first.bytes == ShiftTensorPreserveBoundary(source, 2, -1).bytes &&
            pair.second.bytes == memory.bytes,
        "feature and memory did not share the same shift");
  for (int kx : {5, -5, INT_MIN, INT_MAX}) {
    bool rejected = false;
    try { (void)ShiftTensorPreserveBoundary(source, kx, 0); }
    catch (const mlvc::Error&) { rejected = true; }
    Check(rejected, "out-of-grid translation was accepted");
  }
  source.bytes.pop_back();
  bool rejected = false;
  try { (void)ShiftTensorPreserveBoundary(source, 0, 0); }
  catch (const mlvc::Error&) { rejected = true; }
  Check(rejected, "truncated tensor storage was accepted");
  rejected = false;
  try { mlvc::codec::ShiftTensorPreserveBoundaryInto(memory, 0, 0, nullptr); }
  catch (const mlvc::Error&) { rejected = true; }
  Check(rejected, "null warp output was accepted");
  const auto temp = std::filesystem::temp_directory_path() / "mlvc_warp_policy_check.json";
  {
    std::ofstream file(temp);
    Check(file.good(), "cannot write warp policy test manifest");
    file << R"({"runtime":"acl","soc_version":"Ascend310P1","dtype":"fp16",
      "sidecar":{"file":"sidecar","bytes":1,"sha256":"00"},"models":[
      {"name":"MLVCEncoder","file":"encoder.om","backend":"acl","bytes":1,
       "sha256":"e43b78e55ff2d4cda220623e63aeab4c40e823025203c0ed4baca9146e5daa4d",
       "inputs":[{"name":"ref_feature","dtype":"float16","shape":[1,96,136,240]}],"outputs":[]},
      {"name":"MLVCDecoder","file":"decoder.om","backend":"acl","bytes":1,
       "sha256":"040ba537a07864e53f1b54e288dae293fee91e1265151ae6d6c5c1141627045b",
       "inputs":[{"name":"ref_feature","dtype":"float16","shape":[1,96,136,240]}],"outputs":[]},
      {"name":"MLVCReferenceFromFrame","file":"reset.om","backend":"acl","bytes":1,
       "sha256":"91f68c4cf52286ff20f2ca1b80d453fa70d275925d9c80680b56c90874ab962f",
       "inputs":[{"name":"ref_frame","dtype":"float16","shape":[1,3,1088,1920]}],
       "outputs":[{"name":"ref_feature","dtype":"float16","shape":[1,96,136,240]}]}]})";
  }
  auto manifest = mlvc::ModelManifest::Load(temp);
  std::filesystem::remove(temp);
  mlvc::codec::RequireTranslationWarpModels(manifest);
  for (int model = 0; model < 3; ++model) {
    const auto sha = manifest.models()[model].sha256;
    manifest.models()[model].sha256 = std::string(64, '0');
    rejected = false;
    try { mlvc::codec::RequireTranslationWarpModels(manifest); }
    catch (const mlvc::Error&) { rejected = true; }
    Check(rejected, "warp accepted a model outside the audited fingerprint set");
    manifest.models()[model].sha256 = sha;
  }
  manifest.models()[0].inputs[0].shape[1] = 48;
  rejected = false;
  try { mlvc::codec::RequireTranslationWarpModels(manifest); }
  catch (const mlvc::Error&) { rejected = true; }
  Check(rejected, "warp accepted reference shape inconsistent with the audited graph");
  std::cout << "translation warp direct copies, reuse, aliases, FP16 bytes and bounds passed\n";
}
