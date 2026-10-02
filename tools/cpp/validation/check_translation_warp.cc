#include <mlvc/codec/translation_warp.h>
#include <mlvc/core/status.h>

#include <climits>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <fstream>

int main() {
  using mlvc::Check;
  using mlvc::codec::MakeTensor;
  using mlvc::codec::ShiftTensorPreserveBoundary;
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
  std::cout << "translation warp slices, retained boundaries, immutability and bounds passed\n";
}
