#include <mlvc/core/status.h>
#include <mlvc/runtime/model_manifest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr char kEmptySha256[] =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
constexpr std::array<const char*, 3> kArtifacts = {
    "metadata.json", "gaussian_pmf.json", "bit_estimator_pmf.json"};

void Expect(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void WriteText(const std::filesystem::path& path, const std::string& text) {
  std::ofstream output(path, std::ios::binary);
  output << text;
  Expect(output.good(), "failed to write test artifact: " + path.string());
}

std::filesystem::path WriteManifest(const std::filesystem::path& root) {
  WriteText(root / "sidecar.bin", "");
  WriteText(root / "stage.om", "x");
  std::string json =
      "{\n"
      "  \"runtime\": \"acl\",\n"
      "  \"soc_version\": \"Ascend310P1\",\n"
      "  \"dtype\": \"fp16\",\n"
      "  \"sidecar\": {\"file\": \"sidecar.bin\", \"bytes\": 0, \"sha256\": \"\"},\n"
      "  \"runtime_artifacts\": [\n";
  for (std::size_t i = 0; i < kArtifacts.size(); ++i) {
    WriteText(root / kArtifacts[i], "");
    json += "    {\"name\": \"" + std::string(kArtifacts[i]) + "\", \"file\": \"" +
            kArtifacts[i] + "\", \"bytes\": 0, \"sha256\": \"" + kEmptySha256 + "\"}";
    json += i + 1 == kArtifacts.size() ? "\n" : ",\n";
  }
  json +=
      "  ],\n"
      "  \"models\": [{\"name\": \"stage\", \"file\": \"stage.om\", \"bytes\": 1, \"sha256\": \"x\"}]\n"
      "}\n";
  const auto path = root / "manifest.json";
  WriteText(path, json);
  return path;
}

bool VerifyRejects(const mlvc::ModelManifest& manifest) {
  try {
    mlvc::VerifyRuntimeArtifacts(manifest);
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

}  // namespace

int main() {
  const auto unique_id = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto root = std::filesystem::temp_directory_path() /
                    ("mlvc_model_bundle_integrity_test_" + std::to_string(unique_id));
  try {
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const mlvc::ModelManifest manifest = mlvc::ModelManifest::Load(WriteManifest(root));
    mlvc::VerifyRuntimeArtifacts(manifest);
    const auto original_hash = mlvc::ComputeModelBundleSha256(manifest);

    for (const char* name : kArtifacts) {
      auto changed_manifest = manifest;
      auto artifact = std::find_if(
          changed_manifest.runtime_artifacts().begin(), changed_manifest.runtime_artifacts().end(),
          [name](const mlvc::RuntimeArtifactRecord& record) { return record.name == name; });
      Expect(artifact != changed_manifest.runtime_artifacts().end(),
             std::string("missing runtime artifact record: ") + name);
      artifact->sha256 = "changed";
      Expect(mlvc::ComputeModelBundleSha256(changed_manifest) != original_hash,
             std::string("bundle digest ignored runtime artifact: ") + name);

      WriteText(root / name, "tampered");
      Expect(VerifyRejects(manifest), std::string("tampered runtime artifact was accepted: ") + name);
      WriteText(root / name, "");
    }

    auto incomplete_manifest = manifest;
    incomplete_manifest.runtime_artifacts().pop_back();
    bool incomplete_rejected = false;
    try {
      (void)mlvc::ComputeModelBundleSha256(incomplete_manifest);
    } catch (const std::exception&) {
      incomplete_rejected = true;
    }
    Expect(incomplete_rejected, "bundle digest accepted a manifest without all runtime artifacts");
    std::filesystem::remove_all(root);
    std::cout << "model bundle integrity test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::filesystem::remove_all(root);
    std::cerr << "test_model_bundle_integrity failed: " << error.what() << "\n";
    return 1;
  }
}
