#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "mlvc/core/status.h"
#include "mlvc/runtime/acl_runtime.h"
#include "mlvc/runtime/model_manifest.h"

namespace {

void ExpectErrorContains(const char* label, const std::string& actual,
                         const std::string& expected) {
  mlvc::Check(actual.find(expected) != std::string::npos,
              std::string(label) + " error mismatch: " + actual);
}

mlvc::ModelManifest LoadManifest(const std::filesystem::path& path) {
  return mlvc::ModelManifest::Load(path);
}

void WriteText(const std::filesystem::path& path, const std::string& text) {
  std::ofstream output(path);
  mlvc::Check(output.good(), "failed to open temp manifest: " + path.string());
  output << text;
}

std::string AclManifestWithModelFields(const std::string& fields) {
  return std::string(R"json({
  "created_at": "2026-05-16T00:00:00Z",
  "runtime": "acl",
  "soc_version": "Ascend310P3",
  "dtype": "fp16",
  "sidecar": {"file": "sidecars.mlvcsc", "bytes": 1, "sha256": "00"},
  "models": [
    {
      "name": "stage",
      "bytes": 1,
      "sha256": "01",
      "frame_type": "I",
      "route": "encode",
      "inputs": [{"name": "x", "dtype": "float16", "shape": [1]}],
      "outputs": [{"name": "y", "dtype": "float16", "shape": [1]}],
)json") + fields +
         R"json(
    }
  ]
})json";
}

std::string ValidAclModelFields() {
  return R"json(
      "file": "om_aoe/stage.sim.om",
      "onnx_file": "onnx_original/stage.sim.onnx",
      "optimized_onnx_file": "onnx_optimized/stage.sim.onnx",
      "atc_om_file": "om_atc/stage.sim.om",
      "backend": "acl"
)json";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " <legacy-manifest.json> <acl-manifest.json>\n";
    return 2;
  }

  try {
    try {
      mlvc::ValidateAclManifestForAclRuntime(LoadManifest(argv[1]));
      throw mlvc::Error("legacy manifest was unexpectedly accepted by ACL runtime");
    } catch (const mlvc::Error& error) {
      ExpectErrorContains("legacy manifest", error.what(), "runtime = \"acl\"");
    }

    try {
      const mlvc::ModelManifest manifest = LoadManifest(argv[2]);
      mlvc::ValidateAclManifestForAclRuntime(manifest);
      mlvc::Check(manifest.runtime() == "acl", "accepted manifest did not preserve runtime = acl");
    } catch (const std::exception& error) {
      throw mlvc::Error(std::string("ACL manifest was unexpectedly rejected: ") + error.what());
    }

    const std::filesystem::path temp_dir =
        std::filesystem::temp_directory_path() / "mlvc_acl_manifest_policy_check";
    std::filesystem::create_directories(temp_dir);
    const std::filesystem::path bad_backend = temp_dir / "bad_backend.json";
    WriteText(bad_backend, AclManifestWithModelFields(R"json(
      "file": "om_aoe/stage.sim.om",
      "onnx_file": "onnx_original/stage.sim.onnx",
      "optimized_onnx_file": "onnx_optimized/stage.sim.onnx",
      "atc_om_file": "om_atc/stage.sim.om",
      "backend": "cpu"
)json"));
    try {
      mlvc::ValidateAclManifestForAclRuntime(LoadManifest(bad_backend));
      throw mlvc::Error("bad backend ACL manifest was unexpectedly accepted");
    } catch (const mlvc::Error& error) {
      ExpectErrorContains("bad backend manifest", error.what(), "backend = \"acl\"");
    }

    const std::filesystem::path bad_path = temp_dir / "bad_path.json";
    WriteText(bad_path, AclManifestWithModelFields(R"json(
      "file": "om_atc/stage.sim.om",
      "onnx_file": "onnx_original/stage.sim.onnx",
      "optimized_onnx_file": "onnx_optimized/stage.sim.onnx",
      "atc_om_file": "om_atc/stage.sim.om",
      "backend": "acl"
)json"));
    try {
      mlvc::ValidateAclManifestForAclRuntime(LoadManifest(bad_path));
      throw mlvc::Error("bad path ACL manifest was unexpectedly accepted");
    } catch (const mlvc::Error& error) {
      ExpectErrorContains("bad path manifest", error.what(), "om_aoe");
    }

    const std::filesystem::path bad_ext = temp_dir / "bad_ext.json";
    WriteText(bad_ext, AclManifestWithModelFields(R"json(
      "file": "onnx_optimized/stage.sim.onnx",
      "onnx_file": "onnx_original/stage.sim.onnx",
      "optimized_onnx_file": "onnx_optimized/stage.sim.onnx",
      "atc_om_file": "om_atc/stage.sim.om",
      "backend": "acl"
)json"));
    try {
      mlvc::ValidateAclManifestForAclRuntime(LoadManifest(bad_ext));
      throw mlvc::Error("bad extension ACL manifest was unexpectedly accepted");
    } catch (const mlvc::Error& error) {
      ExpectErrorContains("bad extension manifest", error.what(), "OM model files");
    }

    const std::filesystem::path bad_dtype = temp_dir / "bad_dtype.json";
    WriteText(bad_dtype, R"json({
  "created_at": "2026-05-16T00:00:00Z",
  "runtime": "acl",
  "soc_version": "Ascend310P3",
  "dtype": "fp32",
  "sidecar": {"file": "sidecars.mlvcsc", "bytes": 1, "sha256": "00"},
  "models": [
    {
      "name": "stage",
      "bytes": 1,
      "sha256": "01",
      "frame_type": "I",
      "route": "encode",
      "inputs": [{"name": "x", "dtype": "float16", "shape": [1]}],
      "outputs": [{"name": "y", "dtype": "float16", "shape": [1]}],
)json" + ValidAclModelFields() +
                             R"json(
    }
  ]
})json");
    try {
      mlvc::ValidateAclManifestForAclRuntime(LoadManifest(bad_dtype));
      throw mlvc::Error("bad dtype ACL manifest was unexpectedly accepted");
    } catch (const mlvc::Error& error) {
      ExpectErrorContains("bad dtype manifest", error.what(), "dtype = fp16");
    }

    const std::filesystem::path missing_traceability = temp_dir / "missing_traceability.json";
    WriteText(missing_traceability, AclManifestWithModelFields(R"json(
      "file": "om_aoe/stage.sim.om",
      "backend": "acl"
)json"));
    try {
      mlvc::ValidateAclManifestForAclRuntime(LoadManifest(missing_traceability));
      throw mlvc::Error("missing traceability ACL manifest was unexpectedly accepted");
    } catch (const mlvc::Error& error) {
      ExpectErrorContains("missing traceability manifest", error.what(), "onnx_file");
    }

    const std::filesystem::path bad_input_dtype = temp_dir / "bad_input_dtype.json";
    WriteText(bad_input_dtype, R"json({
  "created_at": "2026-05-16T00:00:00Z",
  "runtime": "acl",
  "soc_version": "Ascend310P3",
  "dtype": "fp16",
  "sidecar": {"file": "sidecars.mlvcsc", "bytes": 1, "sha256": "00"},
  "models": [
    {
      "name": "stage",
      "bytes": 1,
      "sha256": "01",
      "frame_type": "I",
      "route": "encode",
      "inputs": [{"name": "x", "dtype": "float32", "shape": [1]}],
      "outputs": [{"name": "y", "dtype": "float16", "shape": [1]}],
)json" + ValidAclModelFields() + R"json(
    }
  ]
})json");
    try {
      mlvc::ValidateAclManifestForAclRuntime(LoadManifest(bad_input_dtype));
      throw mlvc::Error("bad input dtype ACL manifest was unexpectedly accepted");
    } catch (const mlvc::Error& error) {
      ExpectErrorContains("bad input dtype manifest", error.what(), "requires FP16 input");
    }

    const std::filesystem::path int8_z_output = temp_dir / "int8_z_output.json";
    WriteText(int8_z_output, R"json({
  "created_at": "2026-05-16T00:00:00Z",
  "runtime": "acl",
  "soc_version": "Ascend310P3",
  "dtype": "fp16",
  "sidecar": {"file": "sidecars.mlvcsc", "bytes": 1, "sha256": "00"},
  "models": [
    {
      "name": "stage",
      "bytes": 1,
      "sha256": "01",
      "frame_type": "I",
      "route": "encode",
      "inputs": [{"name": "x", "dtype": "float16", "shape": [1]}],
      "outputs": [{"name": "z_symbols", "dtype": "int8", "shape": [1]}],
)json" + ValidAclModelFields() + R"json(
    }
  ]
})json");
    try {
      mlvc::ValidateAclManifestForAclRuntime(LoadManifest(int8_z_output));
      throw mlvc::Error("int8 z_symbols ACL manifest was unexpectedly accepted");
    } catch (const mlvc::Error& error) {
      ExpectErrorContains("int8 z_symbols manifest", error.what(), "requires FP16 output");
    }

    std::cout << "acl_manifest_policy status=ok\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "check_acl_manifest_policy failed: " << error.what() << "\n";
    return 1;
  }
}
