#include "mlvc/runtime/model_manifest.h"

#include <cstdint>
#include <fstream>
#include <regex>
#include <sstream>
#include <string_view>
#include <utility>

#include "mlvc/core/status.h"

namespace mlvc {
namespace {

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream input(path);
  Check(input.good(), "failed to open manifest: " + path.string());
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::string ExtractStringField(const std::string& text, const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
  std::smatch match;
  Check(std::regex_search(text, match, pattern), "missing string field: " + key);
  return match[1].str();
}

std::string ExtractOptionalStringField(const std::string& text, const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) {
    return "";
  }
  return match[1].str();
}

std::string ExtractModelName(const std::string& text) {
  const std::regex pattern("\"name\"\\s*:\\s*\"([^\"]*)\"");
  std::smatch match;
  Check(std::regex_search(text, match, pattern), "missing model name");
  return match[1].str();
}

uint64_t ExtractUIntField(const std::string& text, const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9]+)");
  std::smatch match;
  Check(std::regex_search(text, match, pattern), "missing integer field: " + key);
  return static_cast<uint64_t>(std::stoull(match[1].str()));
}

std::vector<std::string> ExtractModelBlocks(const std::string& text) {
  const std::string models_key = "\"models\"";
  const std::size_t key_pos = text.find(models_key);
  Check(key_pos != std::string::npos, "manifest missing models array");
  const std::size_t array_start = text.find('[', key_pos);
  Check(array_start != std::string::npos, "manifest models is not an array");

  std::vector<std::string> blocks;
  int depth = 0;
  std::size_t block_start = std::string::npos;
  for (std::size_t i = array_start + 1; i < text.size(); ++i) {
    if (text[i] == '{') {
      if (depth == 0) {
        block_start = i;
      }
      ++depth;
    } else if (text[i] == '}') {
      --depth;
      if (depth == 0 && block_start != std::string::npos) {
        blocks.push_back(text.substr(block_start, i - block_start + 1));
        block_start = std::string::npos;
      }
    } else if (text[i] == ']' && depth == 0) {
      break;
    }
  }
  return blocks;
}

std::vector<std::string> ExtractOptionalStringArrayField(const std::string& text,
                                                         const std::string& key) {
  const std::string quoted_key = "\"" + key + "\"";
  const std::size_t key_pos = text.find(quoted_key);
  if (key_pos == std::string::npos) {
    return {};
  }
  const std::size_t array_start = text.find('[', key_pos);
  Check(array_start != std::string::npos, "manifest array field is not an array: " + key);
  const std::size_t array_end = text.find(']', array_start);
  Check(array_end != std::string::npos, "unterminated manifest array field: " + key);
  const std::string array_text = text.substr(array_start + 1, array_end - array_start - 1);
  const std::regex item_pattern("\"([^\"]*)\"");
  std::vector<std::string> values;
  for (std::sregex_iterator it(array_text.begin(), array_text.end(), item_pattern), end; it != end;
       ++it) {
    values.push_back((*it)[1].str());
  }
  return values;
}

DataType ParseDataType(const std::string& value) {
  if (value == "float16" || value == "fp16") {
    return DataType::kFloat16;
  }
  if (value == "float32" || value == "fp32") {
    return DataType::kFloat32;
  }
  if (value == "int8") {
    return DataType::kInt8;
  }
  if (value == "int16") {
    return DataType::kInt16;
  }
  if (value == "int32") {
    return DataType::kInt32;
  }
  if (value == "uint8") {
    return DataType::kUInt8;
  }
  throw Error("unsupported manifest dtype: " + value);
}

std::string ExtractArrayText(const std::string& text, const std::string& key) {
  const std::string quoted_key = "\"" + key + "\"";
  const std::size_t key_pos = text.find(quoted_key);
  if (key_pos == std::string::npos) {
    return "";
  }
  const std::size_t array_start = text.find('[', key_pos);
  Check(array_start != std::string::npos, "manifest array field is not an array: " + key);
  int depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (std::size_t i = array_start; i < text.size(); ++i) {
    const char ch = text[i];
    if (in_string) {
      escaped = (ch == '\\' && !escaped);
      if (ch == '"' && !escaped) {
        in_string = false;
      } else if (ch != '\\') {
        escaped = false;
      }
      continue;
    }
    if (ch == '"') {
      in_string = true;
    } else if (ch == '[') {
      ++depth;
    } else if (ch == ']') {
      --depth;
      if (depth == 0) {
        return text.substr(array_start, i - array_start + 1);
      }
    }
  }
  throw Error("unterminated manifest array field: " + key);
}

std::vector<int64_t> ExtractShapeField(const std::string& text) {
  const std::string array = ExtractArrayText(text, "shape");
  if (array.empty()) {
    return {};
  }
  const std::regex item_pattern("-?[0-9]+");
  std::vector<int64_t> values;
  for (std::sregex_iterator it(array.begin(), array.end(), item_pattern), end; it != end; ++it) {
    values.push_back(std::stoll((*it)[0].str()));
  }
  return values;
}

std::vector<TensorSpec> ExtractTensorSpecs(const std::string& text, const std::string& key) {
  const std::string array = ExtractArrayText(text, key);
  if (array.empty()) {
    return {};
  }
  std::vector<TensorSpec> specs;
  for (const std::string& block : ExtractModelBlocks("{\"models\":" + array + "}")) {
    TensorSpec spec;
    spec.name = ExtractStringField(block, "name");
    spec.dtype = ParseDataType(ExtractStringField(block, "dtype"));
    spec.shape = ExtractShapeField(block);
    specs.push_back(std::move(spec));
  }
  return specs;
}

bool SameStringList(std::initializer_list<std::string_view> expected,
                    const std::vector<std::string>& actual) {
  if (expected.size() != actual.size()) {
    return false;
  }
  std::size_t index = 0;
  for (std::string_view value : expected) {
    if (actual[index] != value) {
      return false;
    }
    ++index;
  }
  return true;
}

SidecarRecord ParseSidecar(const std::string& text) {
  const std::string key = "\"sidecar\"";
  const std::size_t key_pos = text.find(key);
  Check(key_pos != std::string::npos, "manifest missing sidecar");
  const std::size_t object_start = text.find('{', key_pos);
  Check(object_start != std::string::npos, "manifest sidecar is not an object");

  int depth = 0;
  for (std::size_t i = object_start; i < text.size(); ++i) {
    if (text[i] == '{') {
      ++depth;
    } else if (text[i] == '}') {
      --depth;
      if (depth == 0) {
        const std::string block = text.substr(object_start, i - object_start + 1);
        SidecarRecord sidecar;
        sidecar.file = ExtractStringField(block, "file");
        sidecar.bytes = ExtractUIntField(block, "bytes");
        sidecar.sha256 = ExtractStringField(block, "sha256");
        return sidecar;
      }
    }
  }
  throw Error("unterminated manifest sidecar object");
}

}  // namespace

ModelManifest ModelManifest::Load(const std::filesystem::path& manifest_path) {
  const std::string text = ReadTextFile(manifest_path);
  ModelManifest manifest;
  manifest.path_ = std::filesystem::absolute(manifest_path);
  manifest.directory_ = manifest.path_.parent_path();
  manifest.runtime_ = ExtractOptionalStringField(text, "runtime");
  manifest.soc_version_ = ExtractOptionalStringField(text, "soc_version");
  manifest.dtype_ = ExtractStringField(text, "dtype");
  manifest.sidecar_ = ParseSidecar(text);

  for (const std::string& block : ExtractModelBlocks(text)) {
    ModelRecord record;
    record.name = ExtractModelName(block);
    Check(manifest.model_index_.find(record.name) == manifest.model_index_.end(),
          "duplicated model name in manifest: " + record.name);
    record.frame_type = ExtractOptionalStringField(block, "frame_type");
    record.route = ExtractOptionalStringField(block, "route");
    record.backend = ExtractOptionalStringField(block, "backend");
    record.model = ExtractOptionalStringField(block, "file");
    if (record.model.empty()) {
      record.model =
          manifest.runtime_ == "acl" ? record.name + ".sim.om" : record.name + ".sim.onnx";
    }
    record.onnx_model = ExtractOptionalStringField(block, "onnx_file");
    record.optimized_onnx_model = ExtractOptionalStringField(block, "optimized_onnx_file");
    record.atc_model = ExtractOptionalStringField(block, "atc_om_file");
    record.bytes = ExtractUIntField(block, "bytes");
    record.sha256 = ExtractStringField(block, "sha256");
    record.replaces = ExtractOptionalStringArrayField(block, "replaces");
    record.inputs = ExtractTensorSpecs(block, "inputs");
    record.outputs = ExtractTensorSpecs(block, "outputs");
    manifest.model_index_[record.name] = manifest.models_.size();
    manifest.models_.push_back(std::move(record));
  }

  Check(!manifest.models_.empty(), "manifest contains no models");
  return manifest;
}

bool ModelManifest::HasModel(const std::string& name) const {
  return model_index_.find(name) != model_index_.end();
}

const ModelRecord* ModelManifest::FindModel(const std::string& name) const {
  const auto it = model_index_.find(name);
  if (it == model_index_.end()) {
    return nullptr;
  }
  return &models_.at(it->second);
}

const ModelRecord* ModelManifest::FindFusionCandidate(
    std::initializer_list<std::string_view> replaces) const {
  for (const ModelRecord& record : models_) {
    if (SameStringList(replaces, record.replaces)) {
      return &record;
    }
  }
  return nullptr;
}

const ModelRecord& ModelManifest::GetModel(const std::string& name) const {
  const auto it = model_index_.find(name);
  Check(it != model_index_.end(), "unknown model in manifest: " + name);
  return models_.at(it->second);
}

}  // namespace mlvc
