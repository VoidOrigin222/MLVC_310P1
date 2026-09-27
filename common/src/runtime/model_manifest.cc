#include "mlvc/runtime/model_manifest.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

#include "mlvc/core/status.h"

namespace mlvc {
namespace {

constexpr std::array<uint32_t, 64> kSha256K = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
    0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
    0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
    0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

uint32_t RotR(uint32_t value, unsigned count) {
  return (value >> count) | (value << (32u - count));
}

void Sha256Block(const uint8_t* block, std::array<uint32_t, 8>* state) {
  std::array<uint32_t, 64> words{};
  for (std::size_t i = 0; i < 16; ++i) {
    words[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
               (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
               (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
               static_cast<uint32_t>(block[i * 4 + 3]);
  }
  for (std::size_t i = 16; i < words.size(); ++i) {
    const uint32_t s0 = RotR(words[i - 15], 7) ^ RotR(words[i - 15], 18) ^ (words[i - 15] >> 3);
    const uint32_t s1 = RotR(words[i - 2], 17) ^ RotR(words[i - 2], 19) ^ (words[i - 2] >> 10);
    words[i] = words[i - 16] + s0 + words[i - 7] + s1;
  }
  uint32_t a = (*state)[0], b = (*state)[1], c = (*state)[2], d = (*state)[3];
  uint32_t e = (*state)[4], f = (*state)[5], g = (*state)[6], h = (*state)[7];
  for (std::size_t i = 0; i < words.size(); ++i) {
    const uint32_t t1 = h + (RotR(e, 6) ^ RotR(e, 11) ^ RotR(e, 25)) +
                        ((e & f) ^ (~e & g)) + kSha256K[i] + words[i];
    const uint32_t t2 = (RotR(a, 2) ^ RotR(a, 13) ^ RotR(a, 22)) +
                        ((a & b) ^ (a & c) ^ (b & c));
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  (*state)[0] += a;
  (*state)[1] += b;
  (*state)[2] += c;
  (*state)[3] += d;
  (*state)[4] += e;
  (*state)[5] += f;
  (*state)[6] += g;
  (*state)[7] += h;
}

std::array<uint8_t, 32> Sha256(const std::vector<uint8_t>& input) {
  std::array<uint32_t, 8> state = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  std::array<uint8_t, 64> block{};
  std::size_t block_size = 0;
  for (uint8_t byte : input) {
    block[block_size++] = byte;
    if (block_size == block.size()) {
      Sha256Block(block.data(), &state);
      block_size = 0;
    }
  }
  block[block_size++] = 0x80;
  if (block_size > 56) {
    std::fill(block.begin() + static_cast<std::ptrdiff_t>(block_size), block.end(), 0);
    Sha256Block(block.data(), &state);
    block_size = 0;
  }
  std::fill(block.begin() + static_cast<std::ptrdiff_t>(block_size), block.begin() + 56, 0);
  const uint64_t bit_length = static_cast<uint64_t>(input.size()) * 8u;
  for (unsigned i = 0; i < 8; ++i) {
    block[56 + i] = static_cast<uint8_t>(bit_length >> (56u - i * 8u));
  }
  Sha256Block(block.data(), &state);
  std::array<uint8_t, 32> digest{};
  for (std::size_t i = 0; i < state.size(); ++i) {
    for (unsigned shift = 0; shift < 4; ++shift) {
      digest[i * 4 + shift] = static_cast<uint8_t>(state[i] >> (24u - shift * 8u));
    }
  }
  return digest;
}

void AppendU32(std::vector<uint8_t>* bytes, uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8) {
    bytes->push_back(static_cast<uint8_t>(value >> shift));
  }
}

void AppendU64(std::vector<uint8_t>* bytes, uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    bytes->push_back(static_cast<uint8_t>(value >> shift));
  }
}

void AppendString(std::vector<uint8_t>* bytes, const std::string& value) {
  Check(value.size() <= std::numeric_limits<uint32_t>::max(),
        "manifest string is too large for bundle hash");
  AppendU32(bytes, static_cast<uint32_t>(value.size()));
  bytes->insert(bytes->end(), value.begin(), value.end());
}

void AppendTensorSpecs(std::vector<uint8_t>* bytes, const std::vector<TensorSpec>& specs) {
  Check(specs.size() <= std::numeric_limits<uint32_t>::max(),
        "manifest tensor list is too large for bundle hash");
  AppendU32(bytes, static_cast<uint32_t>(specs.size()));
  for (const TensorSpec& spec : specs) {
    AppendString(bytes, spec.name);
    AppendU32(bytes, static_cast<uint32_t>(spec.dtype));
    Check(spec.shape.size() <= std::numeric_limits<uint32_t>::max(),
          "manifest tensor shape is too large for bundle hash");
    AppendU32(bytes, static_cast<uint32_t>(spec.shape.size()));
    for (int64_t dimension : spec.shape) {
      AppendU64(bytes, static_cast<uint64_t>(dimension));
    }
  }
}

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

std::array<uint8_t, 32> ComputeModelBundleSha256(const ModelManifest& manifest) {
  // Length-prefix every field so that two different manifests cannot collide
  // merely because their textual fields happen to concatenate identically.
  std::vector<uint8_t> canonical;
  AppendString(&canonical, "MLVC-MODEL-BUNDLE-V1");
  AppendString(&canonical, manifest.runtime());
  AppendString(&canonical, manifest.soc_version());
  AppendString(&canonical, manifest.dtype());
  const SidecarRecord& sidecar = manifest.sidecar();
  AppendU64(&canonical, sidecar.bytes);
  AppendString(&canonical, sidecar.sha256);
  Check(manifest.models().size() <= std::numeric_limits<uint32_t>::max(),
        "manifest model list is too large for bundle hash");
  AppendU32(&canonical, static_cast<uint32_t>(manifest.models().size()));
  for (const ModelRecord& model : manifest.models()) {
    AppendString(&canonical, model.name);
    AppendString(&canonical, model.frame_type);
    AppendString(&canonical, model.route);
    AppendString(&canonical, model.backend);
    AppendU64(&canonical, model.bytes);
    AppendString(&canonical, model.sha256);
    Check(model.replaces.size() <= std::numeric_limits<uint32_t>::max(),
          "manifest replacement list is too large for bundle hash");
    AppendU32(&canonical, static_cast<uint32_t>(model.replaces.size()));
    for (const std::string& replacement : model.replaces) {
      AppendString(&canonical, replacement);
    }
    AppendTensorSpecs(&canonical, model.inputs);
    AppendTensorSpecs(&canonical, model.outputs);
  }
  return Sha256(canonical);
}

}  // namespace mlvc
