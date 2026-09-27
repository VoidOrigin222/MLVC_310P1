#include "mlvc/entropy/sidecar.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>

#include "mlvc/core/status.h"

namespace mlvc {
namespace {

constexpr char kMagic[] = "ULBVC_SC";
constexpr uint32_t kVersion = 1;

constexpr uint32_t kMaxArrayCount = 1024;
constexpr uint32_t kMaxNameLength = 256;
constexpr uint32_t kMaxRank = 8;
constexpr uint64_t kMaxTotalFileSize = 256 * 1024 * 1024;
constexpr uint64_t kMaxSingleArraySize = 64 * 1024 * 1024;

DataType ReadDataType(std::istream& input) {
  uint32_t value = 0;
  input.read(reinterpret_cast<char*>(&value), sizeof(value));
  switch (value) {
    case 0:
      return DataType::kFloat16;
    case 1:
      return DataType::kFloat32;
    case 2:
      return DataType::kInt32;
    default:
      throw Error("unsupported sidecar dtype id");
  }
}

template <typename T>
T ReadPod(std::istream& input) {
  T value{};
  input.read(reinterpret_cast<char*>(&value), sizeof(T));
  Check(input.good(), "failed to read sidecar payload");
  return value;
}

bool MultiplyWouldOverflow(uint64_t a, uint64_t b) {
  if (a == 0 || b == 0) return false;
  return a > std::numeric_limits<uint64_t>::max() / b;
}

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream input(path);
  Check(input.good(), "failed to open model metadata: " + path.string());
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::size_t FindJsonValue(const std::string& json, const std::string& key) {
  const std::string quoted_key = "\"" + key + "\"";
  const std::size_t key_pos = json.find(quoted_key);
  Check(key_pos != std::string::npos, "model metadata is missing key: " + key);
  const std::size_t colon = json.find(':', key_pos + quoted_key.size());
  Check(colon != std::string::npos, "model metadata key has no value: " + key);
  return colon + 1;
}

void SkipJsonWhitespace(const std::string& json, std::size_t* pos) {
  while (*pos < json.size() && std::isspace(static_cast<unsigned char>(json[*pos]))) {
    ++*pos;
  }
}

int32_t ReadJsonInt(const std::string& json, std::size_t* pos, const std::string& key) {
  SkipJsonWhitespace(json, pos);
  bool negative = false;
  if (*pos < json.size() && json[*pos] == '-') {
    negative = true;
    ++*pos;
  }
  const std::size_t digits_begin = *pos;
  int64_t value = 0;
  while (*pos < json.size() && std::isdigit(static_cast<unsigned char>(json[*pos]))) {
    const int digit = json[*pos] - '0';
    Check(value <= (std::numeric_limits<int64_t>::max() - digit) / 10,
          "model metadata integer overflows: " + key);
    value = value * 10 + digit;
    ++*pos;
  }
  Check(*pos != digits_begin, "model metadata has an invalid integer: " + key);
  if (negative) value = -value;
  Check(value >= std::numeric_limits<int32_t>::min() &&
            value <= std::numeric_limits<int32_t>::max(),
        "model metadata integer is out of range: " + key);
  return static_cast<int32_t>(value);
}

std::vector<int32_t> ReadJsonIntArray(const std::string& json, const std::string& key) {
  std::size_t pos = FindJsonValue(json, key);
  SkipJsonWhitespace(json, &pos);
  Check(pos < json.size() && json[pos] == '[', "model metadata value is not an array: " + key);
  ++pos;
  std::vector<int32_t> values;
  while (true) {
    SkipJsonWhitespace(json, &pos);
    Check(pos < json.size(), "unterminated model metadata array: " + key);
    if (json[pos] == ']') {
      ++pos;
      break;
    }
    values.push_back(ReadJsonInt(json, &pos, key));
    SkipJsonWhitespace(json, &pos);
    Check(pos < json.size(), "unterminated model metadata array: " + key);
    if (json[pos] == ',') {
      ++pos;
      continue;
    }
    Check(json[pos] == ']', "invalid model metadata array separator: " + key);
    ++pos;
    break;
  }
  Check(!values.empty(), "model metadata array is empty: " + key);
  return values;
}

int32_t ReadJsonIntValue(const std::string& json, const std::string& key) {
  std::size_t pos = FindJsonValue(json, key);
  return ReadJsonInt(json, &pos, key);
}

}  // namespace

std::size_t SidecarArray::ElementCount() const {
  if (shape.empty()) {
    return 0;
  }
  std::size_t count = 1;
  for (const auto dim : shape) {
    Check(dim >= 0, "sidecar array dimension must be non-negative, got " + std::to_string(dim));
    if (MultiplyWouldOverflow(count, static_cast<uint64_t>(dim))) {
      throw Error("sidecar array element count overflow");
    }
    count *= static_cast<std::size_t>(dim);
  }
  return count;
}

std::vector<int32_t> SidecarArray::AsInt32() const {
  Check(dtype == DataType::kInt32, "sidecar array is not int32");
  const std::size_t element_count = ElementCount();
  Check(element_count <= std::numeric_limits<std::size_t>::max() / sizeof(int32_t),
        "sidecar int32 byte count overflow");
  const std::size_t expected_bytes = element_count * sizeof(int32_t);
  Check(bytes.size() == expected_bytes, "sidecar array byte count mismatch: expected " +
                                            std::to_string(expected_bytes) + ", got " +
                                            std::to_string(bytes.size()));
  std::vector<int32_t> result(element_count);
  std::memcpy(result.data(), bytes.data(), expected_bytes);
  return result;
}

std::vector<uint16_t> SidecarArray::AsFloat16Bits() const {
  Check(dtype == DataType::kFloat16, "sidecar array is not float16");
  const std::size_t element_count = ElementCount();
  Check(element_count <= std::numeric_limits<std::size_t>::max() / sizeof(uint16_t),
        "sidecar float16 byte count overflow");
  const std::size_t expected_bytes = element_count * sizeof(uint16_t);
  Check(bytes.size() == expected_bytes, "sidecar array byte count mismatch: expected " +
                                            std::to_string(expected_bytes) + ", got " +
                                            std::to_string(bytes.size()));
  std::vector<uint16_t> result(element_count);
  std::memcpy(result.data(), bytes.data(), expected_bytes);
  return result;
}

float SidecarArray::ScalarFloat32() const {
  Check(dtype == DataType::kFloat32, "sidecar array is not float32");
  Check(shape.size() == 1 && shape[0] == 1, "sidecar array is not scalar");
  Check(bytes.size() == sizeof(float), "sidecar scalar has incorrect byte count");
  float value = 0.0f;
  std::memcpy(&value, bytes.data(), sizeof(float));
  return value;
}

RuntimeSidecar RuntimeSidecar::Load(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  Check(input.good(), "failed to open sidecar: " + path.string());

  input.seekg(0, std::ios::end);
  const std::streampos file_size = input.tellg();
  Check(file_size >= 16, "sidecar file is truncated");
  Check(static_cast<uint64_t>(file_size) <= kMaxTotalFileSize,
        "sidecar file too large: " + std::to_string(file_size) + " > " +
            std::to_string(kMaxTotalFileSize));
  input.seekg(0, std::ios::beg);

  char magic[8] = {};
  input.read(magic, 8);
  Check(input.good() && std::memcmp(magic, kMagic, 8) == 0,
        "invalid sidecar magic: " + path.string());

  const uint32_t version = ReadPod<uint32_t>(input);
  Check(version == kVersion, "unsupported sidecar version: " + std::to_string(version));

  const uint32_t array_count = ReadPod<uint32_t>(input);
  Check(array_count <= kMaxArrayCount,
        "sidecar array count exceeds maximum: " + std::to_string(array_count) + " > " +
            std::to_string(kMaxArrayCount));

  RuntimeSidecar sidecar;
  uint64_t total_bytes_read = 16;

  for (uint32_t i = 0; i < array_count; ++i) {
    SidecarArray array;

    const uint32_t name_length = ReadPod<uint32_t>(input);
    Check(name_length > 0 && name_length <= kMaxNameLength,
          "sidecar array name length out of range: " + std::to_string(name_length));
    total_bytes_read += sizeof(uint32_t);

    array.name.resize(name_length);
    input.read(array.name.data(), name_length);
    Check(input.good(), "failed to read sidecar array name");
    total_bytes_read += name_length;

    array.dtype = ReadDataType(input);
    total_bytes_read += sizeof(uint32_t);

    const uint32_t rank = ReadPod<uint32_t>(input);
    Check(rank > 0 && rank <= kMaxRank, "sidecar array rank out of range: " + std::to_string(rank));
    total_bytes_read += sizeof(uint32_t);

    array.shape.resize(rank);
    for (uint32_t j = 0; j < rank; ++j) {
      const int64_t dim = ReadPod<int64_t>(input);
      Check(dim >= 0, "sidecar array dimension must be non-negative, got " + std::to_string(dim));
      array.shape[j] = dim;
      total_bytes_read += sizeof(int64_t);
    }

    const uint64_t byte_count = ReadPod<uint64_t>(input);
    total_bytes_read += sizeof(uint64_t);

    const std::size_t element_count = array.ElementCount();
    const std::size_t element_size = ElementSize(array.dtype);
    if (MultiplyWouldOverflow(element_count, element_size)) {
      throw Error("sidecar array byte count calculation overflow for array: " + array.name);
    }
    const uint64_t expected_byte_count = element_count * element_size;
    Check(byte_count == expected_byte_count,
          "sidecar array byte count mismatch for " + array.name + ": declared " +
              std::to_string(byte_count) + ", expected " + std::to_string(expected_byte_count));

    Check(byte_count <= kMaxSingleArraySize, "sidecar array too large: " + array.name + " has " +
                                                 std::to_string(byte_count) + " bytes > " +
                                                 std::to_string(kMaxSingleArraySize));

    Check(total_bytes_read + byte_count <= static_cast<uint64_t>(file_size),
          "sidecar array byte count exceeds remaining file size for " + array.name);

    array.bytes.resize(static_cast<std::size_t>(byte_count));
    input.read(reinterpret_cast<char*>(array.bytes.data()),
               static_cast<std::streamsize>(byte_count));
    Check(input.good(), "failed to read sidecar array data: " + array.name);
    total_bytes_read += byte_count;

    Check(sidecar.arrays_.find(array.name) == sidecar.arrays_.end(),
          "duplicate sidecar array name: " + array.name);
    sidecar.arrays_.emplace(array.name, std::move(array));
  }

  Check(total_bytes_read == static_cast<uint64_t>(file_size),
        "sidecar contains trailing or truncated bytes: consumed " +
            std::to_string(total_bytes_read) + " of " + std::to_string(file_size));

  return sidecar;
}

RuntimeSidecar RuntimeSidecar::LoadQpShiftMetadata(const std::filesystem::path& path) {
  const std::string json = ReadText(path);
  RuntimeSidecar metadata;
  metadata.metadata_qp_shift_ = ReadJsonIntArray(json, "qp_shift");
  metadata.metadata_qp_index_count_ = ReadJsonIntValue(json, "total_qp_num");
  Check(metadata.metadata_qp_index_count_ > 0,
        "model metadata total_qp_num must be positive");
  const int32_t max_shift = *std::max_element(metadata.metadata_qp_shift_.begin(),
                                               metadata.metadata_qp_shift_.end());
  Check(max_shift < metadata.metadata_qp_index_count_,
        "model metadata qp_shift exceeds total_qp_num");
  return metadata;
}

const SidecarArray& RuntimeSidecar::Get(const std::string& name) const {
  const auto it = arrays_.find(name);
  Check(it != arrays_.end(), "sidecar array not found: " + name);
  return it->second;
}

float RuntimeSidecar::force_zero_thres() const { return Get("force_zero_thres").ScalarFloat32(); }

float RuntimeSidecar::python_fast_force_zero_thres() const {
  return Get("python_fast_force_zero_thres").ScalarFloat32();
}

int RuntimeSidecar::z_channel(const std::string& prefix) const {
  const SidecarArray& array = Get(prefix + "_z_channel");
  Check(array.dtype == DataType::kInt32, "z_channel array is not int32: " + prefix);
  Check(array.shape.size() == 1 && array.shape[0] == 1, "z_channel array is not scalar");
  const auto values = array.AsInt32();
  return values[0];
}

int RuntimeSidecar::q_index_count() const {
  if (metadata_qp_index_count_ > 0) return metadata_qp_index_count_;
  int qp_rows = 64;
  bool found_q_scale = false;
  for (const auto& [name, candidate] : arrays_) {
    if (name.find("_q_scale") == std::string::npos || candidate.shape.empty()) continue;
    Check(candidate.shape[0] > 0 && candidate.shape[0] <= std::numeric_limits<int>::max(),
          "invalid Q scale row count: " + name);
    qp_rows = std::min(qp_rows, static_cast<int>(candidate.shape[0]));
    found_q_scale = true;
  }
  Check(found_q_scale, "sidecar contains no Q scale tables");
  return qp_rows;
}

int RuntimeSidecar::ShiftedQp(int base_qp, int frame_adaptation_index) const {
  Check(frame_adaptation_index >= 0, "frame adaptation index must be non-negative");
  if (!metadata_qp_shift_.empty()) {
    Check(static_cast<std::size_t>(frame_adaptation_index) < metadata_qp_shift_.size(),
          "frame adaptation index out of range");
    const int64_t shifted = static_cast<int64_t>(base_qp) +
                            metadata_qp_shift_[static_cast<std::size_t>(frame_adaptation_index)];
    Check(shifted >= std::numeric_limits<int>::min() &&
              shifted <= std::numeric_limits<int>::max(),
          "shifted QP integer overflow");
    return std::clamp(static_cast<int>(shifted), 0, q_index_count() - 1);
  }
  // q_shift is optional for backward compatibility with older sidecar files
  const auto it = arrays_.find("q_shift");
  if (it == arrays_.end()) {
    // Older sidecars do not carry a q_shift table, but their Q scale tables
    // still define the only valid runtime range.  Do not let an out-of-range
    // base QP bypass that contract.
    return std::clamp(base_qp, 0, q_index_count() - 1);
  }

  const SidecarArray& array = it->second;
  Check(array.dtype == DataType::kInt32, "q_shift array is not int32");
  Check(array.shape.size() == 1 && array.shape[0] > frame_adaptation_index,
        "frame adaptation index out of range");
  const auto values = array.AsInt32();
  const int64_t shifted = static_cast<int64_t>(base_qp) +
                          static_cast<int64_t>(values[frame_adaptation_index]);
  Check(shifted >= std::numeric_limits<int>::min() &&
            shifted <= std::numeric_limits<int>::max(),
        "shifted QP integer overflow");

  // All runtime q-scale tables use the first dimension as the supported QP
  // rows.  Clamp to the intersection of those tables so a shifted QP can
  // never select a row that one of the codec stages cannot materialize.
  const int qp_rows = q_index_count();
  return std::clamp(static_cast<int>(shifted), 0, qp_rows - 1);
}

const uint16_t* RuntimeSidecar::QScaleData(const std::string& name, int qp) const {
  const SidecarArray& array = Get(name);
  Check(array.dtype == DataType::kFloat16, "Q scale array is not fp16: " + name);
  Check(!array.shape.empty(), "Q scale array has empty shape: " + name);
  Check(qp >= 0 && qp < array.shape[0], "QP is out of range for " + name);
  std::size_t row_elements = 1;
  for (std::size_t axis = 1; axis < array.shape.size(); ++axis) {
    Check(array.shape[axis] >= 0, "Q scale shape has a negative dimension: " + name);
    const std::size_t dim = static_cast<std::size_t>(array.shape[axis]);
    if (dim == 0) {
      row_elements = 0;
      continue;
    }
    Check(row_elements <= std::numeric_limits<std::size_t>::max() / dim,
          "Q scale row element count overflow: " + name);
    row_elements *= dim;
  }
  Check(array.ElementCount() <= std::numeric_limits<std::size_t>::max() / sizeof(uint16_t) &&
            array.bytes.size() == array.ElementCount() * sizeof(uint16_t),
        "Q scale byte count mismatch: " + name);
  return reinterpret_cast<const uint16_t*>(array.bytes.data()) + qp * row_elements;
}

std::shared_ptr<CdfGroup> RuntimeSidecar::MakeCdfGroupForPrefix(const std::string& prefix,
                                                                bool z_table) const {
  const std::string kind = z_table ? "z" : "gaussian";
  const SidecarArray& cdf = Get(prefix + "_" + kind + "_cdf");
  const SidecarArray& cdf_length = Get(prefix + "_" + kind + "_cdf_length");
  const SidecarArray& offset = Get(prefix + "_" + kind + "_offset");
  return MakeCdfGroup(cdf.AsInt32(), cdf.shape, cdf_length.AsInt32(), offset.AsInt32());
}

}  // namespace mlvc
