#include "mlvc/entropy/sidecar.h"

#include <cmath>
#include <cstring>
#include <fstream>

#include "mlvc/core/status.h"

namespace mlvc {
namespace {

constexpr char kMagic[] = "ULBVC_SC";
constexpr uint32_t kVersion = 1;

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

}  // namespace

std::size_t SidecarArray::ElementCount() const {
  if (shape.empty()) {
    return 0;
  }
  std::size_t count = 1;
  for (int64_t dim : shape) {
    count *= static_cast<std::size_t>(dim);
  }
  return count;
}

std::vector<int32_t> SidecarArray::AsInt32() const {
  Check(dtype == DataType::kInt32, "sidecar array is not int32: " + name);
  std::vector<int32_t> values(ElementCount());
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

std::vector<uint16_t> SidecarArray::AsFloat16Bits() const {
  Check(dtype == DataType::kFloat16, "sidecar array is not float16: " + name);
  std::vector<uint16_t> values(ElementCount());
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

float SidecarArray::ScalarFloat32() const {
  Check(dtype == DataType::kFloat32, "sidecar array is not float32: " + name);
  Check(bytes.size() == sizeof(float), "sidecar float32 scalar has invalid size: " + name);
  float value = 0.0f;
  std::memcpy(&value, bytes.data(), sizeof(value));
  return value;
}

RuntimeSidecar RuntimeSidecar::Load(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  Check(input.good(), "failed to open converted sidecar: " + path.string());

  char magic[sizeof(kMagic)] = {};
  input.read(magic, sizeof(kMagic) - 1);
  Check(std::string(magic, sizeof(kMagic) - 1) == kMagic, "invalid sidecar magic");

  const uint32_t version = ReadPod<uint32_t>(input);
  Check(version == kVersion, "unsupported sidecar version");
  const uint32_t array_count = ReadPod<uint32_t>(input);

  RuntimeSidecar sidecar;
  for (uint32_t i = 0; i < array_count; ++i) {
    const uint32_t name_size = ReadPod<uint32_t>(input);
    SidecarArray array;
    array.name.resize(name_size);
    input.read(array.name.data(), name_size);
    Check(input.good(), "failed to read sidecar array name");
    array.dtype = ReadDataType(input);
    const uint32_t rank = ReadPod<uint32_t>(input);
    array.shape.resize(rank);
    for (uint32_t axis = 0; axis < rank; ++axis) {
      array.shape[axis] = ReadPod<int64_t>(input);
    }
    const uint64_t byte_count = ReadPod<uint64_t>(input);
    array.bytes.resize(static_cast<std::size_t>(byte_count));
    input.read(reinterpret_cast<char*>(array.bytes.data()),
               static_cast<std::streamsize>(byte_count));
    Check(input.good(), "failed to read sidecar array bytes");
    sidecar.arrays_.emplace(array.name, std::move(array));
  }
  return sidecar;
}

const SidecarArray& RuntimeSidecar::Get(const std::string& name) const {
  auto it = arrays_.find(name);
  Check(it != arrays_.end(), "missing sidecar array: " + name);
  return it->second;
}

float RuntimeSidecar::force_zero_thres() const { return Get("force_zero_thres").ScalarFloat32(); }

float RuntimeSidecar::python_fast_force_zero_thres() const {
  const float value = force_zero_thres();
  if (value < 0.0f) {
    return value;
  }
  const float rounded = std::round(value * 100.0f) / 100.0f;
  if (std::abs(value - rounded) <= 1.0e-6f) {
    return rounded;
  }
  return value;
}

int RuntimeSidecar::z_channel(const std::string& prefix) const {
  const auto values = Get(prefix + "_z_channel").AsInt32();
  Check(values.size() == 1, "z_channel must be scalar");
  return values[0];
}

int RuntimeSidecar::ShiftedQp(int base_qp, int frame_adaptation_index) const {
  const auto offsets = Get("p_qp_offsets").AsInt32();
  Check(frame_adaptation_index >= 0 && frame_adaptation_index < static_cast<int>(offsets.size()),
        "invalid frame adaptation index");
  return base_qp + offsets[frame_adaptation_index];
}

const uint16_t* RuntimeSidecar::QScaleData(const std::string& name, int qp) const {
  const SidecarArray& array = Get(name);
  Check(array.dtype == DataType::kFloat16, "Q scale array is not fp16: " + name);
  Check(!array.shape.empty(), "Q scale array has empty shape: " + name);
  Check(qp >= 0 && qp < array.shape[0], "QP is out of range for " + name);
  std::size_t row_elements = 1;
  for (std::size_t axis = 1; axis < array.shape.size(); ++axis) {
    row_elements *= static_cast<std::size_t>(array.shape[axis]);
  }
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
