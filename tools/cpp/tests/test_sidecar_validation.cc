#include <mlvc/core/status.h>
#include <mlvc/entropy/sidecar.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
template <typename T>
void Put(std::vector<uint8_t>* out, T value) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
  out->insert(out->end(), bytes, bytes + sizeof(T));
}
void PutArray(std::vector<uint8_t>* out, const std::string& name, uint32_t dtype,
              const std::vector<int64_t>& shape, uint64_t declared_bytes,
              const std::vector<uint8_t>& bytes) {
  Put<uint32_t>(out, static_cast<uint32_t>(name.size()));
  out->insert(out->end(), name.begin(), name.end());
  Put<uint32_t>(out, dtype);
  Put<uint32_t>(out, static_cast<uint32_t>(shape.size()));
  for (int64_t dim : shape) Put<int64_t>(out, dim);
  Put<uint64_t>(out, declared_bytes);
  out->insert(out->end(), bytes.begin(), bytes.end());
}
void Write(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
}
std::vector<uint8_t> Base(uint32_t count = 1) {
  std::vector<uint8_t> bytes({'U', 'L', 'B', 'V', 'C', '_', 'S', 'C'});
  Put<uint32_t>(&bytes, 1);
  Put<uint32_t>(&bytes, count);
  return bytes;
}
template <typename F>
void Reject(F&& f, const char* message) {
  bool rejected = false;
  try {
    f();
  } catch (const std::exception&) {
    rejected = true;
  }
  mlvc::Check(rejected, message);
}
}  // namespace

int main() {
  try {
    const auto root = std::filesystem::temp_directory_path();
    const auto valid = root / "mlvc_sidecar_valid.bin";
    auto bytes = Base();
    const float value = 0.5f;
    std::vector<uint8_t> scalar(sizeof(value));
    std::memcpy(scalar.data(), &value, sizeof(value));
    PutArray(&bytes, "force_zero_thres", 1, {1}, sizeof(value), scalar);
    Write(valid, bytes);
    const auto loaded = mlvc::RuntimeSidecar::Load(valid);
    mlvc::Check(loaded.force_zero_thres() == value, "valid sidecar failed to load");

    auto trailing = bytes;
    trailing.push_back(0);
    const auto trailing_path = root / "mlvc_sidecar_trailing.bin";
    Write(trailing_path, trailing);
    Reject([&] { (void)mlvc::RuntimeSidecar::Load(trailing_path); },
           "trailing sidecar bytes accepted");

    auto duplicate = Base(2);
    PutArray(&duplicate, "x", 2, {1}, 4, {1, 2, 3, 4});
    PutArray(&duplicate, "x", 2, {1}, 4, {1, 2, 3, 4});
    const auto duplicate_path = root / "mlvc_sidecar_duplicate.bin";
    Write(duplicate_path, duplicate);
    Reject([&] { (void)mlvc::RuntimeSidecar::Load(duplicate_path); },
           "duplicate sidecar name accepted");

    auto mismatch = Base();
    PutArray(&mismatch, "x", 2, {1}, 8, {1, 2, 3, 4});
    const auto mismatch_path = root / "mlvc_sidecar_mismatch.bin";
    Write(mismatch_path, mismatch);
    Reject([&] { (void)mlvc::RuntimeSidecar::Load(mismatch_path); },
           "sidecar byte mismatch accepted");

    auto negative = Base();
    PutArray(&negative, "x", 2, {-1}, 0, {});
    const auto negative_path = root / "mlvc_sidecar_negative.bin";
    Write(negative_path, negative);
    Reject([&] { (void)mlvc::RuntimeSidecar::Load(negative_path); },
           "negative sidecar dimension accepted");
    std::cout << "sidecar validation test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_sidecar_validation failed: " << error.what() << "\n";
    return 1;
  }
}
