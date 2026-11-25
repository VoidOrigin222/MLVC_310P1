#ifndef ULBVC_CODEC_TENSOR_DATA_H_
#define ULBVC_CODEC_TENSOR_DATA_H_

#include <ulbvc/core/tensor.h>
#include <ulbvc/runtime/model_manifest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string_view>
#include <vector>

namespace ulbvc::codec {

struct TensorData {
  ulbvc::TensorShape shape;
  ulbvc::DataType dtype = ulbvc::DataType::kFloat16;
  std::vector<uint8_t> bytes;
  void* external_data = nullptr;
  std::size_t external_bytes = 0;
  ulbvc::MemoryLocation external_location = ulbvc::MemoryLocation::kCpu;
  std::shared_ptr<void> external_owner;

  ulbvc::TensorView View() {
    if (external_data != nullptr) {
      return ulbvc::TensorView::Borrowed(external_data, &shape, dtype, external_location);
    }
    return ulbvc::TensorView::Borrowed(bytes.data(), &shape, dtype, ulbvc::MemoryLocation::kCpu);
  }

  ulbvc::TensorView View() const {
    if (external_data != nullptr) {
      return ulbvc::TensorView::Borrowed(external_data, &shape, dtype, external_location);
    }
    return ulbvc::TensorView::Borrowed(const_cast<uint8_t*>(bytes.data()), &shape, dtype,
                                      ulbvc::MemoryLocation::kCpu);
  }

  void AttachExternalBuffer(void* data, std::size_t byte_count,
                            ulbvc::MemoryLocation location, std::shared_ptr<void> owner);
  void ClearExternalBuffer();
  std::size_t ByteSize() const { return external_data != nullptr ? external_bytes : bytes.size(); }
  bool has_external_buffer() const { return external_data != nullptr; }

  std::size_t Elements() const { return shape.NumElements(); }
};

std::size_t TensorBytes(const TensorData& tensor);
uint32_t FloatToHalfBits(float value);
float HalfBitsToFloat(uint16_t value);
float RoundToFp16(float value);
TensorData MakeTensor(const std::vector<int64_t>& shape, ulbvc::DataType dtype);
TensorData MakeTensorLike(const ulbvc::TensorSpec& spec);
TensorData CloneTensor(const TensorData& tensor);
void CloneTensorInto(const TensorData& tensor, TensorData* output);
const ulbvc::TensorSpec& OutputSpec(const ulbvc::ModelRecord& record, std::string_view name);
TensorData MakeFp16Tensor(const std::vector<int64_t>& shape, float value);
void FillFp16Tensor(float value, TensorData* tensor);
TensorData ReadTensorFile(const std::filesystem::path& path, const std::vector<int64_t>& shape,
                          ulbvc::DataType dtype);
void ReadTensorFileInto(const std::filesystem::path& path, TensorData* tensor);
void WriteTensorFile(const std::filesystem::path& path, const TensorData& tensor);

}  // namespace ulbvc::codec

#endif  // ULBVC_CODEC_TENSOR_DATA_H_
