#ifndef MLVC_CODEC_DETAIL_STAGE_TYPES_H_
#define MLVC_CODEC_DETAIL_STAGE_TYPES_H_

#include <mlvc/codec/tensor_data.h>
#include <mlvc/core/status.h>
#include <mlvc/core/tensor_handle.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include <mlvc/codec/detail/stage/constants.h>

namespace mlvc::codec {

struct TensorBytesView {
  mlvc::TensorShape shape;
  mlvc::DataType dtype = mlvc::DataType::kFloat16;
  const uint8_t* bytes = nullptr;
  std::size_t byte_count = 0;

  std::size_t Elements() const { return shape.NumElements(); }
};

struct AsyncEntropyInputs {
  TensorBytesView z_symbols;
  std::array<TensorBytesView, 4> y_symbols;
  std::array<TensorBytesView, 4> scales;
  std::array<TensorBytesView, 4> y_combined_indexes;
  std::array<TensorBytesView, 4> y_keep_masks;
  int part_count = 0;
  bool y_indexes_prebuilt = false;
};

struct PinnedCopySpan {
  double start_ms = 0.0;
  double duration_ms = 0.0;
};

struct StageInput {
  const char* name = nullptr;
  const TensorData* tensor = nullptr;
  const mlvc::TensorHandle* handle = nullptr;
};

inline StageInput TensorInput(const char* name, const TensorData& tensor) {
  return StageInput{name, &tensor, nullptr};
}

inline StageInput TensorOrHandleInput(const char* name, const TensorData& tensor,
                                      const mlvc::TensorHandle* handle) {
  return handle != nullptr ? StageInput{name, nullptr, handle} : StageInput{name, &tensor, nullptr};
}

struct RunOutput {
  struct Entry {
    const std::string* name = nullptr;
    TensorData* tensor = nullptr;
    mlvc::TensorHandle* handle = nullptr;
  };

  TensorData& At(std::string_view name) {
    for (std::size_t i = 0; i < size; ++i) {
      if (tensors[i].name != nullptr && *tensors[i].name == name) {
        return *tensors[i].tensor;
      }
    }
    throw mlvc::Error("missing stage output tensor: " + std::string(name));
  }

  const TensorData& At(std::string_view name) const {
    for (std::size_t i = 0; i < size; ++i) {
      if (tensors[i].name != nullptr && *tensors[i].name == name) {
        return *tensors[i].tensor;
      }
    }
    throw mlvc::Error("missing stage output tensor: " + std::string(name));
  }

  mlvc::TensorHandle* Handle(std::string_view name) {
    for (std::size_t i = 0; i < size; ++i) {
      if (tensors[i].name != nullptr && *tensors[i].name == name) {
        return tensors[i].handle;
      }
    }
    throw mlvc::Error("missing stage output tensor handle: " + std::string(name));
  }

  const mlvc::TensorHandle* Handle(std::string_view name) const {
    for (std::size_t i = 0; i < size; ++i) {
      if (tensors[i].name != nullptr && *tensors[i].name == name) {
        return tensors[i].handle;
      }
    }
    throw mlvc::Error("missing stage output tensor handle: " + std::string(name));
  }

  std::size_t Size() const { return size; }

  std::array<Entry, kMaxStageOutputs> tensors;
  std::size_t size = 0;
};

}  // namespace mlvc::codec

#endif  // MLVC_CODEC_DETAIL_STAGE_TYPES_H_
