#ifndef ULBVC_CODEC_DETAIL_STAGE_TYPES_H_
#define ULBVC_CODEC_DETAIL_STAGE_TYPES_H_

#include <ulbvc/codec/detail/stage/constants.h>
#include <ulbvc/codec/tensor_data.h>
#include <ulbvc/core/status.h>
#include <ulbvc/core/tensor_handle.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace ulbvc::codec {

struct TensorBytesView {
  ulbvc::TensorShape shape;
  ulbvc::DataType dtype = ulbvc::DataType::kFloat16;
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
  const ulbvc::TensorHandle* handle = nullptr;
};

inline StageInput TensorInput(const char* name, const TensorData& tensor) {
  return StageInput{name, &tensor, nullptr};
}

inline StageInput TensorOrHandleInput(const char* name, const TensorData& tensor,
                                      const ulbvc::TensorHandle* handle) {
  return handle != nullptr ? StageInput{name, nullptr, handle} : StageInput{name, &tensor, nullptr};
}

struct RunOutput {
  struct Entry {
    const std::string* name = nullptr;
    TensorData* tensor = nullptr;
    ulbvc::TensorHandle* handle = nullptr;
  };

  TensorData& At(std::string_view name) {
    for (std::size_t i = 0; i < size; ++i) {
      if (tensors[i].name != nullptr && *tensors[i].name == name) {
        return *tensors[i].tensor;
      }
    }
    throw ulbvc::Error("missing stage output tensor: " + std::string(name));
  }

  const TensorData& At(std::string_view name) const {
    for (std::size_t i = 0; i < size; ++i) {
      if (tensors[i].name != nullptr && *tensors[i].name == name) {
        return *tensors[i].tensor;
      }
    }
    throw ulbvc::Error("missing stage output tensor: " + std::string(name));
  }

  ulbvc::TensorHandle* Handle(std::string_view name) {
    for (std::size_t i = 0; i < size; ++i) {
      if (tensors[i].name != nullptr && *tensors[i].name == name) {
        return tensors[i].handle;
      }
    }
    throw ulbvc::Error("missing stage output tensor handle: " + std::string(name));
  }

  const ulbvc::TensorHandle* Handle(std::string_view name) const {
    for (std::size_t i = 0; i < size; ++i) {
      if (tensors[i].name != nullptr && *tensors[i].name == name) {
        return tensors[i].handle;
      }
    }
    throw ulbvc::Error("missing stage output tensor handle: " + std::string(name));
  }

  std::size_t Size() const { return size; }

  std::array<Entry, kMaxStageOutputs> tensors;
  std::size_t size = 0;
};

}  // namespace ulbvc::codec

#endif  // ULBVC_CODEC_DETAIL_STAGE_TYPES_H_
