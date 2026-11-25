#ifndef ULBVC_CODEC_DETAIL_REFERENCE_STATE_H_
#define ULBVC_CODEC_DETAIL_REFERENCE_STATE_H_

#include <ulbvc/codec/tensor_data.h>
#include <ulbvc/core/tensor_handle.h>

#include <optional>

namespace ulbvc::codec {

struct ReferenceState {
  std::optional<TensorData> frame;
  std::optional<TensorData> feature;
  std::optional<ulbvc::TensorHandle> frame_handle;
  std::optional<ulbvc::TensorHandle> feature_handle;

  void ResetFrame() {
    frame.reset();
    frame_handle.reset();
  }

  void ResetFeature() {
    feature.reset();
    feature_handle.reset();
  }
};

struct ReferenceFeatureBinding {
  TensorData* tensor = nullptr;
  const ulbvc::TensorHandle* handle = nullptr;
};

struct SourceFrameGeometry {
  int width = 0;
  int height = 0;
  int padded_width = 0;
  int padded_height = 0;

  bool has_source_size() const { return width > 0 && height > 0; }
};

}  // namespace ulbvc::codec

#endif  // ULBVC_CODEC_DETAIL_REFERENCE_STATE_H_
