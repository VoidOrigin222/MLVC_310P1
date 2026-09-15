#ifndef MLVC_CODEC_DETAIL_REFERENCE_STATE_H_
#define MLVC_CODEC_DETAIL_REFERENCE_STATE_H_

#include <mlvc/codec/tensor_data.h>
#include <mlvc/core/tensor_handle.h>

#include <optional>

namespace mlvc::codec {

struct ReferenceState {
  std::optional<TensorData> frame;
  std::optional<TensorData> feature;
  std::optional<mlvc::TensorHandle> frame_handle;
  std::optional<mlvc::TensorHandle> feature_handle;

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
  const mlvc::TensorHandle* handle = nullptr;
};

struct SourceFrameGeometry {
  int width = 0;
  int height = 0;
  int padded_width = 0;
  int padded_height = 0;

  bool has_source_size() const { return width > 0 && height > 0; }
};

}  // namespace mlvc::codec

#endif  // MLVC_CODEC_DETAIL_REFERENCE_STATE_H_
