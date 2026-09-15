#include <mlvc/io/frame_source.h>

#include <string>
#include <utility>

#include "mlvc/core/status.h"

namespace mlvc::io {
namespace codec = mlvc::codec;

FrameSource::FrameSource(int configured_frame_num, std::filesystem::path input_video_path,
                         std::filesystem::path input_frame_dir, mlvc::TensorSpec frame_spec)
    : configured_frame_num_(configured_frame_num),
      input_frame_dir_(std::move(input_frame_dir)),
      frame_spec_(std::move(frame_spec)) {
  if (!input_video_path.empty()) {
    video_reader_.emplace(input_video_path);
  }
}

bool FrameSource::ReadFrame(int frame_index, codec::TensorData* frame) {
  if (video_reader_.has_value()) {
    if (!video_reader_->ReadFrameAsTensor(frame_spec_, frame)) {
      if (configured_frame_num_ < 0 && frame_index > 0) {
        return false;
      }
      throw mlvc::Error("input video ended before requested frame count");
    }
    return true;
  }

  if (input_frame_dir_.empty()) {
    codec::FillFp16Tensor(frame_index == 0 ? 0.5f : 0.51f, frame);
    return true;
  }

  const std::filesystem::path frame_path =
      input_frame_dir_ / ("frame_" + std::to_string(frame_index) + ".fp16");
  if (!std::filesystem::exists(frame_path)) {
    if (configured_frame_num_ < 0 && frame_index > 0) {
      return false;
    }
    throw mlvc::Error("missing input frame: " + frame_path.string());
  }
  codec::ReadTensorFileInto(frame_path, frame);
  return true;
}

}  // namespace mlvc::io
