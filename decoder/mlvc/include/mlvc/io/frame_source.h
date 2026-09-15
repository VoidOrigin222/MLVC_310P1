#ifndef MLVC_IO_FRAME_SOURCE_H_
#define MLVC_IO_FRAME_SOURCE_H_

#include <mlvc/codec/tensor_data.h>
#include <mlvc/io/video_io.h>

#include <filesystem>
#include <optional>

namespace mlvc::io {

class FrameSource {
 public:
  FrameSource(int configured_frame_num, std::filesystem::path input_video_path,
              std::filesystem::path input_frame_dir, mlvc::TensorSpec frame_spec);

  bool ReadFrame(int frame_index, codec::TensorData* frame);
  bool has_video_input() const { return video_reader_.has_value(); }
  const std::filesystem::path& input_frame_dir() const { return input_frame_dir_; }

 private:
  int configured_frame_num_ = 0;
  std::filesystem::path input_frame_dir_;
  mlvc::TensorSpec frame_spec_;
  std::optional<VideoFrameReader> video_reader_;
};

}  // namespace mlvc::io

#endif  // MLVC_IO_FRAME_SOURCE_H_
