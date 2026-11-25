#ifndef ULBVC_IO_FRAME_SOURCE_H_
#define ULBVC_IO_FRAME_SOURCE_H_

#include <ulbvc/codec/tensor_data.h>
#include <ulbvc/io/video_io.h>

#include <filesystem>
#include <optional>

namespace ulbvc::io {

class FrameSource {
 public:
  FrameSource(int configured_frame_num, std::filesystem::path input_video_path,
              std::filesystem::path input_frame_dir, ulbvc::TensorSpec frame_spec,
              std::optional<CameraCaptureOptions> camera_options = std::nullopt,
              aclrtContext context = nullptr);

  bool ReadFrame(int frame_index, codec::TensorData* frame,
                 std::vector<uint8_t>* motion_nv12 = nullptr);
  void Close();
  bool has_video_input() const { return video_reader_.has_value(); }
  const std::filesystem::path& input_frame_dir() const { return input_frame_dir_; }

 private:
  int configured_frame_num_ = 0;
  std::filesystem::path input_frame_dir_;
  ulbvc::TensorSpec frame_spec_;
  std::optional<VideoFrameReader> video_reader_;
};

}  // namespace ulbvc::io

#endif  // ULBVC_IO_FRAME_SOURCE_H_
