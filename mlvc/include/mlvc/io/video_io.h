#ifndef MLVC_IO_VIDEO_IO_H_
#define MLVC_IO_VIDEO_IO_H_

#include <mlvc/codec/tensor_data.h>

#include <cstdio>
#include <filesystem>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <string>

namespace mlvc::io {

struct VideoInfo {
  int width = 0;
  int height = 0;
  int frame_count = 0;
  double fps = 30.0;
};

class VideoFrameReader {
 public:
  explicit VideoFrameReader(const std::filesystem::path& input_path);
  VideoFrameReader(const VideoFrameReader&) = delete;
  VideoFrameReader& operator=(const VideoFrameReader&) = delete;
  ~VideoFrameReader();

  const VideoInfo& info() const { return info_; }
  bool ReadFrameAsTensor(const mlvc::TensorSpec& frame_spec, codec::TensorData* tensor);

 private:
  class Impl;

  std::unique_ptr<Impl> impl_;
  VideoInfo info_;
};

class DecodedVideoWriter {
 public:
  DecodedVideoWriter(const std::filesystem::path& output_path, const std::string& output_format,
                     double fps, int width, int height, int crf, const std::string& bitrate,
                     const std::string& preset);
  DecodedVideoWriter(const DecodedVideoWriter&) = delete;
  DecodedVideoWriter& operator=(const DecodedVideoWriter&) = delete;
  ~DecodedVideoWriter();

  void WriteTensorFrame(const codec::TensorData& tensor, int frame_index);
  int frame_count() const { return frame_count_; }
  void Close();

 private:
  std::filesystem::path output_path_;
  std::string output_format_;
  double fps_ = 30.0;
  int width_ = 0;
  int height_ = 0;
  int crf_ = 23;
  std::string bitrate_;
  std::string preset_;
  FILE* pipe_ = nullptr;
  int frame_count_ = 0;
};

bool IsVideoPath(const std::filesystem::path& path);
std::string GuessDecodeOutputFormat(const std::filesystem::path& output_path,
                                    const std::string& configured_format);
cv::Mat ConvertTensorToBgr(const codec::TensorData& tensor, int width, int height);
void WriteTensorFrameAsPng(const codec::TensorData& tensor, int width, int height,
                           const std::filesystem::path& path);

}  // namespace mlvc::io

#endif  // MLVC_IO_VIDEO_IO_H_
