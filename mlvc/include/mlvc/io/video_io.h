#ifndef MLVC_IO_VIDEO_IO_H_
#define MLVC_IO_VIDEO_IO_H_

#include <mlvc/codec/tensor_data.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <string>
#include <thread>
#include <vector>

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

// Publishes decoded frames to a MediaMTX RTSP endpoint through FFmpeg.
// Conversion and pipe writes run on a dedicated worker so decode is not
// blocked by network backpressure.
class RtspVideoPublisher {
 public:
  RtspVideoPublisher(const std::string& url, double fps, int width, int height,
                     const std::string& preset = "ultrafast", int crf = 0,
                     std::size_t queue_capacity = 3, const std::string& transport = "udp",
                     bool h264_copy = false);
  RtspVideoPublisher(const RtspVideoPublisher&) = delete;
  RtspVideoPublisher& operator=(const RtspVideoPublisher&) = delete;
  ~RtspVideoPublisher();

  void WriteTensorFrame(const codec::TensorData& tensor);
  void WriteNv12Frame(std::vector<uint8_t> frame);
  void WriteH264Frame(std::vector<uint8_t> frame);
  void Close();
  int frame_count() const { return frame_count_; }
  uint64_t dropped_frames() const { return dropped_frames_; }

 private:
  FILE* pipe_ = nullptr;
  int width_ = 0;
  int height_ = 0;
  int frame_count_ = 0;
  uint64_t dropped_frames_ = 0;
  std::size_t queue_capacity_ = 3;
  bool h264_copy_ = false;
  struct QueuedFrame {
    codec::TensorData tensor;
    std::vector<uint8_t> nv12;
    std::vector<uint8_t> h264;
  };
  std::deque<QueuedFrame> queue_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::thread worker_;
  bool stopping_ = false;
  bool closed_ = false;
  std::exception_ptr worker_error_;
};

bool IsVideoPath(const std::filesystem::path& path);
std::string GuessDecodeOutputFormat(const std::filesystem::path& output_path,
                                    const std::string& configured_format);
cv::Mat ConvertTensorToBgr(const codec::TensorData& tensor, int width, int height);
void WriteTensorFrameAsPng(const codec::TensorData& tensor, int width, int height,
                           const std::filesystem::path& path);

}  // namespace mlvc::io

#endif  // MLVC_IO_VIDEO_IO_H_
