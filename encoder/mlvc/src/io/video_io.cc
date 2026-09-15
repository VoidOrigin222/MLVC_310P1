#include <mlvc/io/video_io.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "mlvc/core/status.h"
#include "mlvc/io/fp16_yuv444_to_nv12.h"

namespace mlvc::io {
namespace codec = mlvc::codec;
namespace {

constexpr float kKr = 0.2126F;
constexpr float kKg = 0.7152F;
constexpr float kKb = 0.0722F;

std::string ToLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

uint8_t ClampToU8(float value) {
  value = std::min(std::max(value, 0.0F), 1.0F);
  return static_cast<uint8_t>(std::lrint(value * 255.0F));
}

cv::Mat TensorToBgrMat(const codec::TensorData& tensor, int width, int height) {
  Check(tensor.dtype == DataType::kFloat16, "decoded video output expects fp16 tensor");
  Check(tensor.shape.rank() == 4 && tensor.shape.dim(0) == 1 && tensor.shape.dim(1) == 3,
        "decoded video output expects NCHW frame tensor");
  Check(tensor.shape.dim(2) >= height && tensor.shape.dim(3) >= width,
        "decoded frame tensor is smaller than requested video size");

  const int padded_height = static_cast<int>(tensor.shape.dim(2));
  const int padded_width = static_cast<int>(tensor.shape.dim(3));
  const auto* input = reinterpret_cast<const uint16_t*>(tensor.bytes.data());
  cv::Mat frame(height, width, CV_8UC3);
  const std::size_t channel_stride = static_cast<std::size_t>(padded_height * padded_width);
  for (int y = 0; y < height; ++y) {
    auto* row = frame.ptr<cv::Vec3b>(y);
    for (int x = 0; x < width; ++x) {
      const std::size_t offset = static_cast<std::size_t>(y * padded_width + x);
      const float y_value = codec::HalfBitsToFloat(input[offset]);
      const float cb = codec::HalfBitsToFloat(input[channel_stride + offset]);
      const float cr = codec::HalfBitsToFloat(input[2 * channel_stride + offset]);
      const float r = y_value + (2.0F - 2.0F * kKr) * (cr - 0.5F);
      const float b = y_value + (2.0F - 2.0F * kKb) * (cb - 0.5F);
      const float g = (y_value - kKr * r - kKb * b) / kKg;
      row[x] = cv::Vec3b(ClampToU8(b), ClampToU8(g), ClampToU8(r));
    }
  }
  return frame;
}

std::string ShellQuote(const std::filesystem::path& path) {
  std::string quoted = "'";
  for (char c : path.string()) {
    if (c == '\'') {
      quoted += "'\\''";
    } else {
      quoted += c;
    }
  }
  quoted += "'";
  return quoted;
}

std::string BuildFfmpegCommand(const std::filesystem::path& output_path, double fps, int width,
                               int height, int crf, const std::string& bitrate,
                               const std::string& preset) {
  std::ostringstream command;
  command << "ffmpeg -y -hide_banner -loglevel error"
          << " -f rawvideo -pix_fmt bgr24"
          << " -s " << width << "x" << height << " -r " << fps << " -i pipe:0"
          << " -an -c:v libx264 -preset " << preset;
  if (bitrate.empty() || ToLower(bitrate) == "none") {
    command << " -crf " << crf;
  } else {
    command << " -b:v " << bitrate << " -maxrate " << bitrate << " -bufsize 5000k";
  }
  command << " -pix_fmt yuv420p -movflags +faststart " << ShellQuote(output_path);
  return command.str();
}

std::string BuildRtspCommand(const std::string& url, double fps, int width, int height,
                             const std::string& preset, int crf, const std::string& transport) {
  Check(!url.empty(), "RTSP output URL must not be empty");
  Check(fps > 0.0 && std::isfinite(fps), "RTSP FPS must be finite and positive");
  Check(width > 0 && height > 0, "RTSP dimensions must be positive");
  Check(!preset.empty(), "RTSP encoder preset must not be empty");
  Check(crf >= 0 && crf <= 51, "RTSP encoder CRF must be in [0, 51]");
  Check(transport == "tcp" || transport == "udp", "RTSP transport must be tcp or udp");
  std::ostringstream command;
  command << "ffmpeg -hide_banner -loglevel error -f rawvideo -pix_fmt nv12"
          << " -s " << width << "x" << height << " -r " << fps << " -i pipe:0"
          << " -an -c:v libx264 -preset " << preset << " -tune zerolatency"
          << " -crf " << crf << " -pix_fmt yuv420p -f rtsp -rtsp_transport " << transport << " "
          << ShellQuote(std::filesystem::path(url));
  return command.str();
}

}  // namespace

cv::Mat ConvertTensorToBgr(const codec::TensorData& tensor, int width, int height) {
  return TensorToBgrMat(tensor, width, height);
}

class VideoFrameReader::Impl {
 public:
  explicit Impl(const std::filesystem::path& input_path) : capture_(input_path.string()) {
    Check(capture_.isOpened(), "failed to open input video: " + input_path.string());
  }

  cv::VideoCapture capture_;
};

VideoFrameReader::VideoFrameReader(const std::filesystem::path& input_path)
    : impl_(std::make_unique<Impl>(input_path)) {
  info_.width = static_cast<int>(impl_->capture_.get(cv::CAP_PROP_FRAME_WIDTH));
  info_.height = static_cast<int>(impl_->capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
  info_.frame_count = static_cast<int>(impl_->capture_.get(cv::CAP_PROP_FRAME_COUNT));
  info_.fps = impl_->capture_.get(cv::CAP_PROP_FPS);
  if (info_.fps <= 0.0 || !std::isfinite(info_.fps)) {
    info_.fps = 30.0;
  }
  Check(info_.width > 0 && info_.height > 0, "failed to read video size from input stream");
}

VideoFrameReader::~VideoFrameReader() = default;

bool VideoFrameReader::ReadFrameAsTensor(const mlvc::TensorSpec& frame_spec,
                                         codec::TensorData* tensor) {
  Check(tensor != nullptr, "video frame tensor output is required");
  Check(frame_spec.dtype == DataType::kFloat16, "video input tensor expects fp16 frame spec");
  Check(frame_spec.shape.size() == 4 && frame_spec.shape[0] == 1 && frame_spec.shape[1] == 3,
        "video input tensor expects NCHW frame spec");

  cv::Mat frame;
  if (!impl_->capture_.read(frame)) {
    return false;
  }
  Check(frame.type() == CV_8UC3, "OpenCV returned an unsupported video frame type");

  const int padded_height = static_cast<int>(frame_spec.shape[2]);
  const int padded_width = static_cast<int>(frame_spec.shape[3]);
  Check(info_.height <= padded_height && info_.width <= padded_width,
        "input video dimensions exceed the model frame tensor shape");

  if (tensor->shape.dims() != frame_spec.shape || tensor->dtype != frame_spec.dtype) {
    *tensor = codec::MakeTensor(frame_spec.shape, frame_spec.dtype);
  }
  auto* output = reinterpret_cast<uint16_t*>(tensor->bytes.data());
  const std::size_t channel_stride = static_cast<std::size_t>(padded_height * padded_width);

  for (int y = 0; y < padded_height; ++y) {
    const int source_y = std::min(y, info_.height - 1);
    const auto* row = frame.ptr<cv::Vec3b>(source_y);
    for (int x = 0; x < padded_width; ++x) {
      const int source_x = std::min(x, info_.width - 1);
      const cv::Vec3b pixel = row[source_x];
      const float b = static_cast<float>(pixel[0]) / 255.0F;
      const float g = static_cast<float>(pixel[1]) / 255.0F;
      const float r = static_cast<float>(pixel[2]) / 255.0F;
      const float y_value = std::min(std::max(kKr * r + kKg * g + kKb * b, 0.0F), 1.0F);
      const float cb = std::min(std::max(0.5F * (b - y_value) / (1.0F - kKb) + 0.5F, 0.0F), 1.0F);
      const float cr = std::min(std::max(0.5F * (r - y_value) / (1.0F - kKr) + 0.5F, 0.0F), 1.0F);
      const std::size_t offset = static_cast<std::size_t>(y * padded_width + x);
      output[offset] = static_cast<uint16_t>(codec::FloatToHalfBits(y_value));
      output[channel_stride + offset] = static_cast<uint16_t>(codec::FloatToHalfBits(cb));
      output[2 * channel_stride + offset] = static_cast<uint16_t>(codec::FloatToHalfBits(cr));
    }
  }
  return true;
}

DecodedVideoWriter::DecodedVideoWriter(const std::filesystem::path& output_path,
                                       const std::string& output_format, double fps, int width,
                                       int height, int crf, const std::string& bitrate,
                                       const std::string& preset)
    : output_path_(output_path),
      output_format_(GuessDecodeOutputFormat(output_path, output_format)),
      fps_(fps),
      width_(width),
      height_(height),
      crf_(crf),
      bitrate_(bitrate),
      preset_(preset) {
  if (output_format_ == "png") {
    std::filesystem::create_directories(output_path_);
    return;
  }
  Check(output_format_ == "mp4", "decoded video output format must be mp4 or png");
  if (!output_path_.parent_path().empty()) {
    std::filesystem::create_directories(output_path_.parent_path());
  }
  const std::string command =
      BuildFfmpegCommand(output_path_, fps_, width_, height_, crf_, bitrate_, preset_);
  pipe_ = popen(command.c_str(), "w");
  Check(pipe_ != nullptr, "failed to open ffmpeg pipe for decoded video output");
}

DecodedVideoWriter::~DecodedVideoWriter() {
  if (pipe_ != nullptr) {
    const int status = pclose(pipe_);
    pipe_ = nullptr;
    (void)status;
  }
}

void DecodedVideoWriter::WriteTensorFrame(const codec::TensorData& tensor, int frame_index) {
  const cv::Mat frame = TensorToBgrMat(tensor, width_, height_);
  if (output_format_ == "png") {
    std::ostringstream name;
    name << "im" << std::setfill('0') << std::setw(5) << (frame_index + 1) << ".png";
    const std::filesystem::path path = output_path_ / name.str();
    Check(cv::imwrite(path.string(), frame), "failed to write PNG frame: " + path.string());
  } else {
    const std::size_t bytes = frame.total() * frame.elemSize();
    const std::size_t written = fwrite(frame.data, 1, bytes, pipe_);
    Check(written == bytes, "failed to write decoded frame to ffmpeg pipe");
  }
  ++frame_count_;
}

void DecodedVideoWriter::Close() {
  if (pipe_ == nullptr) {
    return;
  }
  const int status = pclose(pipe_);
  pipe_ = nullptr;
  Check(status == 0, "ffmpeg failed while writing decoded output video");
}

RtspVideoPublisher::RtspVideoPublisher(const std::string& url, double fps, int width, int height,
                                       const std::string& preset, int crf,
                                       std::size_t queue_capacity, const std::string& transport)
    : width_(width), height_(height), queue_capacity_(queue_capacity) {
  Check(queue_capacity_ > 0, "RTSP output queue capacity must be positive");
  const std::string command = BuildRtspCommand(url, fps, width_, height_, preset, crf, transport);
  pipe_ = popen(command.c_str(), "w");
  Check(pipe_ != nullptr, "failed to open FFmpeg RTSP publisher");
  worker_ = std::thread([this] {
    try {
      std::vector<uint8_t> reusable;
      const Nv12Layout layout{width_, height_, width_, height_};
      for (;;) {
        QueuedFrame queued;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
          if (queue_.empty() && stopping_) break;
          queued = std::move(queue_.front());
          queue_.pop_front();
        }
        if (queued.nv12.empty()) {
          ConvertFp16Yuv444ToNv12(queued.tensor, layout, &reusable);
        } else {
          reusable = std::move(queued.nv12);
        }
        const std::size_t bytes = reusable.size();
        Check(fwrite(reusable.data(), 1, bytes, pipe_) == bytes,
              "failed to write decoded frame to FFmpeg RTSP publisher");
        std::lock_guard<std::mutex> lock(mutex_);
        ++frame_count_;
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      worker_error_ = std::current_exception();
      stopping_ = true;
      cv_.notify_all();
    }
  });
}

RtspVideoPublisher::~RtspVideoPublisher() {
  try {
    Close();
  } catch (...) {
  }
}

void RtspVideoPublisher::WriteTensorFrame(const codec::TensorData& tensor) {
  std::lock_guard<std::mutex> lock(mutex_);
  Check(pipe_ != nullptr && !closed_, "RTSP publisher is closed");
  if (worker_error_ != nullptr) std::rethrow_exception(worker_error_);
  if (queue_.size() >= queue_capacity_) {
    ++dropped_frames_;
    return;
  }
  queue_.push_back(QueuedFrame{codec::CloneTensor(tensor), {}});
  cv_.notify_one();
}

void RtspVideoPublisher::WriteNv12Frame(std::vector<uint8_t> frame) {
  std::lock_guard<std::mutex> lock(mutex_);
  Check(pipe_ != nullptr && !closed_, "RTSP publisher is closed");
  if (worker_error_ != nullptr) std::rethrow_exception(worker_error_);
  if (queue_.size() >= queue_capacity_) {
    ++dropped_frames_;
    return;
  }
  queue_.push_back(QueuedFrame{{}, std::move(frame)});
  cv_.notify_one();
}

void RtspVideoPublisher::Close() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return;
    stopping_ = true;
    closed_ = true;
  }
  cv_.notify_all();
  if (worker_.joinable()) worker_.join();
  std::exception_ptr worker_error;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    worker_error = worker_error_;
  }
  const int status = pipe_ != nullptr ? pclose(pipe_) : 0;
  pipe_ = nullptr;
  if (worker_error != nullptr) std::rethrow_exception(worker_error);
  Check(status == 0, "FFmpeg RTSP publisher failed");
}

bool IsVideoPath(const std::filesystem::path& path) {
  const std::string extension = ToLower(path.extension().string());
  return extension == ".mp4" || extension == ".mov" || extension == ".mkv" || extension == ".avi" ||
         extension == ".webm";
}

std::string GuessDecodeOutputFormat(const std::filesystem::path& output_path,
                                    const std::string& configured_format) {
  if (!configured_format.empty()) {
    return ToLower(configured_format);
  }
  const std::string extension = ToLower(output_path.extension().string());
  if (extension == ".mp4" || extension == ".mov" || extension == ".mkv" || extension == ".avi" ||
      extension == ".webm") {
    return "mp4";
  }
  return "png";
}

void WriteTensorFrameAsPng(const codec::TensorData& tensor, int width, int height,
                           const std::filesystem::path& path) {
  std::filesystem::create_directories(path.parent_path());
  const cv::Mat frame = TensorToBgrMat(tensor, width, height);
  Check(cv::imwrite(path.string(), frame), "failed to write PNG frame: " + path.string());
}

}  // namespace mlvc::io
