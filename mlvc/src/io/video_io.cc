#include <mlvc/io/video_io.h>
#include <mlvc/io/fp16_yuv444_to_nv12.h>
#include <mlvc/io/dvpp_jpeg_decoder.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <csignal>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <acl/ops/acl_dvpp.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/rational.h>
}

#include "mlvc/core/status.h"
#include "mlvc/core/buffer.h"
#include "mlvc/runtime/acl_runtime.h"
#include "mlvc/io/fp16_yuv444_to_nv12.h"
#include "mlvc/io/dvpp_h264_encoder.h"

namespace mlvc::io {
namespace codec = mlvc::codec;
namespace {

// Strip JPEGD stride padding without changing any visible pixel values.
void CopyVisibleNv12(const void* source, std::size_t bytes, const Nv12Layout& layout,
                     std::vector<uint8_t>* output) {
  Check(source != nullptr && output != nullptr && layout.width > 0 && layout.height > 0 &&
            layout.width % 2 == 0 && layout.height % 2 == 0 &&
            layout.width_stride >= layout.width && layout.height_stride >= layout.height,
        "camera motion NV12 layout is invalid");
  Check(bytes >= Nv12BufferSize(layout), "camera motion NV12 storage is too short");
  const Nv12Layout packed{layout.width, layout.height, layout.width, layout.height};
  output->resize(Nv12BufferSize(packed));
  const auto* input = static_cast<const uint8_t*>(source);
  const auto source_stride = static_cast<std::size_t>(layout.width_stride);
  const auto target_stride = static_cast<std::size_t>(layout.width);
  const auto source_uv = source_stride * layout.height_stride;
  const auto target_uv = target_stride * layout.height;
  for (int row = 0; row < layout.height; ++row)
    std::memcpy(output->data() + row * target_stride, input + row * source_stride, target_stride);
  for (int row = 0; row < layout.height / 2; ++row)
    std::memcpy(output->data() + target_uv + row * target_stride,
                input + source_uv + row * source_stride, target_stride);
}

struct V4l2Buffer {
  void* address = MAP_FAILED;
  std::size_t length = 0;
};

int Align16(int value) { return (value + 15) / 16 * 16; }

class V4l2MjpegCapture {
 public:
  class FrameLease {
   public:
    FrameLease() = default;
    FrameLease(const uint8_t* data, std::size_t bytes, uint32_t sequence,
               std::function<void()> requeue)
        : data_(data), bytes_(bytes), sequence_(sequence), requeue_(std::move(requeue)) {}
    FrameLease(const FrameLease&) = delete;
    FrameLease& operator=(const FrameLease&) = delete;
    FrameLease(FrameLease&& other) noexcept
        : data_(other.data_), bytes_(other.bytes_), sequence_(other.sequence_),
          requeue_(std::move(other.requeue_)) {
      other.data_ = nullptr;
      other.bytes_ = 0;
    }
    FrameLease& operator=(FrameLease&& other) noexcept {
      if (this != &other) {
        ReleaseNoThrow();
        data_ = other.data_;
        bytes_ = other.bytes_;
        sequence_ = other.sequence_;
        requeue_ = std::move(other.requeue_);
        other.data_ = nullptr;
        other.bytes_ = 0;
      }
      return *this;
    }
    ~FrameLease() { ReleaseNoThrow(); }

    const uint8_t* data() const { return data_; }
    std::size_t bytes() const { return bytes_; }
    uint32_t sequence() const { return sequence_; }
    void Release() {
      if (!requeue_) return;
      auto requeue = std::move(requeue_);
      data_ = nullptr;
      bytes_ = 0;
      requeue();
    }

   private:
    void ReleaseNoThrow() noexcept {
      try {
        Release();
      } catch (...) {
      }
    }

    const uint8_t* data_ = nullptr;
    std::size_t bytes_ = 0;
    uint32_t sequence_ = 0;
    std::function<void()> requeue_;
  };

  V4l2MjpegCapture(const std::filesystem::path& device,
                   const CameraCaptureOptions& options) {
    Check(options.width > 0 && options.height > 0 &&
              std::isfinite(options.fps) && options.fps > 0.0 && options.fps <= 240.0,
          "invalid camera dimensions or frame rate");
    fd_ = open(device.c_str(), O_RDWR | O_NONBLOCK);
    Check(fd_ >= 0, "failed to open camera: " + device.string());
    try {
      v4l2_capability capability{};
      Ioctl(VIDIOC_QUERYCAP, &capability, "VIDIOC_QUERYCAP");
      const uint32_t caps = (capability.capabilities & V4L2_CAP_DEVICE_CAPS)
                                ? capability.device_caps
                                : capability.capabilities;
      Check((caps & V4L2_CAP_VIDEO_CAPTURE) != 0 && (caps & V4L2_CAP_STREAMING) != 0,
            "camera must support V4L2 video capture and streaming");
      v4l2_format format{};
      format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      format.fmt.pix.width = static_cast<uint32_t>(options.width);
      format.fmt.pix.height = static_cast<uint32_t>(options.height);
      Check(options.pixel_format.size() == 4, "camera pixel format must be four characters");
      format.fmt.pix.pixelformat = v4l2_fourcc(options.pixel_format[0], options.pixel_format[1],
                                               options.pixel_format[2], options.pixel_format[3]);
      format.fmt.pix.field = V4L2_FIELD_ANY;
      Ioctl(VIDIOC_S_FMT, &format, "VIDIOC_S_FMT");
      width_ = static_cast<int>(format.fmt.pix.width);
      height_ = static_cast<int>(format.fmt.pix.height);
      Check(width_ == options.width && height_ == options.height &&
                format.fmt.pix.pixelformat == v4l2_fourcc('M', 'J', 'P', 'G'),
            "camera did not accept the requested MJPEG format and dimensions");
      v4l2_streamparm parameters{};
      parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      parameters.parm.capture.timeperframe.numerator = 1000;
      parameters.parm.capture.timeperframe.denominator =
          static_cast<uint32_t>(std::lround(options.fps * 1000.0));
      Ioctl(VIDIOC_S_PARM, &parameters, "VIDIOC_S_PARM");
      const auto& interval = parameters.parm.capture.timeperframe;
      Check(interval.numerator != 0 && interval.denominator != 0,
            "camera returned an invalid frame interval");
      const double actual_fps = static_cast<double>(interval.denominator) / interval.numerator;
      Check(std::abs(actual_fps - options.fps) <= std::max(0.5, options.fps * 0.02),
            "camera did not accept the requested frame rate");
      v4l2_requestbuffers request{};
      request.count = 4;
      request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      request.memory = V4L2_MEMORY_MMAP;
      Ioctl(VIDIOC_REQBUFS, &request, "VIDIOC_REQBUFS");
      Check(request.count > 0, "camera returned no V4L2 buffers");
      buffers_.resize(request.count);
      for (uint32_t index = 0; index < request.count; ++index) {
        v4l2_buffer buffer{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = index;
        Ioctl(VIDIOC_QUERYBUF, &buffer, "VIDIOC_QUERYBUF");
        buffers_[index].length = buffer.length;
        buffers_[index].address = mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED,
                                       fd_, buffer.m.offset);
        Check(buffers_[index].address != MAP_FAILED, "camera mmap failed");
        Ioctl(VIDIOC_QBUF, &buffer, "VIDIOC_QBUF");
      }
      v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      Ioctl(VIDIOC_STREAMON, &type, "VIDIOC_STREAMON");
      streaming_ = true;
    } catch (...) {
      Close();
      throw;
    }
  }

  ~V4l2MjpegCapture() { Close(); }
  V4l2MjpegCapture(const V4l2MjpegCapture&) = delete;
  V4l2MjpegCapture& operator=(const V4l2MjpegCapture&) = delete;

  std::optional<FrameLease> Read() {
    for (int attempt = 0; attempt < 200; ++attempt) {
      v4l2_buffer buffer{};
      buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buffer.memory = V4L2_MEMORY_MMAP;
      if (ioctl(fd_, VIDIOC_DQBUF, &buffer) < 0) {
        if (errno == EAGAIN) {
          usleep(5000);
          continue;
        }
        throw Error(std::string("VIDIOC_DQBUF failed: ") + std::strerror(errno));
      }
      Check(buffer.index < buffers_.size(), "camera returned an invalid buffer index");
      Check(buffer.bytesused > 0 && buffer.bytesused <= buffers_[buffer.index].length,
            "camera returned an invalid MJPEG buffer length");
      const auto* data = static_cast<const uint8_t*>(buffers_[buffer.index].address);
      const uint32_t index = buffer.index;
      return FrameLease(data, buffer.bytesused, buffer.sequence,
                        [this, index] { Requeue(index); });
    }
    return std::nullopt;
  }

  int width() const { return width_; }
  int height() const { return height_; }

 private:
  void Requeue(uint32_t index) {
    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = index;
    Ioctl(VIDIOC_QBUF, &buffer, "VIDIOC_QBUF");
  }

  void Ioctl(unsigned long request, void* argument, const char* name) {
    if (ioctl(fd_, request, argument) < 0) {
      throw Error(std::string(name) + " failed: " + std::strerror(errno));
    }
  }

  void Close() noexcept {
    if (streaming_) {
      v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      (void)ioctl(fd_, VIDIOC_STREAMOFF, &type);
      streaming_ = false;
    }
    for (const V4l2Buffer& buffer : buffers_) {
      if (buffer.address != MAP_FAILED) munmap(buffer.address, buffer.length);
    }
    buffers_.clear();
    if (fd_ >= 0) close(fd_);
    fd_ = -1;
  }

  int fd_ = -1;
  int width_ = 0;
  int height_ = 0;
  bool streaming_ = false;
  std::vector<V4l2Buffer> buffers_;
};

class AclFrameBufferPool : public std::enable_shared_from_this<AclFrameBufferPool> {
 public:
  AclFrameBufferPool(aclrtContext context, std::size_t bytes, std::size_t count, bool device)
      : context_(context), bytes_(bytes), device_(device) {
    Check(count > 0, "camera FP16 output pool must not be empty");
    CheckAcl(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext camera output pool");
    buffers_.reserve(count);
    try {
      for (std::size_t index = 0; index < count; ++index) {
        void* buffer = nullptr;
        if (device_) {
          CheckAcl(aclrtMalloc(&buffer, bytes_, ACL_MEM_MALLOC_NORMAL_ONLY),
                   "aclrtMalloc camera NV12 output pool");
        } else {
          CheckAcl(aclrtMallocHost(&buffer, bytes_), "aclrtMallocHost camera FP16 output pool");
        }
        buffers_.push_back(buffer);
        free_.push_back(index);
      }
    } catch (...) {
      for (void* buffer : buffers_) {
        if (device_) (void)aclrtFree(buffer);
        else (void)aclrtFreeHost(buffer);
      }
      buffers_.clear();
      throw;
    }
  }

  ~AclFrameBufferPool() {
    (void)aclrtSetCurrentContext(context_);
    for (void* buffer : buffers_) {
      if (buffer != nullptr) {
        if (device_) (void)aclrtFree(buffer);
        else (void)aclrtFreeHost(buffer);
      }
    }
  }

  std::pair<void*, std::shared_ptr<void>> Acquire() {
    std::unique_lock<std::mutex> lock(mutex_);
    available_.wait(lock, [this] { return !free_.empty(); });
    const std::size_t index = free_.front();
    free_.pop_front();
    void* buffer = buffers_[index];
    lock.unlock();
    auto lease = std::shared_ptr<void>(buffer, [pool = shared_from_this(), index](void*) {
      pool->Release(index);
    });
    return {buffer, std::move(lease)};
  }

 private:
  void Release(std::size_t index) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      free_.push_back(index);
    }
    available_.notify_one();
  }

  aclrtContext context_ = nullptr;
  std::size_t bytes_ = 0;
  bool device_ = false;
  std::vector<void*> buffers_;
  std::deque<std::size_t> free_;
  std::mutex mutex_;
  std::condition_variable available_;
};

class CameraRtspPipeline {
 public:
  CameraRtspPipeline(aclrtContext context, const CameraCaptureOptions& options)
      : context_(context), width_(options.width), height_(options.height),
        queue_capacity_(options.rtsp_queue_capacity) {
    Check(!options.rtsp_url.empty(), "camera RTSP output URL must not be empty");
    Check(queue_capacity_ > 0, "camera RTSP queue capacity must be positive");
    publisher_ = std::make_unique<RtspVideoPublisher>(
        options.rtsp_url, options.fps, options.width, options.height, "ultrafast", 18,
        queue_capacity_, options.rtsp_transport, false);
    worker_ = std::thread([this] { WorkerMain(); });
  }

  CameraRtspPipeline(const CameraRtspPipeline&) = delete;
  CameraRtspPipeline& operator=(const CameraRtspPipeline&) = delete;

  ~CameraRtspPipeline() {
    try {
      Close();
    } catch (...) {
    }
  }

  void Enqueue(DvppJpegDecodedFrame frame) {
    Check(frame.device_data() != nullptr, "camera RTSP received an empty NV12 surface");
    std::lock_guard<std::mutex> lock(mutex_);
    if (worker_error_ != nullptr || stopping_ || closed_) return;
    if (queue_.size() >= queue_capacity_) {
      queue_.pop_front();
      ++dropped_frames_;
    }
    queue_.push_back(std::move(frame));
    cv_.notify_one();
  }

  void Close() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closed_) {
        if (final_error_ != nullptr) std::rethrow_exception(final_error_);
        return;
      }
      stopping_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    std::exception_ptr error;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      error = worker_error_;
    }
    try {
      publisher_->Close();
    } catch (...) {
      if (error == nullptr) error = std::current_exception();
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
      final_error_ = error;
    }
    if (error != nullptr) {
      try {
        std::rethrow_exception(error);
      } catch (const std::exception& exception) {
        std::cerr << "camera_rtsp_error=" << exception.what() << std::endl;
      } catch (...) {
        std::cerr << "camera_rtsp_error=unknown" << std::endl;
      }
    }
  }

  uint64_t frames() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return frames_;
  }
  uint64_t dropped_frames() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return dropped_frames_;
  }

 private:
  void WorkerMain() {
    try {
      CheckAcl(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext camera RTSP worker");
      for (;;) {
        DvppJpegDecodedFrame frame;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
          if (queue_.empty() && stopping_) break;
          frame = std::move(queue_.front());
          queue_.pop_front();
        }
        std::vector<uint8_t> padded_nv12(frame.bytes());
        CheckAcl(aclrtMemcpy(padded_nv12.data(), padded_nv12.size(), frame.device_data(),
                             frame.bytes(), ACL_MEMCPY_DEVICE_TO_HOST),
                 "copy camera NV12 to host for RTSP encoder");
        const std::size_t output_y_bytes = static_cast<std::size_t>(width_) * height_;
        std::vector<uint8_t> packed_nv12(output_y_bytes + output_y_bytes / 2);
        const std::size_t source_y_bytes =
            static_cast<std::size_t>(frame.width_stride()) * frame.height_stride();
        for (int row = 0; row < height_; ++row) {
          std::memcpy(packed_nv12.data() + static_cast<std::size_t>(row) * width_,
                      padded_nv12.data() + static_cast<std::size_t>(row) * frame.width_stride(),
                      width_);
        }
        uint8_t* output_uv = packed_nv12.data() + output_y_bytes;
        const uint8_t* source_uv = padded_nv12.data() + source_y_bytes;
        for (int row = 0; row < height_ / 2; ++row) {
          std::memcpy(output_uv + static_cast<std::size_t>(row) * width_,
                      source_uv + static_cast<std::size_t>(row) * frame.width_stride(), width_);
        }
        publisher_->WriteNv12Frame(std::move(packed_nv12));
        std::lock_guard<std::mutex> lock(mutex_);
        ++frames_;
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      worker_error_ = std::current_exception();
      stopping_ = true;
      queue_.clear();
      cv_.notify_all();
    }
  }

  aclrtContext context_ = nullptr;
  int width_ = 0;
  int height_ = 0;
  std::size_t queue_capacity_ = 3;
  std::unique_ptr<RtspVideoPublisher> publisher_;
  std::deque<DvppJpegDecodedFrame> queue_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::thread worker_;
  bool stopping_ = false;
  bool closed_ = false;
  std::exception_ptr worker_error_;
  std::exception_ptr final_error_;
  uint64_t frames_ = 0;
  uint64_t dropped_frames_ = 0;
};

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

void BgrToFp16Yuv444(const cv::Mat& frame, int width, int height, int padded_width,
                     int padded_height, uint16_t* output) {
  Check(frame.type() == CV_8UC3 && frame.cols == width && frame.rows == height,
        "decoded camera frame has unexpected format or dimensions");
  Check(padded_width >= width && padded_height >= height && output != nullptr,
        "camera frame exceeds model input tensor shape");
  const std::size_t channel_stride =
      static_cast<std::size_t>(padded_height) * static_cast<std::size_t>(padded_width);
  cv::parallel_for_(cv::Range(0, padded_height), [&](const cv::Range& rows) {
    for (int y = rows.start; y < rows.end; ++y) {
      const auto* row = frame.ptr<cv::Vec3b>(std::min(y, height - 1));
      for (int x = 0; x < padded_width; ++x) {
        const cv::Vec3b pixel = row[std::min(x, width - 1)];
        const float b = static_cast<float>(pixel[0]) / 255.0F;
        const float g = static_cast<float>(pixel[1]) / 255.0F;
        const float r = static_cast<float>(pixel[2]) / 255.0F;
        const float luma = kKr * r + kKg * g + kKb * b;
        const float cb = 0.5F * (b - luma) / (1.0F - kKb) + 0.5F;
        const float cr = 0.5F * (r - luma) / (1.0F - kKr) + 0.5F;
        const std::size_t offset = static_cast<std::size_t>(y) * padded_width + x;
        output[offset] = static_cast<uint16_t>(codec::FloatToHalfBits(luma));
        output[channel_stride + offset] = static_cast<uint16_t>(codec::FloatToHalfBits(cb));
        output[2 * channel_stride + offset] = static_cast<uint16_t>(codec::FloatToHalfBits(cr));
      }
    }
  });
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


}  // namespace

struct RtspVideoPublisher::DirectState {
  AVFormatContext* format = nullptr;
  AVStream* stream = nullptr;
  AVCodecContext* encoder = nullptr;
  AVRational input_time_base{1, 30};
  AVRational stream_time_base{1, 90000};
  int64_t next_frame_pts = 0;
  bool h264_copy = false;
  bool header_written = false;
  std::string transport = "udp";

  ~DirectState() {
    if (encoder != nullptr) avcodec_free_context(&encoder);
    if (format != nullptr) {
      if (!(format->oformat->flags & AVFMT_NOFILE) && format->pb != nullptr) {
        avio_closep(&format->pb);
      }
      avformat_free_context(format);
    }
  }
};

std::string AvErrorText(int error) {
  char buffer[AV_ERROR_MAX_STRING_SIZE] = {};
  av_strerror(error, buffer, sizeof(buffer));
  return std::string(buffer);
}

void CheckAv(int error, const std::string& operation) {
  if (error < 0) {
    Check(false, operation + " failed: " + AvErrorText(error));
  }
}

AVRational FpsRational(double fps) {
  Check(fps > 0.0, "RTSP FPS must be positive");
  AVRational result = av_d2q(1.0 / fps, 100000);
  Check(result.num > 0 && result.den > 0, "RTSP frame time base is invalid");
  return result;
}

std::vector<std::pair<const uint8_t*, std::size_t>> FindAnnexBNals(
    const std::vector<uint8_t>& frame) {
  std::vector<std::pair<const uint8_t*, std::size_t>> nals;
  auto start_code = [&frame](std::size_t offset) -> std::size_t {
    if (offset + 3 <= frame.size() && frame[offset] == 0 && frame[offset + 1] == 0 &&
        frame[offset + 2] == 1) {
      return 3;
    }
    if (offset + 4 <= frame.size() && frame[offset] == 0 && frame[offset + 1] == 0 &&
        frame[offset + 2] == 0 && frame[offset + 3] == 1) {
      return 4;
    }
    return 0;
  };
  std::size_t start = frame.size();
  std::size_t start_size = 0;
  for (std::size_t i = 0; i < frame.size();) {
    const std::size_t code_size = start_code(i);
    if (code_size == 0) {
      ++i;
      continue;
    }
    if (start != frame.size()) {
      const std::size_t nal_begin = start + start_size;
      if (i > nal_begin) nals.emplace_back(frame.data() + nal_begin, i - nal_begin);
    }
    start = i;
    start_size = code_size;
    i += code_size;
  }
  if (start != frame.size()) {
    const std::size_t nal_begin = start + start_size;
    if (frame.size() > nal_begin) {
      nals.emplace_back(frame.data() + nal_begin, frame.size() - nal_begin);
    }
  }
  const bool annexb_prefix = frame.size() >= 3 && frame[0] == 0 && frame[1] == 0 &&
                             (frame[2] == 1 || (frame.size() >= 4 && frame[2] == 0 && frame[3] == 1));
  if (annexb_prefix) return nals;
  nals.clear();
  // DVPP may provide length-prefixed AVCC rather than Annex-B NAL units.
  std::size_t offset = 0;
  while (offset + 4 <= frame.size()) {
    const uint32_t nal_size = (static_cast<uint32_t>(frame[offset]) << 24) |
                              (static_cast<uint32_t>(frame[offset + 1]) << 16) |
                              (static_cast<uint32_t>(frame[offset + 2]) << 8) |
                              static_cast<uint32_t>(frame[offset + 3]);
    offset += 4;
    if (nal_size == 0 || nal_size > frame.size() - offset) { nals.clear(); break; }
    const uint8_t nal_type = frame[offset] & 0x1f;
    if (nal_type == 0 || nal_type > 23) { nals.clear(); break; }
    nals.emplace_back(frame.data() + offset, nal_size);
    offset += nal_size;
  }
  if (offset != frame.size()) nals.clear();
  return nals;
}

std::vector<uint8_t> BuildAvccExtradata(const std::vector<uint8_t>& frame) {
  const auto nals = FindAnnexBNals(frame);
  const uint8_t* sps = nullptr;
  std::size_t sps_size = 0;
  const uint8_t* pps = nullptr;
  std::size_t pps_size = 0;
  for (const auto& nal : nals) {
    if (nal.second == 0) continue;
    const int type = nal.first[0] & 0x1f;
    if (type == 7 && sps == nullptr) {
      sps = nal.first;
      sps_size = nal.second;
    } else if (type == 8 && pps == nullptr) {
      pps = nal.first;
      pps_size = nal.second;
    }
  }
  if (sps == nullptr || pps == nullptr || sps_size > 0xffff || pps_size > 0xffff ||
      sps_size < 4) {
    return {};
  }
  std::vector<uint8_t> extradata(11 + sps_size + pps_size);
  extradata[0] = 1;
  extradata[1] = sps[1];
  extradata[2] = sps[2];
  extradata[3] = sps[3];
  extradata[4] = 0xff;
  extradata[5] = 0xe1;
  extradata[6] = static_cast<uint8_t>(sps_size >> 8);
  extradata[7] = static_cast<uint8_t>(sps_size);
  std::memcpy(extradata.data() + 8, sps, sps_size);
  const std::size_t pps_header = 8 + sps_size;
  extradata[pps_header] = 1;
  extradata[pps_header + 1] = static_cast<uint8_t>(pps_size >> 8);
  extradata[pps_header + 2] = static_cast<uint8_t>(pps_size);
  std::memcpy(extradata.data() + pps_header + 3, pps, pps_size);
  return extradata;
}

std::vector<uint8_t> ConvertAnnexBToAvcc(const std::vector<uint8_t>& frame) {
  const auto nals = FindAnnexBNals(frame);
  if (nals.empty()) return frame;
  std::size_t total_size = 0;
  for (const auto& nal : nals) {
    Check(nal.second <= UINT32_MAX, "RTSP H.264 NAL unit is too large");
    total_size += 4 + nal.second;
  }
  std::vector<uint8_t> converted;
  converted.reserve(total_size);
  for (const auto& nal : nals) {
    const uint32_t size = static_cast<uint32_t>(nal.second);
    converted.push_back(static_cast<uint8_t>(size >> 24));
    converted.push_back(static_cast<uint8_t>(size >> 16));
    converted.push_back(static_cast<uint8_t>(size >> 8));
    converted.push_back(static_cast<uint8_t>(size));
    converted.insert(converted.end(), nal.first, nal.first + nal.second);
  }
  return converted;
}

void SetDirectH264Extradata(AVStream* stream, const std::vector<uint8_t>& frame) {
  const std::vector<uint8_t> extradata = BuildAvccExtradata(frame);
  if (extradata.empty()) return;
  stream->codecpar->extradata = static_cast<uint8_t*>(
      av_mallocz(extradata.size() + AV_INPUT_BUFFER_PADDING_SIZE));
  Check(stream->codecpar->extradata != nullptr, "allocate RTSP H.264 extradata");
  std::memcpy(stream->codecpar->extradata, extradata.data(), extradata.size());
  stream->codecpar->extradata_size = static_cast<int>(extradata.size());
}

cv::Mat ConvertTensorToBgr(const codec::TensorData& tensor, int width, int height) {
  return TensorToBgrMat(tensor, width, height);
}

bool IsCameraDevicePath(const std::filesystem::path& path) {
  const std::string value = path.string();
  return value.rfind("/dev/video", 0) == 0;
}

class VideoFrameReader::Impl {
 public:
  Impl(const std::filesystem::path& input_path,
       const std::optional<CameraCaptureOptions>& camera_options, aclrtContext context) {
    if (camera_options.has_value()) {
      context_ = context;
      motion_nv12_ = camera_options->motion_nv12;
      if (motion_nv12_) {
        Check(camera_options->width > 0 && camera_options->height > 0 &&
                  camera_options->width % 2 == 0 && camera_options->height % 2 == 0,
              "camera motion NV12 requires positive even visible dimensions");
        (void)Nv12BufferSize({camera_options->width, camera_options->height,
                             camera_options->width, camera_options->height});
      }
      Check(IsCameraDevicePath(input_path),
            "camera input must be a V4L2 device path such as /dev/video0");
      camera_capture_ = std::make_unique<V4l2MjpegCapture>(input_path, *camera_options);
      info_width_ = camera_capture_->width();
      info_height_ = camera_capture_->height();
      Check(context != nullptr, "camera input requires an ACL context for DVPP JPEG decode");
      DvppJpegDecodeConfig decode_config;
      decode_config.width = camera_capture_->width();
      decode_config.height = camera_capture_->height();
      jpeg_decoder_ = std::make_unique<DvppJpegDecoder>(context, decode_config);
      if (!camera_options->rtsp_url.empty()) {
        rtsp_pipeline_ = std::make_unique<CameraRtspPipeline>(context, *camera_options);
      }
      Check(camera_options->width > 0 && camera_options->height > 0 &&
                camera_options->fps > 0.0 && std::isfinite(camera_options->fps),
            "camera dimensions and fps must be positive");
      if (camera_options->preload_frames != 0) {
        replay_frames_.reserve(camera_options->preload_frames);
        for (std::size_t i = 0; i < camera_options->preload_frames; ++i) {
          auto lease = camera_capture_->Read();
          Check(lease.has_value(), "camera ended while preloading benchmark frames");
          replay_frames_.emplace_back(lease->data(), lease->data() + lease->bytes());
          lease->Release();
        }
      }
      StartCameraPrefetch();
      return;
    } else {
      capture_.open(input_path.string());
    }
    Check(capture_.isOpened(), "failed to open input video: " + input_path.string());
  }

  ~Impl() {
    try {
      Close();
    } catch (...) {
    }
  }

  void Close() {
    if (closed_) return;
    {
      std::lock_guard<std::mutex> lock(camera_mutex_);
      camera_stop_ = true;
    }
    {
      std::lock_guard<std::mutex> lock(tensor_mutex_);
      tensor_stop_ = true;
    }
    camera_cv_.notify_all();
    tensor_cv_.notify_all();
    if (camera_thread_.joinable()) camera_thread_.join();
    if (tensor_thread_.joinable()) tensor_thread_.join();
    std::exception_ptr error;
    {
      std::lock_guard<std::mutex> lock(camera_mutex_);
      error = camera_error_;
      camera_queue_.clear();
    }
    {
      std::lock_guard<std::mutex> lock(tensor_mutex_);
      if (error == nullptr) error = tensor_error_;
      tensor_queue_.clear();
    }
    if (rtsp_pipeline_ != nullptr) {
      try {
        rtsp_pipeline_->Close();
      } catch (...) {
        if (error == nullptr) error = std::current_exception();
      }
    }
    if (camera_frames_ > 0) {
      std::cerr << "camera_pipeline frames=" << camera_frames_
                << " preload_frames=" << replay_frames_.size()
                << " capture_ms=" << capture_ms_ / camera_frames_
                << " jpegd_device_ms=" << jpegd_ms_ / camera_frames_
                << " nv12_d2h_ms=" << nv12_d2h_ms_ / std::max<uint64_t>(1, converted_frames_)
                << " cpu_nv12_bgr_ms=" << cpu_nv12_bgr_ms_ / std::max<uint64_t>(1, converted_frames_)
                << " cpu_bgr_fp16_ms=" << cpu_bgr_fp16_ms_ / std::max<uint64_t>(1, converted_frames_)
                << " aipp_nv12_pad_ms=" << aipp_pad_ms_ / std::max<uint64_t>(1, converted_frames_)
                << " v4l2_sequence_gaps=" << v4l2_sequence_gaps_
                << " jpeg_decode_errors=" << jpeg_decode_errors_
                << " camera_motion_copy_ms="
                << camera_motion_copy_ms_ / std::max<uint64_t>(1, camera_motion_copy_frames_)
                << " camera_motion_copy_total_ms=" << camera_motion_copy_ms_
                << " camera_motion_copy_frames=" << camera_motion_copy_frames_
                << " camera_queue_max_depth=" << camera_queue_max_depth_
                << " camera_queue_full_waits=" << camera_queue_full_waits_
                << " camera_queue_full_wait_total_ms=" << camera_queue_full_wait_total_ms_
                << " camera_queue_full_wait_max_ms=" << camera_queue_full_wait_max_ms_
                << " tensor_queue_max_depth=" << tensor_queue_max_depth_
                << " tensor_queue_full_waits=" << tensor_queue_full_waits_
                << " tensor_queue_full_wait_total_ms=" << tensor_queue_full_wait_total_ms_
                << " tensor_queue_full_wait_max_ms=" << tensor_queue_full_wait_max_ms_
                << " rtsp_frames=" << (rtsp_pipeline_ != nullptr ? rtsp_pipeline_->frames() : 0)
                << " rtsp_dropped="
                << (rtsp_pipeline_ != nullptr ? rtsp_pipeline_->dropped_frames() : 0)
                << std::endl;
    }
    closed_ = true;
    if (error != nullptr) std::rethrow_exception(error);
  }

  void StartCameraPrefetch() {
    camera_thread_ = std::thread([this] {
      try {
        for (;;) {
          {
            std::unique_lock<std::mutex> lock(camera_mutex_);
            const bool was_full = camera_queue_.size() >= kCameraQueueCapacity;
            const auto wait_begin = std::chrono::steady_clock::now();
            camera_cv_.wait(lock, [this] {
              return camera_stop_ || camera_queue_.size() < kCameraQueueCapacity;
            });
            if (was_full) {
              const double wait_ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - wait_begin).count();
              ++camera_queue_full_waits_;
              camera_queue_full_wait_total_ms_ += wait_ms;
              camera_queue_full_wait_max_ms_ =
                  std::max(camera_queue_full_wait_max_ms_, wait_ms);
            }
            if (camera_stop_) return;
          }
          const auto capture_begin = std::chrono::steady_clock::now();
          std::optional<V4l2MjpegCapture::FrameLease> jpeg;
          const std::vector<uint8_t>* replay = nullptr;
          if (!replay_frames_.empty()) {
            replay = &replay_frames_[replay_index_++ % replay_frames_.size()];
          } else {
            jpeg = camera_capture_->Read();
          }
          if (replay == nullptr && !jpeg.has_value()) {
            std::lock_guard<std::mutex> lock(camera_mutex_);
            camera_eof_ = true;
            camera_cv_.notify_all();
            return;
          }
          if (jpeg.has_value()) {
            const uint32_t sequence = jpeg->sequence();
            if (have_v4l2_sequence_) {
              const uint32_t delta = sequence - last_v4l2_sequence_;
              if (delta > 1 && delta < 10000) v4l2_sequence_gaps_ += delta - 1;
            }
            last_v4l2_sequence_ = sequence;
            have_v4l2_sequence_ = true;
          }
          const auto capture_end = std::chrono::steady_clock::now();
          const auto jpegd_begin = std::chrono::steady_clock::now();
          DvppJpegDecodedFrame nv12;
          try {
            nv12 = replay != nullptr
                       ? jpeg_decoder_->DecodeDevice(replay->data(), replay->size())
                       : jpeg_decoder_->DecodeDevice(jpeg->data(), jpeg->bytes());
          } catch (const std::exception& decode_error) {
            if (jpeg.has_value()) jpeg->Release();
            const std::string message = decode_error.what();
            if (message.find("acldvppJpegPredictDecSize") == std::string::npos) {
              throw;
            }
            {
              std::lock_guard<std::mutex> stats_lock(stats_mutex_);
              ++jpeg_decode_errors_;
              if (jpeg_decode_errors_ <= 5 || jpeg_decode_errors_ % 100 == 0) {
                std::cerr << "camera_jpeg_drop count=" << jpeg_decode_errors_
                          << " reason=" << message << std::endl;
              }
            }
            continue;
          }
          if (jpeg.has_value()) jpeg->Release();
          const auto jpegd_end = std::chrono::steady_clock::now();
          {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            capture_ms_ += std::chrono::duration<double, std::milli>(capture_end - capture_begin).count();
            jpegd_ms_ += std::chrono::duration<double, std::milli>(jpegd_end - jpegd_begin).count();
            ++camera_frames_;
            if (camera_frames_ % 900 == 0) {
              std::cerr << "camera_progress frames=" << camera_frames_
                        << " v4l2_sequence_gaps=" << v4l2_sequence_gaps_
                        << " camera_queue_full_waits=" << camera_queue_full_waits_
                        << std::endl;
            }
          }
          if (rtsp_pipeline_ != nullptr) rtsp_pipeline_->Enqueue(nv12);
          {
            std::lock_guard<std::mutex> lock(camera_mutex_);
            if (camera_stop_) return;
            camera_queue_.push_back(std::move(nv12));
            camera_queue_max_depth_ = std::max(camera_queue_max_depth_, camera_queue_.size());
          }
          camera_cv_.notify_all();
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(camera_mutex_);
        camera_error_ = std::current_exception();
        camera_stop_ = true;
        camera_cv_.notify_all();
      }
    });
  }

  DvppJpegDecodedFrame PopCameraNv12() {
    std::unique_lock<std::mutex> lock(camera_mutex_);
    camera_cv_.wait(lock, [this] {
      return camera_stop_ || camera_eof_ || camera_error_ != nullptr || !camera_queue_.empty();
    });
    if (camera_error_ != nullptr) std::rethrow_exception(camera_error_);
    if (camera_queue_.empty()) return {};
    DvppJpegDecodedFrame frame = std::move(camera_queue_.front());
    camera_queue_.pop_front();
    lock.unlock();
    camera_cv_.notify_all();
    return frame;
  }

  codec::TensorData ConvertCameraFrame(const DvppJpegDecodedFrame& nv12,
                                       const mlvc::TensorSpec& frame_spec,
                                       std::vector<uint8_t>* motion_nv12) {
    codec::TensorData result;
    result.shape = mlvc::TensorShape(frame_spec.shape);
    result.dtype = frame_spec.dtype;
    const std::size_t output_bytes = result.shape.NumElements() * ElementSize(result.dtype);
    CheckAcl(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext camera CPU conversion");
    auto [host_output, output_owner] = fp16_output_pool_->Acquire();
    if (frame_spec.dtype == DataType::kUInt8) {
      const std::size_t y_source_bytes =
          static_cast<std::size_t>(nv12.width_stride()) * nv12.height_stride();
      const std::size_t y_target_bytes =
          static_cast<std::size_t>(info_width_) * static_cast<std::size_t>(result.shape.dim(1) * 2 / 3);
      const std::size_t uv_bytes =
          static_cast<std::size_t>(info_width_) * static_cast<std::size_t>(info_height_ / 2);
      Check(nv12.width_stride() == info_width_ && y_target_bytes >= y_source_bytes &&
                output_bytes == y_target_bytes * 3 / 2,
            "AIPP camera NV12 tensor layout mismatch");
      const auto conversion_begin = std::chrono::steady_clock::now();
      CheckAcl(aclrtMemset(host_output, output_bytes, 128, output_bytes),
               "initialize padded camera NV12");
      CheckAcl(aclrtMemset(host_output, y_target_bytes, 16, y_target_bytes),
               "initialize padded camera luma");
      CheckAcl(aclrtMemcpy(host_output, static_cast<std::size_t>(info_width_) * info_height_,
                           nv12.device_data(), static_cast<std::size_t>(info_width_) * info_height_,
                           ACL_MEMCPY_DEVICE_TO_DEVICE), "copy camera luma for AIPP");
      CheckAcl(aclrtMemcpy(static_cast<uint8_t*>(host_output) + y_target_bytes, uv_bytes,
                           static_cast<const uint8_t*>(nv12.device_data()) + y_source_bytes,
                           uv_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE), "copy camera chroma for AIPP");
      const auto conversion_end = std::chrono::steady_clock::now();
      result.AttachExternalBuffer(host_output, output_bytes, MemoryLocation::kAcl,
                                  std::move(output_owner));
      {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        aipp_pad_ms_ += std::chrono::duration<double, std::milli>(conversion_end - conversion_begin).count();
        ++converted_frames_;
      }
      return result;
    }
    std::vector<uint8_t> host_nv12(nv12.bytes());
    const auto copy_begin = std::chrono::steady_clock::now();
    CheckAcl(aclrtMemcpy(host_nv12.data(), host_nv12.size(), nv12.device_data(),
                         nv12.bytes(), ACL_MEMCPY_DEVICE_TO_HOST), "copy camera NV12 to CPU");
    const auto copy_end = std::chrono::steady_clock::now();
    cv::Mat y_view(info_height_, info_width_, CV_8UC1, host_nv12.data(),
                   nv12.width_stride());
    cv::Mat uv_view(info_height_ / 2, info_width_ / 2, CV_8UC2,
                    host_nv12.data() + nv12.width_stride() * nv12.height_stride(),
                    nv12.width_stride());
    const auto bgr_begin = std::chrono::steady_clock::now();
    cv::Mat bgr;
    cv::cvtColorTwoPlane(y_view, uv_view, bgr, cv::COLOR_YUV2BGR_NV12);
    const auto bgr_end = std::chrono::steady_clock::now();
    BgrToFp16Yuv444(bgr, info_width_, info_height_,
                    static_cast<int>(result.shape.dim(3)),
                    static_cast<int>(result.shape.dim(2)),
                    static_cast<uint16_t*>(host_output));
    const auto conversion_end = std::chrono::steady_clock::now();
    result.AttachExternalBuffer(host_output, output_bytes, MemoryLocation::kPinnedCpu,
                                std::move(output_owner));
    {
      std::lock_guard<std::mutex> stats_lock(stats_mutex_);
      nv12_d2h_ms_ += std::chrono::duration<double, std::milli>(copy_end - copy_begin).count();
      cpu_nv12_bgr_ms_ += std::chrono::duration<double, std::milli>(bgr_end - bgr_begin).count();
      cpu_bgr_fp16_ms_ += std::chrono::duration<double, std::milli>(conversion_end - bgr_end).count();
      ++converted_frames_;
    }
    if (motion_nv12_) {
      Check(motion_nv12 != nullptr, "camera motion proxy output is missing");
      const auto motion_copy_begin = std::chrono::steady_clock::now();
      CopyVisibleNv12(host_nv12.data(), host_nv12.size(),
          {info_width_, info_height_, nv12.width_stride(), nv12.height_stride()}, motion_nv12);
      const double motion_copy_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - motion_copy_begin).count();
      std::lock_guard<std::mutex> stats_lock(stats_mutex_);
      camera_motion_copy_ms_ += motion_copy_ms;
      ++camera_motion_copy_frames_;
    }
    return result;
  }

  struct CameraTensor {
    codec::TensorData tensor;
    std::vector<uint8_t> motion_nv12;
  };

  void StartTensorPrefetch(const mlvc::TensorSpec& frame_spec) {
    Check(!motion_nv12_ || frame_spec.dtype == DataType::kFloat16,
          "camera motion NV12 sidecar requires the FP16 main input path");
    const std::size_t output_bytes =
        mlvc::TensorShape(frame_spec.shape).NumElements() * ElementSize(frame_spec.dtype);
    fp16_output_pool_ = std::make_shared<AclFrameBufferPool>(context_, output_bytes,
                                                             kFp16OutputPoolCapacity,
                                                             frame_spec.dtype == DataType::kUInt8);
    tensor_thread_ = std::thread([this, frame_spec] {
      try {
        for (;;) {
          DvppJpegDecodedFrame nv12 = PopCameraNv12();
          if (nv12.device_data() == nullptr) break;
          CameraTensor tensor;
          tensor.tensor = ConvertCameraFrame(nv12, frame_spec, &tensor.motion_nv12);
          std::unique_lock<std::mutex> lock(tensor_mutex_);
          const bool was_full = tensor_queue_.size() >= kTensorQueueCapacity;
          const auto wait_begin = std::chrono::steady_clock::now();
          tensor_cv_.wait(lock, [this] {
            return tensor_stop_ || tensor_queue_.size() < kTensorQueueCapacity;
          });
          if (was_full) {
            const double wait_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - wait_begin).count();
            ++tensor_queue_full_waits_;
            tensor_queue_full_wait_total_ms_ += wait_ms;
            tensor_queue_full_wait_max_ms_ = std::max(tensor_queue_full_wait_max_ms_, wait_ms);
          }
          if (tensor_stop_) break;
          tensor_queue_.push_back(std::move(tensor));
          tensor_queue_max_depth_ = std::max(tensor_queue_max_depth_, tensor_queue_.size());
          lock.unlock();
          tensor_cv_.notify_all();
        }
        std::lock_guard<std::mutex> lock(tensor_mutex_);
        tensor_eof_ = true;
        tensor_cv_.notify_all();
      } catch (...) {
        std::lock_guard<std::mutex> lock(tensor_mutex_);
        tensor_error_ = std::current_exception();
        tensor_eof_ = true;
        tensor_cv_.notify_all();
      }
    });
  }

  bool PopTensor(codec::TensorData* tensor, std::vector<uint8_t>* motion_nv12) {
    std::unique_lock<std::mutex> lock(tensor_mutex_);
    tensor_cv_.wait(lock, [this] {
      return tensor_eof_ || tensor_error_ != nullptr || !tensor_queue_.empty();
    });
    if (tensor_error_ != nullptr) std::rethrow_exception(tensor_error_);
    if (tensor_queue_.empty()) return false;
    *tensor = std::move(tensor_queue_.front().tensor);
    if (motion_nv12) *motion_nv12 = std::move(tensor_queue_.front().motion_nv12);
    tensor_queue_.pop_front();
    lock.unlock();
    tensor_cv_.notify_all();
    return true;
  }

  static constexpr std::size_t kCameraQueueCapacity = 3;
  static constexpr std::size_t kTensorQueueCapacity = 3;
  static constexpr std::size_t kFp16OutputPoolCapacity = 8;
  cv::VideoCapture capture_;
  std::unique_ptr<V4l2MjpegCapture> camera_capture_;
  std::unique_ptr<DvppJpegDecoder> jpeg_decoder_;
  std::vector<std::vector<uint8_t>> replay_frames_;
  std::size_t replay_index_ = 0;
  std::unique_ptr<CameraRtspPipeline> rtsp_pipeline_;
  std::shared_ptr<AclFrameBufferPool> fp16_output_pool_;
  aclrtContext context_ = nullptr;
  std::thread camera_thread_;
  std::mutex camera_mutex_;
  std::condition_variable camera_cv_;
  std::deque<DvppJpegDecodedFrame> camera_queue_;
  std::exception_ptr camera_error_;
  bool camera_stop_ = false;
  bool camera_eof_ = false;
  bool closed_ = false;
  int info_width_ = 0;
  int info_height_ = 0;
  std::thread tensor_thread_;
  std::mutex tensor_mutex_;
  std::condition_variable tensor_cv_;
  std::deque<CameraTensor> tensor_queue_;
  bool motion_nv12_ = false;
  double camera_motion_copy_ms_ = 0.0;
  uint64_t camera_motion_copy_frames_ = 0;
  std::exception_ptr tensor_error_;
  bool tensor_eof_ = false;
  bool tensor_started_ = false;
  bool tensor_stop_ = false;
  mutable std::mutex stats_mutex_;
  uint64_t camera_frames_ = 0;
  uint64_t v4l2_sequence_gaps_ = 0;
  uint64_t jpeg_decode_errors_ = 0;
  uint32_t last_v4l2_sequence_ = 0;
  bool have_v4l2_sequence_ = false;
  std::size_t camera_queue_max_depth_ = 0;
  uint64_t camera_queue_full_waits_ = 0;
  double camera_queue_full_wait_total_ms_ = 0.0;
  double camera_queue_full_wait_max_ms_ = 0.0;
  std::size_t tensor_queue_max_depth_ = 0;
  uint64_t tensor_queue_full_waits_ = 0;
  double tensor_queue_full_wait_total_ms_ = 0.0;
  double tensor_queue_full_wait_max_ms_ = 0.0;
  double capture_ms_ = 0.0;
  double jpegd_ms_ = 0.0;
  uint64_t converted_frames_ = 0;
  double nv12_d2h_ms_ = 0.0;
  double cpu_nv12_bgr_ms_ = 0.0;
  double cpu_bgr_fp16_ms_ = 0.0;
  double aipp_pad_ms_ = 0.0;
};

VideoFrameReader::VideoFrameReader(const std::filesystem::path& input_path)
    : impl_(std::make_unique<Impl>(input_path, std::nullopt, nullptr)) {
  info_.width = static_cast<int>(impl_->capture_.get(cv::CAP_PROP_FRAME_WIDTH));
  info_.height = static_cast<int>(impl_->capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
  info_.frame_count = static_cast<int>(impl_->capture_.get(cv::CAP_PROP_FRAME_COUNT));
  info_.fps = impl_->capture_.get(cv::CAP_PROP_FPS);
  if (info_.fps <= 0.0 || !std::isfinite(info_.fps)) info_.fps = 30.0;
  Check(info_.width > 0 && info_.height > 0, "failed to read video size from input stream");
}

VideoFrameReader::VideoFrameReader(const std::filesystem::path& input_path,
                                   const CameraCaptureOptions& camera_options,
                                   aclrtContext context)
    : impl_(std::make_unique<Impl>(input_path, camera_options, context)) {
  info_.width = impl_->camera_capture_->width();
  info_.height = impl_->camera_capture_->height();
  info_.frame_count = 0;
  info_.fps = camera_options.fps;
  Check(info_.width > 0 && info_.height > 0, "failed to read video size from input stream");
}

VideoFrameReader::~VideoFrameReader() = default;

void VideoFrameReader::Close() {
  if (impl_ != nullptr) impl_->Close();
}

bool VideoFrameReader::ReadFrameAsTensor(const mlvc::TensorSpec& frame_spec,
                                         codec::TensorData* tensor,
                                         std::vector<uint8_t>* motion_nv12) {
  if (motion_nv12) motion_nv12->clear();
  Check(tensor != nullptr, "video frame tensor output is required");
  Check(!impl_->motion_nv12_ || motion_nv12 != nullptr,
        "enabled camera motion sidecar requires an output vector");
  const bool aipp_camera = impl_->camera_capture_ != nullptr &&
                           frame_spec.dtype == DataType::kUInt8;
  Check(frame_spec.dtype == DataType::kFloat16 || aipp_camera,
        "video input tensor expects fp16 or camera AIPP uint8 frame spec");
  Check(frame_spec.shape.size() == 4 && frame_spec.shape[0] == 1 &&
            (aipp_camera ? frame_spec.shape[3] == 1 : frame_spec.shape[1] == 3),
        "video input tensor has unexpected shape");

  cv::Mat frame;
  if (impl_->camera_capture_ != nullptr) {
    if (!impl_->tensor_started_) {
      Check(aipp_camera || (frame_spec.shape[2] >= info_.height &&
                            frame_spec.shape[3] >= info_.width),
            "input video dimensions exceed the model frame tensor shape");
      impl_->StartTensorPrefetch(frame_spec);
      impl_->tensor_started_ = true;
    }
    codec::TensorData queued;
    if (!impl_->PopTensor(&queued, motion_nv12)) return false;
    *tensor = std::move(queued);
    return true;
  } else if (!impl_->capture_.read(frame)) {
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
                                       std::size_t queue_capacity, const std::string& transport,
                                       bool h264_copy)
    : width_(width), height_(height), queue_capacity_(queue_capacity), h264_copy_(h264_copy) {
  Check(queue_capacity_ > 0, "RTSP output queue capacity must be positive");
  direct_state_ = std::make_unique<DirectState>();
  direct_state_->h264_copy = h264_copy_;
  direct_state_->transport = transport;
  direct_state_->input_time_base = FpsRational(fps);
  direct_state_->stream_time_base = AVRational{1, 90000};
  CheckAv(avformat_alloc_output_context2(&direct_state_->format, nullptr, "rtsp", url.c_str()),
          "allocate RTSP output context");
  Check(direct_state_->format != nullptr, "allocate RTSP output context returned null");
  direct_state_->stream = avformat_new_stream(direct_state_->format, nullptr);
  Check(direct_state_->stream != nullptr, "create RTSP video stream");
  direct_state_->stream->time_base = direct_state_->stream_time_base;
  direct_state_->stream->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
  direct_state_->stream->codecpar->codec_id = AV_CODEC_ID_H264;
  direct_state_->stream->codecpar->width = width_;
  direct_state_->stream->codecpar->height = height_;
  direct_state_->stream->codecpar->format = AV_PIX_FMT_YUV420P;

  if (!h264_copy_) {
    const AVCodec* codec = avcodec_find_encoder_by_name("libx264");
    if (codec == nullptr) codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    Check(codec != nullptr, "find an H.264 encoder for direct RTSP publishing");
    direct_state_->encoder = avcodec_alloc_context3(codec);
    Check(direct_state_->encoder != nullptr, "allocate direct RTSP H.264 encoder");
    direct_state_->encoder->width = width_;
    direct_state_->encoder->height = height_;
    direct_state_->encoder->pix_fmt = AV_PIX_FMT_NV12;
    direct_state_->encoder->time_base = direct_state_->input_time_base;
    direct_state_->encoder->framerate = AVRational{direct_state_->input_time_base.den,
                                                    direct_state_->input_time_base.num};
    direct_state_->encoder->gop_size = std::max(1, static_cast<int>(std::lround(fps * 2.0)));
    direct_state_->encoder->max_b_frames = 0;
    // RTSP/AVC requires SPS/PPS in codecpar extradata (the SDP). Ask the
    // encoder to keep global headers separate from the packet payload.
    direct_state_->encoder->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    if (direct_state_->encoder->priv_data != nullptr) {
      av_opt_set(direct_state_->encoder->priv_data, "preset", preset.c_str(), 0);
      av_opt_set(direct_state_->encoder->priv_data, "tune", "zerolatency", 0);
      av_opt_set_int(direct_state_->encoder->priv_data, "crf", crf, 0);
    }
    CheckAv(avcodec_open2(direct_state_->encoder, codec, nullptr),
            "open direct RTSP H.264 encoder");
    CheckAv(avcodec_parameters_from_context(direct_state_->stream->codecpar,
                                             direct_state_->encoder),
            "copy direct RTSP encoder parameters");
    direct_state_->stream->time_base = direct_state_->stream_time_base;
    direct_state_->header_written = false;
  }

  if (!h264_copy_) {
    AVDictionary* options = nullptr;
    av_dict_set(&options, "rtsp_transport", transport.c_str(), 0);
    CheckAv(avformat_write_header(direct_state_->format, &options),
            "write RTSP stream header");
    av_dict_free(&options);
    direct_state_->header_written = true;
  } else {
    // Wait for the first H.264 access unit so SPS/PPS can be copied into the
    // SDP extradata before the RTSP header is sent.
    direct_state_->header_written = false;
  }
  std::cerr << "rtsp_publisher_backend=libavformat" << std::endl;
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
          cv_.notify_all();
        }
        if (!queued.h264.empty()) {

          Check(direct_state_ != nullptr, "direct RTSP publisher state is missing");
          if (!direct_state_->header_written) {
            // Do not send the RTSP header until an access unit containing SPS
            // and PPS is available. Some VENC firmware emits those records
            // one frame after the first IDR.
            const std::vector<uint8_t> extradata = BuildAvccExtradata(queued.h264);
            if (extradata.empty()) continue;
            direct_state_->stream->codecpar->extradata = static_cast<uint8_t*>(
                av_mallocz(extradata.size() + AV_INPUT_BUFFER_PADDING_SIZE));
            Check(direct_state_->stream->codecpar->extradata != nullptr, "allocate RTSP H.264 extradata");
            std::memcpy(direct_state_->stream->codecpar->extradata, extradata.data(), extradata.size());
            direct_state_->stream->codecpar->extradata_size = static_cast<int>(extradata.size());
            AVDictionary* options = nullptr;
            CheckAv(av_dict_set(&options, "rtsp_transport", direct_state_->transport.c_str(), 0),
                    "set direct RTSP transport");
            CheckAv(avformat_write_header(direct_state_->format, &options),
                    "write RTSP stream header");
            av_dict_free(&options);
            direct_state_->header_written = true;
          }
          const std::vector<uint8_t> avcc_payload = ConvertAnnexBToAvcc(queued.h264);
          AVPacket* packet = av_packet_alloc();
          Check(packet != nullptr, "allocate direct RTSP H.264 packet");
          CheckAv(av_new_packet(packet, static_cast<int>(avcc_payload.size())),
                  "allocate direct RTSP H.264 payload");
          std::memcpy(packet->data, avcc_payload.data(), avcc_payload.size());
          packet->stream_index = direct_state_->stream->index;
          packet->pts = packet->dts = av_rescale_q(
              direct_state_->next_frame_pts++, direct_state_->input_time_base,
              direct_state_->stream_time_base);
          packet->duration = av_rescale_q(1, direct_state_->input_time_base,
                                          direct_state_->stream_time_base);
          CheckAv(av_interleaved_write_frame(direct_state_->format, packet),
                  "write direct RTSP H.264 packet");
          av_packet_free(&packet);
        } else if (queued.nv12.empty()) {
          Check(direct_state_ != nullptr && direct_state_->encoder != nullptr,
                "direct RTSP encoder is missing");
          ConvertFp16Yuv444ToNv12(queued.tensor, layout, &reusable);
          AVFrame* frame = av_frame_alloc();
          Check(frame != nullptr, "allocate direct RTSP NV12 frame");
          frame->format = AV_PIX_FMT_NV12;
          frame->width = width_;
          frame->height = height_;
          frame->pts = direct_state_->next_frame_pts++;
          CheckAv(av_image_fill_arrays(frame->data, frame->linesize, reusable.data(),
                                       AV_PIX_FMT_NV12, width_, height_, 1),
                  "map direct RTSP NV12 frame");
          CheckAv(avcodec_send_frame(direct_state_->encoder, frame),
                  "send direct RTSP NV12 frame");
          av_frame_free(&frame);
          for (;;) {
            AVPacket* packet = av_packet_alloc();
            Check(packet != nullptr, "allocate direct RTSP encoded packet");
            const int receive = avcodec_receive_packet(direct_state_->encoder, packet);
            if (receive == AVERROR(EAGAIN) || receive == AVERROR_EOF) {
              av_packet_free(&packet);
              break;
            }
            CheckAv(receive, "receive direct RTSP encoded packet");
            av_packet_rescale_ts(packet, direct_state_->encoder->time_base,
                                 direct_state_->stream->time_base);
            if (packet->pts == AV_NOPTS_VALUE) packet->pts = direct_state_->next_frame_pts;
            if (packet->dts == AV_NOPTS_VALUE) packet->dts = packet->pts;
            if (packet->duration <= 0) {
              packet->duration = av_rescale_q(1, direct_state_->encoder->time_base,
                                              direct_state_->stream->time_base);
            }
            packet->stream_index = direct_state_->stream->index;
            // Keep libx264 Annex-B framing for FFmpeg RTSP/RTP packetization.
            CheckAv(av_interleaved_write_frame(direct_state_->format, packet),
                    "write direct RTSP encoded packet");
            if (direct_state_->format->pb != nullptr) avio_flush(direct_state_->format->pb);
            av_packet_free(&packet);
          }
        } else {
          Check(direct_state_ != nullptr && direct_state_->encoder != nullptr,
                "direct RTSP encoder is missing");
          AVFrame* frame = av_frame_alloc();
          Check(frame != nullptr, "allocate direct RTSP NV12 frame");
          frame->format = AV_PIX_FMT_NV12;
          frame->width = width_;
          frame->height = height_;
          frame->pts = direct_state_->next_frame_pts++;
          Check(queued.nv12.size() == static_cast<std::size_t>(width_) * height_ * 3 / 2,
                "direct RTSP NV12 frame has unexpected size");
          CheckAv(av_image_fill_arrays(frame->data, frame->linesize, queued.nv12.data(),
                                       AV_PIX_FMT_NV12, width_, height_, 1),
                  "map direct RTSP NV12 frame");
          CheckAv(avcodec_send_frame(direct_state_->encoder, frame),
                  "send direct RTSP NV12 frame");
          av_frame_free(&frame);
          for (;;) {
            AVPacket* packet = av_packet_alloc();
            Check(packet != nullptr, "allocate direct RTSP encoded packet");
            const int receive = avcodec_receive_packet(direct_state_->encoder, packet);
            if (receive == AVERROR(EAGAIN) || receive == AVERROR_EOF) {
              av_packet_free(&packet);
              break;
            }
            CheckAv(receive, "receive direct RTSP encoded packet");
            av_packet_rescale_ts(packet, direct_state_->encoder->time_base,
                                 direct_state_->stream->time_base);
            if (packet->pts == AV_NOPTS_VALUE) packet->pts = direct_state_->next_frame_pts;
            if (packet->dts == AV_NOPTS_VALUE) packet->dts = packet->pts;
            if (packet->duration <= 0) {
              packet->duration = av_rescale_q(1, direct_state_->encoder->time_base,
                                              direct_state_->stream->time_base);
            }
            packet->stream_index = direct_state_->stream->index;
            // Keep libx264 Annex-B framing for FFmpeg RTSP/RTP packetization.
            CheckAv(av_interleaved_write_frame(direct_state_->format, packet),
                    "write direct RTSP encoded packet");
            if (direct_state_->format->pb != nullptr) avio_flush(direct_state_->format->pb);
            av_packet_free(&packet);
          }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        ++frame_count_;
      }
    } catch (const std::exception& error) {
      std::cerr << "rtsp_worker_error=" << error.what() << std::endl;
      std::lock_guard<std::mutex> lock(mutex_);
      worker_error_ = std::current_exception();
      stopping_ = true;
      cv_.notify_all();
    } catch (...) {
      std::cerr << "rtsp_worker_error=unknown" << std::endl;
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
  std::unique_lock<std::mutex> lock(mutex_);
  Check(direct_state_ != nullptr && !closed_, "RTSP publisher is closed");
  if (worker_error_ != nullptr) std::rethrow_exception(worker_error_);
  cv_.wait(lock, [this] {
    return stopping_ || closed_ || worker_error_ != nullptr || queue_.size() < queue_capacity_;
  });
  if (worker_error_ != nullptr) std::rethrow_exception(worker_error_);
  Check(direct_state_ != nullptr && !closed_ && !stopping_, "RTSP publisher is closed");
  queue_.push_back(QueuedFrame{codec::CloneTensor(tensor), {}, {}});
  cv_.notify_one();
}

void RtspVideoPublisher::WriteNv12Frame(std::vector<uint8_t> frame) {
  Check(!frame.empty(), "RTSP NV12 frame must not be empty");
  std::unique_lock<std::mutex> lock(mutex_);
  Check(direct_state_ != nullptr && !closed_, "RTSP publisher is closed");
  if (worker_error_ != nullptr) std::rethrow_exception(worker_error_);
  cv_.wait(lock, [this] {
    return stopping_ || closed_ || worker_error_ != nullptr || queue_.size() < queue_capacity_;
  });
  if (worker_error_ != nullptr) std::rethrow_exception(worker_error_);
  Check(direct_state_ != nullptr && !closed_ && !stopping_, "RTSP publisher is closed");
  queue_.push_back(QueuedFrame{{}, std::move(frame), {}});
  cv_.notify_one();
}

void RtspVideoPublisher::WriteH264Frame(std::vector<uint8_t> frame) {
  Check(h264_copy_, "H.264 bitstream can only be written to a copy-mode RTSP publisher");
  Check(!frame.empty(), "H.264 output frame must not be empty");
  std::unique_lock<std::mutex> lock(mutex_);
  Check(direct_state_ != nullptr && !closed_, "RTSP publisher is closed");
  cv_.wait(lock, [this] {
    return stopping_ || closed_ || worker_error_ != nullptr || queue_.size() < queue_capacity_;
  });
  if (worker_error_ != nullptr) std::rethrow_exception(worker_error_);
  Check(direct_state_ != nullptr && !closed_ && !stopping_, "RTSP publisher is closed");
  queue_.push_back(QueuedFrame{{}, {}, std::move(frame)});
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
  if (direct_state_ != nullptr) {
    if (direct_state_->encoder != nullptr) {
      CheckAv(avcodec_send_frame(direct_state_->encoder, nullptr),
              "flush direct RTSP H.264 encoder");
      for (;;) {
        AVPacket* packet = av_packet_alloc();
        Check(packet != nullptr, "allocate direct RTSP flush packet");
        const int receive = avcodec_receive_packet(direct_state_->encoder, packet);
        if (receive == AVERROR(EAGAIN) || receive == AVERROR_EOF) {
          av_packet_free(&packet);
          break;
        }
        CheckAv(receive, "receive direct RTSP flush packet");
        av_packet_rescale_ts(packet, direct_state_->encoder->time_base,
                             direct_state_->stream->time_base);
        packet->stream_index = direct_state_->stream->index;
        CheckAv(av_interleaved_write_frame(direct_state_->format, packet),
                "write direct RTSP flush packet");
        av_packet_free(&packet);
      }
    }
    if (direct_state_->header_written) {
      CheckAv(av_write_trailer(direct_state_->format), "write RTSP trailer");
    }
    direct_state_.reset();
  }
  if (worker_error != nullptr) std::rethrow_exception(worker_error);
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
