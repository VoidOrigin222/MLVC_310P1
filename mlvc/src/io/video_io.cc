#include <mlvc/io/video_io.h>
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

#include "mlvc/core/status.h"
#include "mlvc/core/buffer.h"
#include "mlvc/runtime/acl_runtime.h"
#include "mlvc/io/fp16_yuv444_to_nv12.h"
#include "mlvc/io/dvpp_h264_encoder.h"

namespace mlvc::io {
namespace codec = mlvc::codec;
namespace {

void IgnoreSigpipeForSubprocessPipes() {
  static std::once_flag once;
  std::call_once(once, [] {
    Check(std::signal(SIGPIPE, SIG_IGN) != SIG_ERR,
          "failed to ignore SIGPIPE for RTSP subprocess pipe");
  });
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
      : queue_capacity_(options.rtsp_queue_capacity) {
    Check(!options.rtsp_url.empty(), "camera RTSP output URL must not be empty");
    Check(queue_capacity_ > 0, "camera RTSP queue capacity must be positive");
    const Nv12Layout layout{options.width, options.height, Align16(options.width),
                            Align16(options.height)};
    DvppH264EncoderConfig encoder_config;
    encoder_config.layout = layout;
    encoder_config.fps = static_cast<uint32_t>(std::lround(options.fps));
    encoder_config.gop = options.rtsp_gop;
    encoder_config.bitrate = options.rtsp_bitrate_bps;
    encoder_config.zero_copy_input = true;
    publisher_ = std::make_unique<RtspVideoPublisher>(
        options.rtsp_url, options.fps, options.width, options.height, "ultrafast", 0,
        queue_capacity_, options.rtsp_transport, true);
    encoder_ = std::make_unique<DvppH264Encoder>(context, encoder_config);
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
      for (;;) {
        DvppJpegDecodedFrame frame;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
          if (queue_.empty() && stopping_) break;
          frame = std::move(queue_.front());
          queue_.pop_front();
        }
        std::vector<uint8_t> h264 = encoder_->EncodeDevice(
            frame.device_data(), frame.bytes(), nullptr, false);
        publisher_->WriteH264Frame(std::move(h264));
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

  std::size_t queue_capacity_ = 3;
  std::unique_ptr<RtspVideoPublisher> publisher_;
  std::unique_ptr<DvppH264Encoder> encoder_;
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

std::string BuildRtspCommand(const std::string& url, double fps, int width, int height,
                             const std::string& preset, int crf, const std::string& transport,
                             bool h264_copy) {
  Check(!url.empty(), "RTSP output URL must not be empty");
  Check(fps > 0.0 && std::isfinite(fps), "RTSP FPS must be finite and positive");
  Check(width > 0 && height > 0, "RTSP dimensions must be positive");
  Check(!preset.empty(), "RTSP encoder preset must not be empty");
  Check(crf >= 0 && crf <= 51, "RTSP encoder CRF must be in [0, 51]");
  Check(transport == "tcp" || transport == "udp", "RTSP transport must be tcp or udp");
  std::ostringstream command;
  command << "ffmpeg -hide_banner -loglevel error";
  if (h264_copy) {
    command << " -r " << fps << " -f h264 -i pipe:0 -an -c:v copy";
  } else {
    command << " -f rawvideo -pix_fmt nv12 -s " << width << "x" << height << " -r " << fps
            << " -i pipe:0 -an -c:v libx264 -preset " << preset << " -tune zerolatency"
            << " -crf " << crf << " -pix_fmt yuv420p";
  }
  command << " -f rtsp -rtsp_transport " << transport << " "
          << ShellQuote(std::filesystem::path(url));
  return command.str();
}

}  // namespace

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
                                       const mlvc::TensorSpec& frame_spec) {
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
    return result;
  }

  void StartTensorPrefetch(const mlvc::TensorSpec& frame_spec) {
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
          codec::TensorData tensor = ConvertCameraFrame(nv12, frame_spec);
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

  bool PopTensor(codec::TensorData* tensor) {
    std::unique_lock<std::mutex> lock(tensor_mutex_);
    tensor_cv_.wait(lock, [this] {
      return tensor_eof_ || tensor_error_ != nullptr || !tensor_queue_.empty();
    });
    if (tensor_error_ != nullptr) std::rethrow_exception(tensor_error_);
    if (tensor_queue_.empty()) return false;
    *tensor = std::move(tensor_queue_.front());
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
  std::deque<codec::TensorData> tensor_queue_;
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
                                         codec::TensorData* tensor) {
  Check(tensor != nullptr, "video frame tensor output is required");
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
    if (!impl_->PopTensor(&queued)) return false;
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
  // A dead FFmpeg child must surface as a pipe write error, not SIGPIPE that
  // terminates unrelated MLVC encoding work in the same process.
  IgnoreSigpipeForSubprocessPipes();
  const std::string command =
      BuildRtspCommand(url, fps, width_, height_, preset, crf, transport, h264_copy_);
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
          cv_.notify_all();
        }
        if (!queued.h264.empty()) {
          const std::size_t bytes = queued.h264.size();
          Check(fwrite(queued.h264.data(), 1, bytes, pipe_) == bytes,
                "failed to write H.264 stream to FFmpeg RTSP publisher");
        } else if (queued.nv12.empty()) {
          ConvertFp16Yuv444ToNv12(queued.tensor, layout, &reusable);
          const std::size_t bytes = reusable.size();
          Check(fwrite(reusable.data(), 1, bytes, pipe_) == bytes,
                "failed to write decoded frame to FFmpeg RTSP publisher");
        } else {
          reusable = std::move(queued.nv12);
          const std::size_t bytes = reusable.size();
          Check(fwrite(reusable.data(), 1, bytes, pipe_) == bytes,
                "failed to write decoded frame to FFmpeg RTSP publisher");
        }
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
  std::unique_lock<std::mutex> lock(mutex_);
  Check(pipe_ != nullptr && !closed_, "RTSP publisher is closed");
  if (worker_error_ != nullptr) std::rethrow_exception(worker_error_);
  cv_.wait(lock, [this] {
    return stopping_ || closed_ || worker_error_ != nullptr || queue_.size() < queue_capacity_;
  });
  if (worker_error_ != nullptr) std::rethrow_exception(worker_error_);
  Check(pipe_ != nullptr && !closed_ && !stopping_, "RTSP publisher is closed");
  queue_.push_back(QueuedFrame{codec::CloneTensor(tensor), {}, {}});
  cv_.notify_one();
}

void RtspVideoPublisher::WriteNv12Frame(std::vector<uint8_t> frame) {
  Check(!frame.empty(), "RTSP NV12 frame must not be empty");
  std::unique_lock<std::mutex> lock(mutex_);
  Check(pipe_ != nullptr && !closed_, "RTSP publisher is closed");
  if (worker_error_ != nullptr) std::rethrow_exception(worker_error_);
  cv_.wait(lock, [this] {
    return stopping_ || closed_ || worker_error_ != nullptr || queue_.size() < queue_capacity_;
  });
  if (worker_error_ != nullptr) std::rethrow_exception(worker_error_);
  Check(pipe_ != nullptr && !closed_ && !stopping_, "RTSP publisher is closed");
  queue_.push_back(QueuedFrame{{}, std::move(frame), {}});
  cv_.notify_one();
}

void RtspVideoPublisher::WriteH264Frame(std::vector<uint8_t> frame) {
  Check(h264_copy_, "H.264 bitstream can only be written to a copy-mode RTSP publisher");
  Check(!frame.empty(), "H.264 output frame must not be empty");
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] {
    return stopping_ || closed_ || worker_error_ != nullptr || queue_.size() < queue_capacity_;
  });
  if (worker_error_ != nullptr) std::rethrow_exception(worker_error_);
  Check(pipe_ != nullptr && !closed_ && !stopping_, "RTSP publisher is closed");
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
