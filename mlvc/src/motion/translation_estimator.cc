#include <mlvc/motion/translation_estimator.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/error.h>
#include <libavutil/motion_vector.h>
#include <libavutil/opt.h>
}

#include <cerrno>
#include <algorithm>
#include <climits>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "mlvc/codec/tensor_data.h"
#include "mlvc/io/fp16_yuv444_to_nv12.h"
#include "mlvc/motion/h264_reference_policy.h"

namespace mlvc::motion {
namespace {

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void CheckAv(int status, const char* operation) {
  if (status < 0) {
    char message[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(status, message, sizeof(message));
    throw std::runtime_error(std::string(operation) + ": " + message);
  }
}

struct CodecContext {
  AVCodecContext* value = nullptr;
  ~CodecContext() { avcodec_free_context(&value); }
};
struct Frame {
  AVFrame* value = av_frame_alloc();
  ~Frame() { av_frame_free(&value); }
};
struct Packet {
  AVPacket* value = av_packet_alloc();
  ~Packet() { av_packet_free(&value); }
};

}  // namespace

class H264MotionExtractor::Impl {
 public:
  explicit Impl(int gop, bool skip_loop_filter) : gop_(gop), skip_loop_filter_(skip_loop_filter) {
    Require(gop > 0, "motion GOP must be positive");
    const AVCodec* decoder = avcodec_find_decoder(AV_CODEC_ID_H264);
    Require(decoder != nullptr, "FFmpeg software H.264 decoder is unavailable");
    context_.value = avcodec_alloc_context3(decoder);
    Require(context_.value && frame_.value && packet_.value, "motion decoder allocation failed");
    context_.value->flags2 |= AV_CODEC_FLAG2_EXPORT_MVS;
    context_.value->thread_count = 1;
    context_.value->thread_type = 0;
    if (skip_loop_filter_) {
      context_.value->skip_loop_filter = AVDISCARD_ALL;
      // Concealment may derive MVs from changed pixels. Reject errors and
      // corrupt/concealed frames reported by FFmpeg; it cannot detect every
      // possible damaged bitstream.
      context_.value->err_recognition |= AV_EF_BITSTREAM | AV_EF_BUFFER | AV_EF_EXPLODE;
    }
    CheckAv(avcodec_open2(context_.value, decoder, nullptr), "open H.264 MV decoder");
  }

  Translation Decode(const uint8_t* data, std::size_t bytes, uint64_t frame_index,
                     bool random_access, std::vector<MotionVector>* raw_vectors) {
    Require(frame_index == next_frame_, "motion decoder requires consecutive frames from zero");
    Require(data != nullptr && bytes > 0 && bytes <= static_cast<std::size_t>(INT_MAX),
            "invalid H.264 proxy access unit");
    const bool gop_start = random_access || frame_index % static_cast<uint64_t>(gop_) == 0;
    reference_policy_.Inspect(data, bytes, gop_start);
    av_packet_unref(packet_.value);
    CheckAv(av_new_packet(packet_.value, static_cast<int>(bytes)), "allocate H.264 proxy packet");
    std::memcpy(packet_.value->data, data, bytes);
    packet_.value->pts = static_cast<int64_t>(frame_index);
    packet_.value->dts = static_cast<int64_t>(frame_index);
    CheckAv(avcodec_send_packet(context_.value, packet_.value), "submit H.264 MV packet");
    av_frame_unref(frame_.value);
    CheckAv(avcodec_receive_frame(context_.value, frame_.value), "receive H.264 MV frame");
    if (skip_loop_filter_) {
      Require(frame_.value->decode_error_flags == 0 &&
                  !(frame_.value->flags & AV_FRAME_FLAG_CORRUPT),
              "MV-only H.264 decode rejects FFmpeg-reported corrupt or concealed frames");
    }
    Require(frame_.value->pict_type == (gop_start ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_P),
            "proxy must produce fixed-GOP I/P frames; all-I DVPP restart mode is unsuitable");
    Require(context_.value->has_b_frames == 0, "proxy must not use B frames");
    const AVFrameSideData* side =
        av_frame_get_side_data(frame_.value, AV_FRAME_DATA_MOTION_VECTORS);
    std::vector<MotionVector> vectors;
    if (side != nullptr) {
      Require(side->size % sizeof(AVMotionVector) == 0, "invalid FFmpeg MV side data size");
      const auto* mvs = reinterpret_cast<const AVMotionVector*>(side->data);
      const std::size_t count = side->size / sizeof(AVMotionVector);
      vectors.reserve(count);
      for (std::size_t i = 0; i < count; ++i) {
        const auto& mv = mvs[i];
        vectors.push_back({mv.source, mv.w, mv.h, mv.motion_x, mv.motion_y, mv.motion_scale});
      }
    }
    Translation result = EstimateTranslation(vectors, gop_start);
    av_frame_unref(frame_.value);
    const int extra = avcodec_receive_frame(context_.value, frame_.value);
    Require(extra == AVERROR(EAGAIN), "H.264 proxy returned unexpected delayed or extra frames");
    ++next_frame_;
    if (raw_vectors) *raw_vectors = std::move(vectors);
    return result;
  }

 private:
  int gop_;
  bool skip_loop_filter_;
  uint64_t next_frame_ = 0;
  CodecContext context_;
  Frame frame_;
  Packet packet_;
  H264ReferencePolicy reference_policy_;
};

H264MotionExtractor::H264MotionExtractor(int gop, bool skip_loop_filter)
    : impl_(std::make_unique<Impl>(gop, skip_loop_filter)) {}
H264MotionExtractor::~H264MotionExtractor() = default;
Translation H264MotionExtractor::Decode(const uint8_t* packet, std::size_t bytes,
                                        uint64_t frame_index, bool random_access,
                                        std::vector<MotionVector>* raw_vectors) {
  return impl_->Decode(packet, bytes, frame_index, random_access, raw_vectors);
}

class TranslationEstimator::Impl {
 public:
  explicit Impl(TranslationEstimatorConfig config, TranslationEstimator::ProxyPacketObserver observer)
      : config_(config), extractor_(config.gop, config.skip_loop_filter),
        observer_(std::move(observer)) {
    Require(config.width > 0 && config.height > 0 && config.width % 2 == 0 &&
                config.height % 2 == 0 && config.fps > 0 && config.gop > 0,
            "invalid libx264 motion proxy dimensions, frame rate or GOP");
    Require(config.preset == "medium" || config.preset == "veryfast" ||
                config.preset == "superfast" || config.preset == "ultrafast",
            "motion libx264 preset must be medium, veryfast, superfast, or ultrafast");
    Require(config.threads >= 1 && config.threads <= 16,
            "motion libx264 threads must be in [1, 16]");
    const AVCodec* encoder = avcodec_find_encoder_by_name("libx264");
    Require(encoder != nullptr,
            "FFmpeg libx264 encoder is unavailable; install a libx264-enabled build");
    context_.value = avcodec_alloc_context3(encoder);
    Require(context_.value && frame_.value && packet_.value, "motion encoder allocation failed");
    auto* context = context_.value;
    context->width = config.width;
    context->height = config.height;
    context->pix_fmt = AV_PIX_FMT_YUV420P;
    context->time_base = {1, config.fps};
    context->framerate = {config.fps, 1};
    context->gop_size = config.gop;
    context->max_b_frames = 0;
    context->refs = 1;
    context->thread_count = config.threads;
    context->thread_type = FF_THREAD_SLICE;
    CheckAv(av_opt_set(context->priv_data, "preset", config.preset.c_str(), 0), "set libx264 preset");
    CheckAv(av_opt_set(context->priv_data, "tune", "zerolatency", 0), "set libx264 latency policy");
    CheckAv(av_opt_set(context->priv_data, "crf", "18", 0), "set libx264 CRF");
    CheckAv(av_opt_set(context->priv_data, "forced-idr", "1", 0), "set libx264 forced IDR policy");
    const std::string params = "keyint=" + std::to_string(config.gop) + ":min-keyint=" +
        std::to_string(config.gop) +
        ":scenecut=0:bframes=0:ref=1:rc-lookahead=0:sync-lookahead=0:mbtree=0:open-gop=0"
        ":sliced-threads=1:threads=" + std::to_string(config.threads);
    CheckAv(av_opt_set(context->priv_data, "x264-params", params.c_str(), 0),
            "set libx264 GOP policy");
    CheckAv(avcodec_open2(context, encoder, nullptr), "open libx264 motion proxy");
    frame_.value->format = context->pix_fmt;
    frame_.value->width = config.width;
    frame_.value->height = config.height;
    CheckAv(av_frame_get_buffer(frame_.value, 32), "allocate libx264 motion input");
  }

  Translation Estimate(const codec::TensorData& input, uint64_t frame_index, bool random_access) {
    Require(input.View().location() == MemoryLocation::kCpu ||
                input.View().location() == MemoryLocation::kPinnedCpu,
            "libx264 proxy requires a host input or EstimateNv12 with a host copy");
    Require(input.dtype == DataType::kFloat16 &&
                input.ByteSize() == input.shape.NumElements() * sizeof(uint16_t),
            "invalid FP16 motion proxy input size");
    const io::Nv12Layout layout{config_.width, config_.height, config_.width, config_.height};
    io::ConvertFp16Yuv444ToNv12(input, layout, &nv12_);
    return EstimateNv12(nv12_.data(), nv12_.size(), layout, frame_index, random_access);
  }

  Translation EstimateNv12(const uint8_t* data, std::size_t bytes,
                           const io::Nv12Layout& layout, uint64_t frame_index, bool random_access) {
    Require(frame_index == next_frame_, "motion encoder requires consecutive frames from zero");
    Require(data && layout.width == config_.width && layout.height == config_.height &&
                layout.width_stride >= layout.width && layout.height_stride >= layout.height &&
                layout.width_stride % 2 == 0 && layout.height_stride % 2 == 0 &&
                bytes == io::Nv12BufferSize(layout),
            "invalid host NV12 motion proxy input");
    Require(frame_index <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
            "motion frame index overflow");
    CheckAv(av_frame_make_writable(frame_.value), "prepare libx264 motion input");
    for (int y = 0; y < config_.height; ++y) {
      std::memcpy(frame_.value->data[0] + y * frame_.value->linesize[0],
                  data + static_cast<std::size_t>(y) * layout.width_stride, config_.width);
    }
    const std::size_t uv_base = static_cast<std::size_t>(layout.width_stride) * layout.height_stride;
    for (int y = 0; y < config_.height / 2; ++y) {
      const auto* uv = data + uv_base + static_cast<std::size_t>(y) * layout.width_stride;
      for (int x = 0; x < config_.width / 2; ++x) {
        frame_.value->data[1][y * frame_.value->linesize[1] + x] = uv[2 * x];
        frame_.value->data[2][y * frame_.value->linesize[2] + x] = uv[2 * x + 1];
      }
    }
    frame_.value->pts = static_cast<int64_t>(frame_index);
    frame_.value->pict_type = random_access || frame_index % static_cast<uint64_t>(config_.gop) == 0
                                ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_P;
    CheckAv(avcodec_send_frame(context_.value, frame_.value), "submit libx264 motion frame");
    av_packet_unref(packet_.value);
    CheckAv(avcodec_receive_packet(context_.value, packet_.value), "receive libx264 motion packet");
    Require(packet_.value->pts == static_cast<int64_t>(frame_index),
            "libx264 motion packet frame index mismatch");
    if (frame_index == 0) {
      // x264 can cap slice threads for short images. Its encoder-info SEI
      // records the effective threads/sliced_threads/slices, unlike AVCodecContext
      // which retains the requested count. Log that actual option text once.
      const std::string info(reinterpret_cast<const char*>(packet_.value->data),
                             static_cast<std::size_t>(packet_.value->size));
      const auto encoder_info = info.find("x264 - core ");
      const auto options = encoder_info == std::string::npos ? std::string::npos
                                                           : info.find("options: ", encoder_info);
      if (options != std::string::npos) {
        const auto end = info.find('\0', options);
        const auto length = std::min<std::size_t>(4096,
            (end == std::string::npos ? info.size() : end) - options);
        av_log(context_.value, AV_LOG_INFO,
               "motion_libx264 preset=%s requested_threads=%d actual_%.*s\n",
               config_.preset.c_str(), config_.threads, static_cast<int>(length),
               info.data() + options);
      } else {
        av_log(context_.value, AV_LOG_WARNING,
               "motion_libx264 preset=%s requested_threads=%d encoder_info_sei=absent\n",
               config_.preset.c_str(), config_.threads);
      }
    }
    if (observer_) observer_(packet_.value->data, packet_.value->size, frame_index, random_access);
    Translation result = extractor_.Decode(packet_.value->data, packet_.value->size, frame_index,
                                           random_access);
    av_packet_unref(packet_.value);
    const int extra = avcodec_receive_packet(context_.value, packet_.value);
    Require(extra == AVERROR(EAGAIN), "libx264 proxy returned unexpected extra packets");
    ++next_frame_;
    return result;
  }

 private:
  TranslationEstimatorConfig config_;
  uint64_t next_frame_ = 0;
  CodecContext context_;
  Frame frame_;
  Packet packet_;
  H264MotionExtractor extractor_;
  TranslationEstimator::ProxyPacketObserver observer_;
  std::vector<uint8_t> nv12_;
};

TranslationEstimator::TranslationEstimator(TranslationEstimatorConfig config,
                                            ProxyPacketObserver observer)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(observer))) {}
TranslationEstimator::~TranslationEstimator() = default;
Translation TranslationEstimator::Estimate(const codec::TensorData& input, uint64_t frame_index,
                                            bool random_access) {
  return impl_->Estimate(input, frame_index, random_access);
}
Translation TranslationEstimator::EstimateNv12(const uint8_t* data, std::size_t bytes,
                                               const io::Nv12Layout& layout, uint64_t frame_index,
                                               bool random_access) {
  return impl_->EstimateNv12(data, bytes, layout, frame_index, random_access);
}

}  // namespace mlvc::motion
