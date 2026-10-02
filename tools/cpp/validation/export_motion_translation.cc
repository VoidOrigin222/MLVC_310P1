// Offline proxy matching the Python ffmpeg medium/CRF18/GOP/no-B/single-ref
// experiment. Keeps source YUV pixels and x264 lookahead defaults. Real-time
// TranslationEstimator intentionally uses a different low-latency policy.
#include <mlvc/motion/translation_estimator.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/opt.h>
}

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void CheckAv(int status, const char* operation) {
  if (status < 0) {
    char error[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(status, error, sizeof(error));
    throw std::runtime_error(std::string(operation) + ": " + error);
  }
}
void Check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
struct Input {
  AVFormatContext* value = nullptr;
  ~Input() { avformat_close_input(&value); }
};
struct Context {
  AVCodecContext* value = nullptr;
  ~Context() { avcodec_free_context(&value); }
};
struct Frame {
  AVFrame* value = av_frame_alloc();
  ~Frame() { av_frame_free(&value); }
};
struct Packet {
  AVPacket* value = av_packet_alloc();
  ~Packet() { av_packet_free(&value); }
};

std::vector<mlvc::motion::Translation> Export(const char* path, int limit, int gop) {
  Input input;
  CheckAv(avformat_open_input(&input.value, path, nullptr, nullptr), "open input video");
  CheckAv(avformat_find_stream_info(input.value, nullptr), "read input video streams");
  const int stream_index = av_find_best_stream(input.value, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  CheckAv(stream_index, "find video stream");
  const AVStream* stream = input.value->streams[stream_index];
  const AVCodec* decoder = avcodec_find_decoder(stream->codecpar->codec_id);
  Check(decoder, "source decoder is unavailable");
  Context source;
  source.value = avcodec_alloc_context3(decoder);
  Check(source.value, "source decoder allocation failed");
  CheckAv(avcodec_parameters_to_context(source.value, stream->codecpar), "configure source decoder");
  CheckAv(avcodec_open2(source.value, decoder, nullptr), "open source decoder");
  const AVCodec* encoder = avcodec_find_encoder_by_name("libx264");
  Check(encoder, "FFmpeg libx264 encoder is unavailable");
  Context proxy;
  Frame frame;
  Packet packet;
  Packet encoded;
  Check(frame.value && packet.value && encoded.value, "proxy input allocation failed");
  mlvc::motion::H264MotionExtractor extractor(gop);
  std::vector<mlvc::motion::Translation> shifts;
  int submitted = 0;
  auto drain_proxy = [&] {
    while (true) {
      av_packet_unref(encoded.value);
      const int status = avcodec_receive_packet(proxy.value, encoded.value);
      if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) return;
      CheckAv(status, "receive offline libx264 packet");
      Check(encoded.value->pts == static_cast<int64_t>(shifts.size()),
            "offline proxy packet frame order mismatch");
      shifts.push_back(extractor.Decode(encoded.value->data, encoded.value->size, shifts.size()));
    }
  };
  auto drain_source = [&] {
    while (submitted < limit) {
      av_frame_unref(frame.value);
      const int status = avcodec_receive_frame(source.value, frame.value);
      if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) return;
      CheckAv(status, "receive source video frame");
      if (proxy.value == nullptr) {
        bool supported = false;
        for (const AVPixelFormat* format = encoder->pix_fmts;
             format && *format != AV_PIX_FMT_NONE; ++format) {
          if (*format == frame.value->format) supported = true;
        }
        Check(supported, "libx264 does not support the source pixel format; explicit conversion is needed");
        proxy.value = avcodec_alloc_context3(encoder);
        Check(proxy.value, "proxy encoder allocation failed");
        proxy.value->width = frame.value->width;
        proxy.value->height = frame.value->height;
        proxy.value->pix_fmt = static_cast<AVPixelFormat>(frame.value->format);
        const AVRational fps = av_guess_frame_rate(input.value, input.value->streams[stream_index], nullptr);
        Check(fps.num > 0 && fps.den > 0, "input frame rate is unavailable");
        proxy.value->time_base = av_inv_q(fps);
        proxy.value->framerate = fps;
        proxy.value->sample_aspect_ratio = frame.value->sample_aspect_ratio;
        proxy.value->color_range = frame.value->color_range;
        proxy.value->color_primaries = frame.value->color_primaries;
        proxy.value->color_trc = frame.value->color_trc;
        proxy.value->colorspace = frame.value->colorspace;
        proxy.value->chroma_sample_location = frame.value->chroma_location;
        proxy.value->gop_size = gop;
        proxy.value->keyint_min = gop;
        proxy.value->max_b_frames = 0;
        proxy.value->refs = 1;
        CheckAv(av_opt_set(proxy.value->priv_data, "preset", "medium", 0), "set offline preset");
        CheckAv(av_opt_set(proxy.value->priv_data, "crf", "18", 0), "set offline CRF");
        CheckAv(av_opt_set(proxy.value->priv_data, "sc_threshold", "0", 0), "set offline scene cut");
        CheckAv(avcodec_open2(proxy.value, encoder, nullptr), "open offline libx264 proxy");
      }
      Check(frame.value->width == proxy.value->width && frame.value->height == proxy.value->height &&
                frame.value->format == proxy.value->pix_fmt, "source format changed during proxy export");
      frame.value->pts = submitted;
      frame.value->pict_type = AV_PICTURE_TYPE_NONE;
      CheckAv(avcodec_send_frame(proxy.value, frame.value), "submit offline proxy frame");
      ++submitted;
      drain_proxy();
    }
  };
  while (submitted < limit) {
    av_packet_unref(packet.value);
    const int status = av_read_frame(input.value, packet.value);
    if (status == AVERROR_EOF) break;
    CheckAv(status, "read source video packet");
    if (packet.value->stream_index != stream_index) continue;
    CheckAv(avcodec_send_packet(source.value, packet.value), "submit source packet");
    drain_source();
  }
  if (submitted < limit) {
    CheckAv(avcodec_send_packet(source.value, nullptr), "flush source decoder");
    drain_source();
  }
  Check(submitted == limit && proxy.value, "source video has fewer frames than requested");
  CheckAv(avcodec_send_frame(proxy.value, nullptr), "flush offline proxy");
  drain_proxy();
  Check(shifts.size() == static_cast<std::size_t>(limit), "offline motion frame count mismatch");
  return shifts;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5 && argc != 6) {
    std::cerr << "usage: export_motion_translation input.mp4 frames output.json output.shifts [gop=96]\n";
    return 2;
  }
  try {
    const int frames = std::stoi(argv[2]);
    const int gop = argc == 6 ? std::stoi(argv[5]) : 96;
    Check(frames > 0 && gop > 0, "frame count and GOP must be positive");
    const auto shifts = Export(argv[1], frames, gop);
    std::ofstream json(argv[3]);
    std::ofstream sidecar(argv[4], std::ios::binary);
    Check(json.is_open() && sidecar.is_open(), "cannot create motion export files");
    json << "{\n  \"frames\": " << frames << ",\n  \"gop\": " << gop
         << ",\n  \"proxy\": {\"codec\": \"libx264\", \"preset\": \"medium\", \"crf\": 18,"
         << " \"bframes\": 0, \"refs\": 1, \"lookahead\": \"default\"},\n  \"motion\": [\n";
    for (std::size_t index = 0; index < shifts.size(); ++index) {
      const auto& s = shifts[index];
      json << "    {\"frame\": " << index << ", \"tx\": " << std::setprecision(12) << s.tx
           << ", \"ty\": " << s.ty << ", \"kx\": " << static_cast<int>(s.kx)
           << ", \"ky\": " << static_cast<int>(s.ky) << ", \"vectors\": " << s.vector_count
           << "}" << (index + 1 == shifts.size() ? "\n" : ",\n");
      if (index % static_cast<std::size_t>(gop) != 0) {
        const char shift[2] = {static_cast<char>(s.kx), static_cast<char>(s.ky)};
        sidecar.write(shift, sizeof(shift));
      }
    }
    json << "  ]\n}\n";
    Check(json.good() && sidecar.good(), "writing motion export failed");
    std::cout << "motion_export frames=" << shifts.size() << " sidecar_bytes="
              << 2 * (shifts.size() - (shifts.size() + gop - 1) / gop) << " status=ok\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "motion_export status=failed: " << error.what() << '\n';
    return 1;
  }
}
