#include <mlvc/application/progress.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace mlvc::app {
namespace {

constexpr int kBarWidth = 28;
constexpr auto kProgressUpdateInterval = std::chrono::milliseconds(250);

double ClampFraction(double value) {
  if (value < 0.0) {
    return 0.0;
  }
  if (value > 1.0) {
    return 1.0;
  }
  return value;
}

std::optional<double> EffectiveFraction(const ProgressMetrics& metrics) {
  if (metrics.fraction.has_value()) {
    return ClampFraction(*metrics.fraction);
  }
  if (metrics.total_frames.has_value() && *metrics.total_frames > 0) {
    return ClampFraction(static_cast<double>(metrics.frames_done) /
                         static_cast<double>(*metrics.total_frames));
  }
  return std::nullopt;
}

std::string BuildBar(std::optional<double> fraction) {
  std::ostringstream out;
  out << '[';
  if (fraction.has_value()) {
    const int filled = static_cast<int>(*fraction * kBarWidth + 0.5);
    for (int i = 0; i < kBarWidth; ++i) {
      out << (i < filled ? '#' : '-');
    }
  } else {
    for (int i = 0; i < kBarWidth; ++i) {
      out << (i % 4 == 0 ? '#' : '-');
    }
  }
  out << ']';
  return out.str();
}

}  // namespace

ProgressReporter::ProgressReporter(std::string label)
    : label_(std::move(label)),
      last_render_(std::chrono::steady_clock::now() - kProgressUpdateInterval) {}

ProgressReporter::~ProgressReporter() {
  if (rendered_) {
    std::cout << '\n';
  }
}

void ProgressReporter::Update(const ProgressMetrics& metrics) { Render(metrics, false); }

void ProgressReporter::Finish(const ProgressMetrics& metrics) {
  Render(metrics, true);
  if (rendered_) {
    std::cout << '\n';
    rendered_ = false;
    last_line_size_ = 0;
  }
}

void ProgressReporter::Render(const ProgressMetrics& metrics, bool force) {
  const auto now = std::chrono::steady_clock::now();
  if (!force && rendered_ && now - last_render_ < kProgressUpdateInterval) {
    return;
  }
  last_render_ = now;

  const std::optional<double> fraction = EffectiveFraction(metrics);
  std::ostringstream line;
  line << label_ << ' ' << BuildBar(fraction) << ' ';
  if (fraction.has_value()) {
    line << std::fixed << std::setprecision(1) << (*fraction * 100.0) << "% ";
  }
  line << metrics.frames_done;
  if (metrics.total_frames.has_value()) {
    line << '/' << *metrics.total_frames;
  }
  line << " frames";
  line << " fps=" << std::fixed << std::setprecision(2) << metrics.processing_fps;
  line << " frame_bpp=" << std::fixed << std::setprecision(4) << metrics.frame_bpp;
  line << " avg_bpp=" << std::fixed << std::setprecision(4) << metrics.average_bpp;
  line << " kbps=" << std::fixed << std::setprecision(2) << metrics.source_fps_kbps;

  const std::string rendered_line = line.str();
  if (force && rendered_ && rendered_line == last_line_) {
    return;
  }
  std::cout << '\r' << rendered_line;
  if (rendered_line.size() < last_line_size_) {
    std::cout << std::string(last_line_size_ - rendered_line.size(), ' ');
  }
  std::cout << std::flush;
  last_line_size_ = rendered_line.size();
  last_line_ = rendered_line;
  rendered_ = true;
}

double ProcessingFps(int frames, std::chrono::steady_clock::time_point start) {
  const auto now = std::chrono::steady_clock::now();
  const double seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(now - start).count();
  return seconds > 0.0 ? static_cast<double>(frames) / seconds : 0.0;
}

double FrameBpp(uint64_t frame_bytes, int width, int height) {
  const uint64_t pixels = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
  return pixels > 0 ? static_cast<double>(frame_bytes * 8) / static_cast<double>(pixels) : 0.0;
}

double AverageBpp(uint64_t total_bytes, int frames, int width, int height) {
  if (frames <= 0) {
    return 0.0;
  }
  const uint64_t pixels = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
  return pixels > 0 ? static_cast<double>(total_bytes * 8) /
                          static_cast<double>(pixels * static_cast<uint64_t>(frames))
                    : 0.0;
}

double SourceFpsKbps(uint64_t total_bytes, int frames, double source_fps) {
  if (frames <= 0 || source_fps <= 0.0) {
    return 0.0;
  }
  const double bits_per_frame = static_cast<double>(total_bytes * 8) / static_cast<double>(frames);
  return bits_per_frame * source_fps / 1000.0;
}

}  // namespace mlvc::app
