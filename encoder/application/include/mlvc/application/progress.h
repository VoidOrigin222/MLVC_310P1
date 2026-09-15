#ifndef MLVC_APPLICATION_PROGRESS_H_
#define MLVC_APPLICATION_PROGRESS_H_

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

namespace mlvc::app {

struct ProgressMetrics {
  int frames_done = 0;
  std::optional<int> total_frames;
  std::optional<double> fraction;
  double processing_fps = 0.0;
  double frame_bpp = 0.0;
  double average_bpp = 0.0;
  double source_fps_kbps = 0.0;
};

class ProgressReporter {
 public:
  explicit ProgressReporter(std::string label);
  ProgressReporter(const ProgressReporter&) = delete;
  ProgressReporter& operator=(const ProgressReporter&) = delete;
  ~ProgressReporter();

  void Update(const ProgressMetrics& metrics);
  void Finish(const ProgressMetrics& metrics);

 private:
  void Render(const ProgressMetrics& metrics, bool force);

  std::string label_;
  std::string last_line_;
  std::chrono::steady_clock::time_point last_render_;
  std::size_t last_line_size_ = 0;
  bool rendered_ = false;
};

double ProcessingFps(int frames, std::chrono::steady_clock::time_point start);
double FrameBpp(uint64_t frame_bytes, int width, int height);
double AverageBpp(uint64_t total_bytes, int frames, int width, int height);
double SourceFpsKbps(uint64_t total_bytes, int frames, double source_fps);

}  // namespace mlvc::app

#endif  // MLVC_APPLICATION_PROGRESS_H_
