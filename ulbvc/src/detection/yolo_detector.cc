#include "ulbvc/detection/yolo_detector.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <utility>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#include "ulbvc/codec/tensor_data.h"
#include "ulbvc/core/status.h"
#include "ulbvc/detection/geometry.h"
#include "ulbvc/detection/nv12_preprocess.h"
#include "ulbvc/runtime/acl_runtime.h"

namespace ulbvc::detection {
namespace {
using Clock = std::chrono::steady_clock;
double Milliseconds(Clock::time_point first, Clock::time_point last) {
  return std::chrono::duration<double, std::milli>(last - first).count();
}
float IntersectionOverUnion(const transport::DetectionBox& a, const transport::DetectionBox& b) {
  const float width = std::max(0.0f, std::min(a.cx + a.width / 2, b.cx + b.width / 2) -
                                         std::max(a.cx - a.width / 2, b.cx - b.width / 2));
  const float height = std::max(0.0f, std::min(a.cy + a.height / 2, b.cy + b.height / 2) -
                                          std::max(a.cy - a.height / 2, b.cy - b.height / 2));
  const float intersection = width * height;
  return intersection / std::max(a.width * a.height + b.width * b.height - intersection, 1e-6f);
}
}  // namespace

class YoloWorker::Impl {
 public:
  Impl(aclrtContext context, Options options, int width, int height, Profiler* profiler)
      : options_(std::move(options)),
        width_(width),
        height_(height),
        runtime_(context),
        stage_(&runtime_, ModelRecord{}, options_.model, true),
        profiler_(profiler) {
    Check(options_.queue_capacity > 0 && std::isfinite(options_.confidence) &&
              options_.confidence > 0 && options_.confidence <= 1 &&
              std::isfinite(options_.nms_iou) && options_.nms_iou > 0 && options_.nms_iou <= 1,
          "invalid detection thresholds or queue capacity");
    Check(width > 0 && height > 0 && width <= 65535 && height <= 65535 && width % 2 == 0 &&
              height % 2 == 0,
          "invalid detection source dimensions");
    const auto& record = stage_.record();
    Check(record.inputs.size() == 1 && record.outputs.size() == 1,
          "YOLO requires one input and one raw output");
    const auto& input = record.inputs[0];
    const auto& output = record.outputs[0];
    Check(input.dtype == DataType::kFloat16 && input.shape.size() == 4 && input.shape[0] == 1 &&
              input.shape[1] == 3,
          "YOLO requires static NCHW FP16 input");
    Check(output.dtype == DataType::kFloat16 && output.shape.size() == 3 && output.shape[0] == 1 &&
              output.shape[1] == 84,
          "YOLO requires raw FP16 [1,84,candidates] output");
    input_width_ = input.shape[3];
    input_height_ = input.shape[2];
    Check(
        input_width_ > 0 && input_height_ > 0 && input_width_ % 32 == 0 && input_height_ % 32 == 0,
        "YOLO input dimensions must be stride-32 aligned");
    plan_ = PlanLetterbox(width_, height_, input_width_, input_height_);
    resized_width_ = plan_.resized_width;
    resized_height_ = plan_.resized_height;
    left_ = plan_.left;
    top_ = plan_.top;
    std::ifstream labels(options_.labels);
    Check(labels.good(), "cannot open detection labels: " + options_.labels.string());
    for (std::string label; std::getline(labels, label);) {
      if (!label.empty() && label.back() == '\r') label.pop_back();
      Check(!label.empty() && label.size() <= 63 &&
                std::none_of(label.begin(), label.end(), [](unsigned char ch) { return ch < 32; }),
            "invalid detection label");
      labels_.push_back(std::move(label));
    }
    Check(labels_.size() == 80, "YOLO11s requires 80 class labels");
    for (auto& tensor : inputs_) tensor = codec::MakeTensorLike(input);
    for (int slot = 0; slot < static_cast<int>(inputs_.size()); ++slot) free_slots_.push_back(slot);
    output_ = codec::MakeTensorLike(output);
    for (int i = 0; i < 256; ++i) lookup_[i] = codec::FloatToHalfBits(i / 255.0f);
    for (auto& tensor : inputs_)
      std::fill_n(reinterpret_cast<uint16_t*>(tensor.bytes.data()), tensor.bytes.size() / 2,
                  lookup_[114]);
    if (!options_.report.empty()) {
      if (!options_.report.parent_path().empty())
        std::filesystem::create_directories(options_.report.parent_path());
      report_.open(options_.report);
      Check(report_.good(), "cannot open detection report");
    }
    worker_ = std::thread([this] { RunInference(); });
    try {
      preparer_ = std::thread([this] { RunPreparation(); });
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        preparation_done_ = true;
      }
      condition_.notify_all();
      worker_.join();
      throw;
    }
  }

  ~Impl() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    condition_.notify_all();
    preparer_.join();
    worker_.join();
  }

  std::shared_future<transport::DetectionFrame> Submit(int frame_id,
                                                       const std::vector<uint8_t>& nv12) {
    Check(frame_id >= 0 && nv12.size() == static_cast<std::size_t>(width_) * height_ * 3 / 2,
          "detection requires the same frame's tightly packed source NV12");
    const auto begin = Clock::now();
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] {
      return stopping_ || error_ ||
             queue_.size() < static_cast<std::size_t>(options_.queue_capacity);
    });
    if (error_) std::rethrow_exception(error_);
    Check(!stopping_, "detection worker is closed");
    const auto available = Clock::now();
    Job job;
    job.frame_id = frame_id;
    job.nv12 = nv12;
    job.queued_at = Clock::now();
    Trace("detection.submit_wait", frame_id, begin, available, "pipeline");
    Trace("detection.submit_copy", frame_id, available, job.queued_at, "pipeline");
    stats_.submit_wait_ms += Milliseconds(begin, available);
    stats_.submit_copy_ms += Milliseconds(available, job.queued_at);
    auto future = job.promise.get_future().share();
    queue_.push_back(std::move(job));
    lock.unlock();
    condition_.notify_all();
    return future;
  }

  Stats stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
  }

 private:
  struct Job {
    int frame_id = 0;
    std::vector<uint8_t> nv12;
    std::promise<transport::DetectionFrame> promise;
    Clock::time_point queued_at;
  };

  struct PreparedJob {
    Job job;
    int slot = 0;
    Clock::time_point prepared_at;
  };

  void Trace(const char* name, int frame_id, Clock::time_point begin, Clock::time_point end,
             const char* thread = "detection") {
    if (profiler_)
      profiler_->AddEventWithArgs(name, "detection", thread, profiler_->StartMs(begin),
                                  Milliseconds(begin, end), {Profiler::Arg("frame_id", frame_id)});
  }

  PreparedJob Prepare(Job& job, int slot) {
    auto& input = inputs_[slot];
    const auto begin = Clock::now();
    const uint8_t* y = job.nv12.data();
    const uint8_t* uv = y + static_cast<std::size_t>(width_) * height_;
    std::size_t stride = width_;
    const auto colored = begin;
    if (resized_width_ != width_ || resized_height_ != height_) {
      resized_nv12_.create(resized_height_ * 3 / 2, resized_width_, CV_8UC1);
      cv::Mat source_y(height_, width_, CV_8UC1, const_cast<uint8_t*>(y));
      cv::Mat source_uv(height_ / 2, width_ / 2, CV_8UC2, const_cast<uint8_t*>(uv));
      cv::Mat target_y(resized_height_, resized_width_, CV_8UC1, resized_nv12_.data);
      cv::Mat target_uv(
          resized_height_ / 2, resized_width_ / 2, CV_8UC2,
          resized_nv12_.data + static_cast<std::size_t>(resized_width_) * resized_height_);
      cv::resize(source_y, target_y, target_y.size(), 0, 0, cv::INTER_LINEAR);
      cv::resize(source_uv, target_uv, target_uv.size(), 0, 0, cv::INTER_LINEAR);
      y = target_y.data;
      uv = target_uv.data;
      stride = resized_width_;
    }
    const auto resized_at = Clock::now();
    PackNv12ToRgbFp16(y, stride, uv, stride, resized_width_, resized_height_, input_width_,
                      input_height_, left_, top_, reinterpret_cast<uint16_t*>(input.bytes.data()));
    const auto prepared = Clock::now();
    Trace("detection.queue_delay", job.frame_id, job.queued_at, begin, "detection_prepare");
    Trace("detection.color_convert", job.frame_id, begin, colored, "detection_prepare");
    Trace("detection.resize", job.frame_id, colored, resized_at, "detection_prepare");
    Trace("detection.pack_fp16", job.frame_id, resized_at, prepared, "detection_prepare");
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stats_.preprocess_ms += Milliseconds(begin, prepared);
      stats_.color_convert_ms += Milliseconds(begin, colored);
      stats_.resize_ms += Milliseconds(colored, resized_at);
      stats_.pack_ms += Milliseconds(resized_at, prepared);
    }
    return PreparedJob{std::move(job), slot, prepared};
  }

  transport::DetectionFrame Infer(const PreparedJob& ready) {
    runtime_.MakeCurrent();
    const auto& job = ready.job;
    auto& input = inputs_[ready.slot];
    const auto prepared = Clock::now();
    Trace("detection.prepared_queue_delay", job.frame_id, ready.prepared_at, prepared);
    stage_.Run({{stage_.record().inputs[0].name.c_str(), input.View()}},
               {{stage_.record().outputs[0].name.c_str(), output_.View()}});
    const auto inferred = Clock::now();
    const auto* raw = reinterpret_cast<const uint16_t*>(output_.bytes.data());
    const std::size_t count = output_.shape.dim(2);
    const uint16_t threshold = codec::FloatToHalfBits(options_.confidence);
    // Traverse each contiguous class plane once, rather than striding across 80 planes per box.
    std::vector<uint16_t> best_scores(count, 0), best_classes(count, 0);
    for (uint16_t cls = 0; cls < 80; ++cls) {
      const auto* scores = raw + (cls + 4) * count;
      std::size_t i = 0;
#ifdef __aarch64__
      const uint16x8_t class_ids = vdupq_n_u16(cls);
      for (; i + 8 <= count; i += 8) {
        const uint16x8_t next = vld1q_u16(scores + i);
        const uint16x8_t best = vld1q_u16(best_scores.data() + i);
        const uint16x8_t mask = vcgtq_u16(next, best);
        vst1q_u16(best_scores.data() + i, vbslq_u16(mask, next, best));
        vst1q_u16(best_classes.data() + i,
                  vbslq_u16(mask, class_ids, vld1q_u16(best_classes.data() + i)));
      }
#endif
      for (; i < count; ++i) {
        if (scores[i] > best_scores[i]) {
          best_scores[i] = scores[i];
          best_classes[i] = cls;
        }
      }
    }
    std::vector<transport::DetectionBox> candidates;
    for (std::size_t i = 0; i < count; ++i) {
      const uint16_t best = best_scores[i], class_id = best_classes[i];
      if (best < threshold) continue;
      const float confidence = codec::HalfBitsToFloat(best);
      const float cx = codec::HalfBitsToFloat(raw[i]);
      const float cy = codec::HalfBitsToFloat(raw[count + i]);
      const float width = codec::HalfBitsToFloat(raw[2 * count + i]);
      const float height = codec::HalfBitsToFloat(raw[3 * count + i]);
      if (!std::isfinite(confidence) || confidence > 1 || !std::isfinite(cx) ||
          !std::isfinite(cy) || !std::isfinite(width) || !std::isfinite(height))
        continue;
      transport::DetectionBox box{class_id, -1,    confidence, cx,
                                  cy,       width, height,     labels_[class_id]};
      if (!RestoreBox(&box, plan_)) continue;
      candidates.push_back(std::move(box));
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const auto& a, const auto& b) { return a.confidence > b.confidence; });
    if (candidates.size() > 3000) candidates.resize(3000);
    transport::DetectionFrame result{static_cast<uint32_t>(job.frame_id),
                                     static_cast<uint16_t>(width_),
                                     static_cast<uint16_t>(height_),
                                     {}};
    for (const auto& candidate : candidates) {
      const bool suppressed =
          std::any_of(result.boxes.begin(), result.boxes.end(), [&](const auto& box) {
            return candidate.class_id == box.class_id &&
                   IntersectionOverUnion(candidate, box) > options_.nms_iou;
          });
      if (!suppressed) result.boxes.push_back(candidate);
      if (result.boxes.size() == transport::kMaxDetections) break;
    }
    const auto finished = Clock::now();
    Trace("detection.model_call", job.frame_id, prepared, inferred);
    Trace("detection.postprocess", job.frame_id, inferred, finished);
    if (report_.is_open()) {
      report_ << "{\"frame_id\":" << result.frame_id << ",\"boxes\":[";
      for (std::size_t i = 0; i < result.boxes.size(); ++i) {
        const auto& box = result.boxes[i];
        if (i) report_ << ',';
        report_ << "{\"class_id\":" << box.class_id
                << ",\"track_id\":-1,\"label\":" << std::quoted(box.label)
                << ",\"conf\":" << box.confidence << ",\"cx\":" << box.cx << ",\"cy\":" << box.cy
                << ",\"w\":" << box.width << ",\"h\":" << box.height << '}';
      }
      report_ << "]}\n";
      Check(report_.good(), "failed to write detection report");
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++stats_.frames;
      stats_.boxes += result.boxes.size();
      stats_.inference_ms += Milliseconds(prepared, inferred);
      stats_.postprocess_ms += Milliseconds(inferred, finished);
    }
    return result;
  }

  void Fail(std::exception_ptr error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!error_) error_ = error;
    for (auto& pending : queue_) pending.promise.set_exception(error_);
    for (auto& pending : prepared_) pending.job.promise.set_exception(error_);
    queue_.clear();
    prepared_.clear();
    stopping_ = true;
    condition_.notify_all();
  }

  void RunPreparation() {
    for (;;) {
      Job job;
      int slot;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] {
          return error_ || (stopping_ && queue_.empty()) ||
                 (!queue_.empty() && !free_slots_.empty());
        });
        if (error_) return;
        if (queue_.empty() && stopping_) {
          preparation_done_ = true;
          condition_.notify_all();
          return;
        }
        job = std::move(queue_.front());
        queue_.pop_front();
        slot = free_slots_.front();
        free_slots_.pop_front();
      }
      condition_.notify_all();
      PreparedJob ready;
      bool prepared = false;
      try {
        ready = Prepare(job, slot);
        prepared = true;
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return error_ || prepared_.size() < 1; });
        if (error_) {
          ready.job.promise.set_exception(error_);
          return;
        }
        prepared_.push_back(std::move(ready));
      } catch (...) {
        const auto error = std::current_exception();
        if (prepared)
          ready.job.promise.set_exception(error);
        else
          job.promise.set_exception(error);
        Fail(error);
        return;
      }
      condition_.notify_all();
    }
  }

  void RunInference() {
    for (;;) {
      PreparedJob ready;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return error_ || preparation_done_ || !prepared_.empty(); });
        if (error_ || (prepared_.empty() && preparation_done_)) return;
        ready = std::move(prepared_.front());
        prepared_.pop_front();
      }
      condition_.notify_all();
      try {
        ready.job.promise.set_value(Infer(ready));
      } catch (...) {
        const auto error = std::current_exception();
        ready.job.promise.set_exception(error);
        Fail(error);
        return;
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        free_slots_.push_back(ready.slot);
      }
      condition_.notify_all();
    }
  }

  Options options_;
  int width_, height_, input_width_, input_height_, resized_width_, resized_height_, left_, top_;
  Letterbox plan_{};
  AclRuntime runtime_;
  AclStage stage_;
  Profiler* profiler_ = nullptr;
  std::array<codec::TensorData, 3> inputs_;
  codec::TensorData output_;
  cv::Mat resized_nv12_;
  std::vector<std::string> labels_;
  std::array<uint16_t, 256> lookup_{};
  std::ofstream report_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<Job> queue_;
  std::deque<PreparedJob> prepared_;
  std::deque<int> free_slots_;
  bool preparation_done_ = false;
  bool stopping_ = false;
  std::exception_ptr error_;
  Stats stats_;
  std::thread worker_, preparer_;
};

YoloWorker::YoloWorker(aclrtContext context, Options options, int width, int height,
                       Profiler* profiler)
    : impl_(std::make_unique<Impl>(context, std::move(options), width, height, profiler)) {}
YoloWorker::~YoloWorker() = default;
std::shared_future<transport::DetectionFrame> YoloWorker::Submit(int frame_id,
                                                                 const std::vector<uint8_t>& nv12) {
  return impl_->Submit(frame_id, nv12);
}
Stats YoloWorker::stats() const { return impl_->stats(); }
}  // namespace ulbvc::detection
