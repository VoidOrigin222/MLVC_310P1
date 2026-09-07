#include <mlvc/application/input/frame_input_queue.h>

#include <chrono>

#include "mlvc/core/status.h"
#include "mlvc/framework/profile_range.h"

namespace mlvc::app {
namespace codec = mlvc::codec;
namespace {

void AddPrepareProfileEvent(mlvc::Profiler* profiler, std::string name, double start_ms,
                            double duration_ms, std::vector<mlvc::ProfileArgument> args) {
  if (profiler == nullptr) {
    return;
  }
  profiler->AddEventWithArgs(std::move(name), "video_io", "video_io", start_ms, duration_ms,
                             std::move(args));
}

}  // namespace

FrameInputArena::FrameInputArena(const mlvc::TensorSpec& frame_spec, int slot_count) {
  mlvc::Check(slot_count > 0, "frame prepare arena requires at least one slot");
  slots_.reserve(static_cast<std::size_t>(slot_count));
  for (int slot = 0; slot < slot_count; ++slot) {
    slots_.push_back(codec::MakeTensorLike(frame_spec));
  }
}

AsyncFrameInputQueue::AsyncFrameInputQueue(int frames_to_attempt, int configured_frame_num,
                                           const std::filesystem::path& input_video_path,
                                           const std::filesystem::path& input_frame_dir,
                                           const mlvc::TensorSpec& frame_spec, int width,
                                           int height, FrameBufferPool* arena,
                                           mlvc::Profiler* profiler,
                                           mlvc::CodecGraphExecutor* graph_executor)
    : frames_to_attempt_(frames_to_attempt),
      configured_frame_num_(configured_frame_num),
      input_video_path_(input_video_path),
      input_frame_dir_(input_frame_dir),
      frame_spec_(frame_spec),
      frame_source_(std::make_unique<mlvc::io::FrameSource>(configured_frame_num, input_video_path,
                                                            input_frame_dir, frame_spec)),
      width_(width),
      height_(height),
      arena_(arena),
      profiler_(profiler),
      graph_executor_(graph_executor),
      worker_pool_(1) {
  mlvc::Check(arena_ != nullptr && arena_->capacity() > 0,
              "async frame prepare queue requires a non-empty arena");
  for (int slot = 0; slot < static_cast<int>(arena_->capacity()); ++slot) {
    free_slots_.push_back(slot);
  }
  worker_future_ = worker_pool_.SubmitValue<void>([this] { WorkerMain(); });
}

AsyncFrameInputQueue::~AsyncFrameInputQueue() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  producer_condition_.notify_all();
  consumer_condition_.notify_all();
  if (worker_future_.valid()) worker_future_.get();
  worker_pool_.Stop();
}

InputFrame AsyncFrameInputQueue::Pop() {
  const auto wait_begin = std::chrono::steady_clock::now();
  std::unique_lock<std::mutex> lock(mutex_);
  consumer_condition_.wait(
      lock, [this] { return !ready_slots_.empty() || producer_done_ || exception_ != nullptr; });
  const auto wait_end = std::chrono::steady_clock::now();
  {
    const double wait_ms = std::chrono::duration<double, std::milli>(wait_end - wait_begin).count();
    stats_.consumer_wait_ms += wait_ms;
    ++stats_.consumer_wait_count;
  }
  if (exception_ != nullptr) {
    std::rethrow_exception(exception_);
  }
  if (ready_slots_.empty()) {
    return InputFrame{next_frame_to_consume_, nullptr, -1, true};
  }
  ReadySlot ready = ready_slots_.front();
  ready_slots_.pop_front();
  next_frame_to_consume_ = ready.frame_index + 1;
  lock.unlock();
  producer_condition_.notify_one();
  return InputFrame{ready.frame_index,
                    &arena_->Get(static_cast<std::size_t>(ready.slot_index)),
                    ready.slot_index, false};
}

void AsyncFrameInputQueue::Release(InputFrame* frame) {
  if (frame == nullptr || frame->slot_index < 0) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    free_slots_.push_back(frame->slot_index);
  }
  frame->frame = nullptr;
  frame->slot_index = -1;
  producer_condition_.notify_one();
}

AsyncFrameInputQueue::Stats AsyncFrameInputQueue::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

void AsyncFrameInputQueue::WorkerMain() {
  try {
    for (int frame_index = 0; frame_index < frames_to_attempt_; ++frame_index) {
      const int slot_index = AcquireFreeSlot();
      if (slot_index < 0) {
        break;
      }
      codec::TensorData& frame = arena_->Get(static_cast<std::size_t>(slot_index));
      const bool has_frame = PrepareOne(frame_index, &frame);
      if (!has_frame) {
        ReturnFreeSlot(slot_index);
        break;
      }
      PushReady(ReadySlot{frame_index, slot_index});
    }
  } catch (...) {
    std::lock_guard<std::mutex> lock(mutex_);
    exception_ = std::current_exception();
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    producer_done_ = true;
  }
  consumer_condition_.notify_all();
}

bool AsyncFrameInputQueue::PrepareOne(int frame_index, codec::TensorData* frame) {
  const auto prepare_begin = std::chrono::steady_clock::now();
  std::optional<mlvc::ScopedProfileHotPath> profile_prepare_hot_path;
  std::optional<mlvc::ScopedProfileRange> profile_prepare_frame;
  if (profiler_ != nullptr) {
    profile_prepare_hot_path.emplace();
    profile_prepare_frame.emplace(std::string("prepare.frame.") + std::to_string(frame_index));
  }
  if (frame_source_->has_video_input()) {
    std::optional<mlvc::ScopedProfileRange> profile_read_frame;
    if (profiler_ != nullptr) {
      profile_read_frame.emplace(std::string("prepare.frame.read_video.") +
                                 std::to_string(frame_index));
    }
    mlvc::ScopedCodecGraphNode graph_node(graph_executor_, profiler_,
                                          mlvc::CodecGraphNodeType::kVideoIo, "read_frame",
                                          "AsyncFrameInputQueue::ReadFrameAsTensor", frame_index);
    const auto video_read_begin = std::chrono::steady_clock::now();
    const bool has_frame = frame_source_->ReadFrame(frame_index, frame);
    const auto video_read_end = std::chrono::steady_clock::now();
    const double read_ms =
        std::chrono::duration<double, std::milli>(video_read_end - video_read_begin).count();
    if (!has_frame) {
      return false;
    }
    AddPrepareProfileEvent(
        profiler_, "video.read.frame",
        profiler_ != nullptr ? profiler_->StartMs(video_read_begin) : 0.0,
        profiler_ != nullptr ? profiler_->DurationMs(video_read_begin, video_read_end) : 0.0,
        {mlvc::Profiler::Arg("frame_index", frame_index), mlvc::Profiler::Arg("width", width_),
         mlvc::Profiler::Arg("height", height_),
         mlvc::Profiler::Arg("bytes", static_cast<uint64_t>(frame->bytes.size())),
         mlvc::Profiler::Arg("prepare_mode", "async_queue")});
    const auto prepare_end = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stats_.prepare_ms +=
          std::chrono::duration<double, std::milli>(prepare_end - prepare_begin).count();
      stats_.video_read_ms += read_ms;
      ++stats_.prepared_frames;
    }
    return true;
  }
  if (input_frame_dir_.empty()) {
    std::optional<mlvc::ScopedProfileRange> profile_synth_frame;
    if (profiler_ != nullptr) {
      profile_synth_frame.emplace(std::string("prepare.frame.synthetic.") +
                                  std::to_string(frame_index));
    }
    mlvc::ScopedCodecGraphNode graph_node(graph_executor_, profiler_,
                                          mlvc::CodecGraphNodeType::kVideoIo, "synthetic_frame",
                                          "AsyncFrameInputQueue::FillFp16Tensor", frame_index);
    const auto video_synth_begin = std::chrono::steady_clock::now();
    const bool has_frame = frame_source_->ReadFrame(frame_index, frame);
    const auto video_synth_end = std::chrono::steady_clock::now();
    const double read_ms =
        std::chrono::duration<double, std::milli>(video_synth_end - video_synth_begin).count();
    if (!has_frame) {
      return false;
    }
    AddPrepareProfileEvent(
        profiler_, "video.synthetic_frame",
        profiler_ != nullptr ? profiler_->StartMs(video_synth_begin) : 0.0,
        profiler_ != nullptr ? profiler_->DurationMs(video_synth_begin, video_synth_end) : 0.0,
        {mlvc::Profiler::Arg("frame_index", frame_index),
         mlvc::Profiler::Arg("bytes", static_cast<uint64_t>(frame->bytes.size())),
         mlvc::Profiler::Arg("prepare_mode", "async_queue")});
    const auto prepare_end = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stats_.prepare_ms +=
          std::chrono::duration<double, std::milli>(prepare_end - prepare_begin).count();
      stats_.synthetic_read_ms += read_ms;
      ++stats_.prepared_frames;
    }
    return true;
  }

  std::optional<mlvc::ScopedProfileRange> profile_read_fp16_frame;
  if (profiler_ != nullptr) {
    profile_read_fp16_frame.emplace(std::string("prepare.frame.read_fp16.") +
                                    std::to_string(frame_index));
  }
  mlvc::ScopedCodecGraphNode graph_node(graph_executor_, profiler_,
                                        mlvc::CodecGraphNodeType::kVideoIo, "read_fp16_frame",
                                        "AsyncFrameInputQueue::ReadTensorFileInto", frame_index);
  const auto frame_read_begin = std::chrono::steady_clock::now();
  const std::filesystem::path frame_path =
      input_frame_dir_ / ("frame_" + std::to_string(frame_index) + ".fp16");
  const bool has_frame = frame_source_->ReadFrame(frame_index, frame);
  const auto frame_read_end = std::chrono::steady_clock::now();
  const double read_ms =
      std::chrono::duration<double, std::milli>(frame_read_end - frame_read_begin).count();
  if (!has_frame) {
    return false;
  }
  AddPrepareProfileEvent(
      profiler_, "video.read.fp16_frame",
      profiler_ != nullptr ? profiler_->StartMs(frame_read_begin) : 0.0,
      profiler_ != nullptr ? profiler_->DurationMs(frame_read_begin, frame_read_end) : 0.0,
      {mlvc::Profiler::Arg("frame_index", frame_index),
       mlvc::Profiler::Arg("path", frame_path.string()),
       mlvc::Profiler::Arg("bytes", static_cast<uint64_t>(frame->bytes.size())),
       mlvc::Profiler::Arg("prepare_mode", "async_queue")});
  const auto prepare_end = std::chrono::steady_clock::now();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.prepare_ms +=
        std::chrono::duration<double, std::milli>(prepare_end - prepare_begin).count();
    stats_.fp16_read_ms += read_ms;
    ++stats_.prepared_frames;
  }
  return true;
}

int AsyncFrameInputQueue::AcquireFreeSlot() {
  const auto wait_begin = std::chrono::steady_clock::now();
  std::unique_lock<std::mutex> lock(mutex_);
  producer_condition_.wait(lock, [this] { return stop_ || !free_slots_.empty(); });
  const auto wait_end = std::chrono::steady_clock::now();
  if (stop_) {
    return -1;
  }
  {
    const double wait_ms = std::chrono::duration<double, std::milli>(wait_end - wait_begin).count();
    stats_.producer_wait_ms += wait_ms;
    ++stats_.producer_wait_count;
  }
  const int slot_index = free_slots_.front();
  free_slots_.pop_front();
  return slot_index;
}

void AsyncFrameInputQueue::ReturnFreeSlot(int slot_index) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    free_slots_.push_back(slot_index);
  }
  producer_condition_.notify_one();
}

void AsyncFrameInputQueue::PushReady(ReadySlot ready) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stop_) {
      free_slots_.push_back(ready.slot_index);
      return;
    }
    ready_slots_.push_back(ready);
  }
  consumer_condition_.notify_one();
}

}  // namespace mlvc::app
