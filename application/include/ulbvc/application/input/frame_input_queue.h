#ifndef ULBVC_APPLICATION_INPUT_FRAME_INPUT_QUEUE_H_
#define ULBVC_APPLICATION_INPUT_FRAME_INPUT_QUEUE_H_

#include <ulbvc/application/input/frame_buffer_pool.h>
#include <ulbvc/codec/tensor_data.h>
#include <ulbvc/io/frame_source.h>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <filesystem>
#include <future>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "ulbvc/framework/codec_graph_executor.h"
#include "ulbvc/framework/profiler.h"
#include "ulbvc/framework/thread_pool.h"

namespace ulbvc::app {

struct InputFrame {
  int frame_index = 0;
  codec::TensorData* frame = nullptr;
  int slot_index = -1;
  bool eof = false;
  std::chrono::steady_clock::time_point ready_at{};
  // Borrowed from the same queue slot; remains valid until Release.
  const std::vector<uint8_t>* motion_nv12 = nullptr;
};

class AsyncFrameInputQueue {
 public:
  using MotionInputPrepareFunction =
      std::function<void(const codec::TensorData&, std::vector<uint8_t>*)>;

  struct Stats {
    double prepare_ms = 0.0;
    double motion_prepare_ms = 0.0;
    double motion_prepare_max_ms = 0.0;
    double video_read_ms = 0.0;
    double fp16_read_ms = 0.0;
    double synthetic_read_ms = 0.0;
    double producer_wait_ms = 0.0;
    double consumer_wait_ms = 0.0;
    int prepared_frames = 0;
    int producer_wait_count = 0;
    int consumer_wait_count = 0;
    int producer_full_wait_count = 0;
    double producer_full_wait_ms = 0.0;
    double producer_full_wait_max_ms = 0.0;
    int consumer_empty_wait_count = 0;
    double consumer_empty_wait_ms = 0.0;
    double consumer_empty_wait_max_ms = 0.0;
    std::size_t ready_max_depth = 0;
    std::size_t max_acquired_frames = 0;
  };

  AsyncFrameInputQueue(int frames_to_attempt, int configured_frame_num,
                       const std::filesystem::path& input_video_path,
                       const std::filesystem::path& input_frame_dir,
                       std::optional<ulbvc::io::CameraCaptureOptions> camera_options,
                       aclrtContext context,
                       const ulbvc::TensorSpec& frame_spec, int width, int height,
                       FrameBufferPool* arena, ulbvc::Profiler* profiler,
                       ulbvc::CodecGraphExecutor* graph_executor,
                       MotionInputPrepareFunction motion_prepare = {});

  AsyncFrameInputQueue(const AsyncFrameInputQueue&) = delete;
  AsyncFrameInputQueue& operator=(const AsyncFrameInputQueue&) = delete;

  ~AsyncFrameInputQueue();

  InputFrame Pop();
  void Cancel(std::exception_ptr error = nullptr) noexcept;
  void RethrowIfFailed() const;
  void Release(InputFrame* frame);
  Stats stats() const;

 private:
  struct ReadySlot {
    int frame_index = 0;
    int slot_index = -1;
    std::chrono::steady_clock::time_point ready_at{};
  };

  void WorkerMain();
  bool PrepareOne(int frame_index, codec::TensorData* frame,
                  std::vector<uint8_t>* motion_nv12 = nullptr);
  int AcquireFreeSlot();
  void ReturnFreeSlot(int slot_index);
  void PushReady(ReadySlot ready);

  int frames_to_attempt_ = 0;
  int configured_frame_num_ = 0;
  std::filesystem::path input_video_path_;
  std::filesystem::path input_frame_dir_;
  ulbvc::TensorSpec frame_spec_;
  bool camera_motion_nv12_ = false;
  std::unique_ptr<ulbvc::io::FrameSource> frame_source_;
  int width_ = 0;
  int height_ = 0;
  FrameBufferPool* arena_ = nullptr;
  ulbvc::Profiler* profiler_ = nullptr;
  ulbvc::CodecGraphExecutor* graph_executor_ = nullptr;
  mutable std::mutex mutex_;
  std::condition_variable producer_condition_;
  std::condition_variable consumer_condition_;
  std::deque<int> free_slots_;
  std::deque<ReadySlot> ready_slots_;
  std::exception_ptr exception_;
  MotionInputPrepareFunction motion_prepare_;
  std::vector<std::vector<uint8_t>> motion_nv12_slots_;
  ulbvc::ThreadPool worker_pool_;
  std::future<void> worker_future_;
  int next_frame_to_consume_ = 0;
  bool producer_done_ = false;
  bool stop_ = false;
  Stats stats_;
};

}  // namespace ulbvc::app

#endif  // ULBVC_APPLICATION_INPUT_FRAME_INPUT_QUEUE_H_
