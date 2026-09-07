#ifndef MLVC_APPLICATION_INPUT_FRAME_INPUT_QUEUE_H_
#define MLVC_APPLICATION_INPUT_FRAME_INPUT_QUEUE_H_

#include <mlvc/codec/tensor_data.h>
#include <mlvc/io/frame_source.h>
#include <mlvc/application/input/frame_buffer_pool.h>

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "mlvc/framework/codec_graph_executor.h"
#include "mlvc/framework/profiler.h"
#include "mlvc/framework/thread_pool.h"

namespace mlvc::app {

struct InputFrame {
  int frame_index = 0;
  codec::TensorData* frame = nullptr;
  int slot_index = -1;
  bool eof = false;
};

class AsyncFrameInputQueue {
 public:
  struct Stats {
    double prepare_ms = 0.0;
    double video_read_ms = 0.0;
    double fp16_read_ms = 0.0;
    double synthetic_read_ms = 0.0;
    double producer_wait_ms = 0.0;
    double consumer_wait_ms = 0.0;
    int prepared_frames = 0;
    int producer_wait_count = 0;
    int consumer_wait_count = 0;
  };

  AsyncFrameInputQueue(int frames_to_attempt, int configured_frame_num,
                       const std::filesystem::path& input_video_path,
                       const std::filesystem::path& input_frame_dir,
                       const mlvc::TensorSpec& frame_spec, int width, int height,
                       FrameBufferPool* arena, mlvc::Profiler* profiler,
                       mlvc::CodecGraphExecutor* graph_executor);

  AsyncFrameInputQueue(const AsyncFrameInputQueue&) = delete;
  AsyncFrameInputQueue& operator=(const AsyncFrameInputQueue&) = delete;

  ~AsyncFrameInputQueue();

  InputFrame Pop();
  void Release(InputFrame* frame);
  Stats stats() const;

 private:
  struct ReadySlot {
    int frame_index = 0;
    int slot_index = -1;
  };

  void WorkerMain();
  bool PrepareOne(int frame_index, codec::TensorData* frame);
  int AcquireFreeSlot();
  void ReturnFreeSlot(int slot_index);
  void PushReady(ReadySlot ready);

  int frames_to_attempt_ = 0;
  int configured_frame_num_ = 0;
  std::filesystem::path input_video_path_;
  std::filesystem::path input_frame_dir_;
  mlvc::TensorSpec frame_spec_;
  std::unique_ptr<mlvc::io::FrameSource> frame_source_;
  int width_ = 0;
  int height_ = 0;
  FrameBufferPool* arena_ = nullptr;
  mlvc::Profiler* profiler_ = nullptr;
  mlvc::CodecGraphExecutor* graph_executor_ = nullptr;
  mutable std::mutex mutex_;
  std::condition_variable producer_condition_;
  std::condition_variable consumer_condition_;
  std::deque<int> free_slots_;
  std::deque<ReadySlot> ready_slots_;
  std::exception_ptr exception_;
  mlvc::ThreadPool worker_pool_;
  std::future<void> worker_future_;
  int next_frame_to_consume_ = 0;
  bool producer_done_ = false;
  bool stop_ = false;
  Stats stats_;
};

}  // namespace mlvc::app

#endif  // MLVC_APPLICATION_INPUT_FRAME_INPUT_QUEUE_H_
