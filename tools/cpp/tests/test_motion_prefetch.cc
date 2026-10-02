#include <mlvc/application/input/frame_input_queue.h>
#include <mlvc/application/pipeline/codec_frame_pipeline.h>
#include <mlvc/core/status.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace std::chrono_literals;

// A broken cancellation path must fail CTest instead of hanging its runner.
class Watchdog {
 public:
  Watchdog() : worker_([this] {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!cv_.wait_for(lock, 20s, [this] { return done_; })) {
      std::cerr << "motion prefetch test exceeded its cancellation deadline\n";
      std::abort();
    }
  }) {}
  ~Watchdog() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      done_ = true;
    }
    cv_.notify_all();
    worker_.join();
  }
 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool done_ = false;
  std::thread worker_;
};

mlvc::TensorSpec FrameSpec() {
  return {"x", mlvc::DataType::kFloat16, {1, 3, 8, 8}};
}

void CheckSynthetic(const mlvc::app::PreparedFramePacket& packet) {
  const auto& input = packet.frame();
  mlvc::Check(input.frame != nullptr && input.slot_index >= 0, "packet lost its arena lease");
  mlvc::Check(input.ready_at != std::chrono::steady_clock::time_point{} &&
                  input.ready_at <= std::chrono::steady_clock::now(),
              "prepared frame readiness timestamp is invalid");
  const auto& bytes = input.frame->bytes;
  const auto bits = mlvc::codec::FloatToHalfBits(input.frame_index == 0 ? 0.5f : 0.51f);
  mlvc::Check(bytes.size() == 3 * 8 * 8 * 2, "unexpected synthetic frame size");
  for (std::size_t i = 0; i < bytes.size(); i += 2) {
    const uint16_t actual = static_cast<uint16_t>(bytes[i]) |
                            (static_cast<uint16_t>(bytes[i + 1]) << 8);
    mlvc::Check(actual == bits, "arena slot was overwritten before packet release");
  }
}

void TestOrderedEof(int count, int prefetch, const std::filesystem::path& directory = {}) {
  const auto spec = FrameSpec();
  mlvc::app::FrameInputArena arena(spec, prefetch + 1);
  mlvc::app::AsyncFrameInputQueue input(directory.empty() ? count : count + 5,
                                      directory.empty() ? count : -1,
                                      {}, directory, std::nullopt, nullptr,
                                      spec, 8, 8, &arena, nullptr, nullptr);
  std::atomic<bool> running{true};
  mlvc::StreamingPipeline pipeline(1, prefetch + 1);
  int motion_index = 0;
  pipeline.SetProcessor([&](const std::shared_ptr<mlvc::DataObject>& data) {
    auto packet = std::dynamic_pointer_cast<mlvc::app::PreparedFramePacket>(data);
    mlvc::Check(packet && packet->frame().frame_index == motion_index++, "motion order mismatch");
    CheckSynthetic(*packet);
    packet->motion = mlvc::motion::Translation{static_cast<int8_t>(packet->frame().frame_index), -1};
    return data;
  });
  pipeline.Start();
  mlvc::app::PreparedFrameProducer producer(pipeline, running, input);
  std::vector<int> frames;
  mlvc::app::CallbackDataConsumer consumer(pipeline, running,
      [&](const std::shared_ptr<mlvc::DataObject>& data) {
        auto packet = std::dynamic_pointer_cast<mlvc::app::PreparedFramePacket>(data);
        mlvc::Check(packet && packet->frame().frame_index == static_cast<int>(frames.size()),
                    "encoding order mismatch");
        CheckSynthetic(*packet);
        mlvc::Check(packet->motion && packet->motion->kx == packet->frame().frame_index &&
                        packet->motion->ky == -1, "motion result attached to the wrong frame");
        frames.push_back(packet->frame().frame_index);
        packet->Release();
      }, [&] { input.Cancel(std::current_exception()); });
  consumer.Start();
  producer.Start();
  producer.Join();
  consumer.Join();
  producer.RethrowIfFailed();
  consumer.RethrowIfFailed();
  pipeline.Stop();
  pipeline.RethrowIfFailed();
  mlvc::Check(static_cast<int>(frames.size()) == count && motion_index == count,
              "EOF lost or duplicated an accepted frame");
  mlvc::Check(input.stats().max_acquired_frames <= static_cast<std::size_t>(prefetch + 1),
              "motion prefetch exceeded the arena bound");
}

void TestTwoFutureFramesWhileEncodingIsBlocked() {
  const auto spec = FrameSpec();
  mlvc::app::FrameInputArena arena(spec, 3);
  mlvc::app::AsyncFrameInputQueue input(9, 9, {}, {}, std::nullopt, nullptr,
                                      spec, 8, 8, &arena, nullptr, nullptr);
  std::atomic<bool> running{true};
  std::atomic<int> last_motion{-1};
  mlvc::StreamingPipeline pipeline(1, 3);
  std::promise<void> third_motion, consumer_started, release_consumer;
  auto third_ready = third_motion.get_future();
  auto started = consumer_started.get_future();
  auto release = release_consumer.get_future().share();
  pipeline.SetProcessor([&](const std::shared_ptr<mlvc::DataObject>& data) {
    auto packet = std::dynamic_pointer_cast<mlvc::app::PreparedFramePacket>(data);
    CheckSynthetic(*packet);
    const int index = packet->frame().frame_index;
    packet->motion = mlvc::motion::Translation{static_cast<int8_t>(index), 0};
    last_motion = index;
    if (index == 2) third_motion.set_value();
    return data;
  });
  pipeline.Start();
  mlvc::app::PreparedFrameProducer producer(pipeline, running, input);
  int consumed = 0;
  mlvc::app::CallbackDataConsumer consumer(pipeline, running,
      [&](const std::shared_ptr<mlvc::DataObject>& data) {
        auto packet = std::dynamic_pointer_cast<mlvc::app::PreparedFramePacket>(data);
        if (packet->frame().frame_index == 0) {
          consumer_started.set_value();
          release.wait();
        }
        CheckSynthetic(*packet);
        mlvc::Check(packet->frame().frame_index == consumed++, "prefetch reordered encoding");
        packet->Release();
      }, [&] { input.Cancel(std::current_exception()); });
  consumer.Start();
  producer.Start();
  const bool did_start = started.wait_for(2s) == std::future_status::ready;
  const bool did_prefetch = third_ready.wait_for(2s) == std::future_status::ready;
  const int motion_before_release = last_motion.load();
  const auto acquired_before_release = input.stats().max_acquired_frames;
  release_consumer.set_value();
  producer.Join();
  consumer.Join();
  producer.RethrowIfFailed();
  consumer.RethrowIfFailed();
  pipeline.Stop();
  pipeline.RethrowIfFailed();
  mlvc::Check(did_start && did_prefetch, "two-frame motion overlap did not make progress");
  mlvc::Check(motion_before_release == 2 && acquired_before_release == 3,
              "motion got more than two frames ahead of the blocked encoder");
  mlvc::Check(consumed == 9, "releasing backpressure lost frames");
}

void TestConsumerFailureCancelsBackpressuredProducer() {
  const auto spec = FrameSpec();
  mlvc::app::FrameInputArena arena(spec, 3);
  mlvc::app::AsyncFrameInputQueue input(1000, 1000, {}, {}, std::nullopt, nullptr,
                                      spec, 8, 8, &arena, nullptr, nullptr);
  std::atomic<bool> running{true};
  mlvc::StreamingPipeline pipeline(1, 3);
  pipeline.SetProcessor([](const std::shared_ptr<mlvc::DataObject>& data) { return data; });
  pipeline.Start();
  mlvc::app::PreparedFrameProducer producer(pipeline, running, input);
  std::atomic<int> consumed{0};
  mlvc::app::CallbackDataConsumer consumer(pipeline, running,
      [&](const std::shared_ptr<mlvc::DataObject>&) {
        ++consumed;
        throw std::runtime_error("expected consumer failure");
      }, [&] { input.Cancel(std::current_exception()); });
  consumer.Start();
  producer.Start();
  producer.Join();
  consumer.Join();
  bool propagated = false;
  try { consumer.RethrowIfFailed(); }
  catch (const std::runtime_error& error) {
    propagated = std::string(error.what()) == "expected consumer failure";
  }
  mlvc::Check(propagated && consumed == 1, "consumer failure was swallowed or processing resumed");
  mlvc::Check(input.stats().prepared_frames < 1000, "consumer failure did not cancel preparation");
  bool original_cause = false;
  try { input.RethrowIfFailed(); }
  catch (const std::runtime_error& error) {
    original_cause = std::string(error.what()) == "expected consumer failure";
  }
  mlvc::Check(original_cause, "consumer cancellation lost its original failure cause");
}

void TestMotionFailureDoesNotEncodeLaterFrames() {
  const auto spec = FrameSpec();
  mlvc::app::FrameInputArena arena(spec, 3);
  mlvc::app::AsyncFrameInputQueue input(1000, 1000, {}, {}, std::nullopt, nullptr,
                                      spec, 8, 8, &arena, nullptr, nullptr);
  std::atomic<bool> running{true};
  mlvc::StreamingPipeline pipeline(1, 3);
  pipeline.SetProcessor([&](const std::shared_ptr<mlvc::DataObject>& data)
      -> std::shared_ptr<mlvc::DataObject> {
    if (!running) return nullptr;
    auto packet = std::dynamic_pointer_cast<mlvc::app::PreparedFramePacket>(data);
    try {
      if (packet->frame().frame_index == 1) throw std::runtime_error("expected motion failure");
      packet->motion = mlvc::motion::Translation{0, 0};
      return data;
    } catch (...) {
      running = false;
      input.Cancel(std::current_exception());
      pipeline.CloseInput();
      throw;
    }
  });
  pipeline.Start();
  mlvc::app::PreparedFrameProducer producer(pipeline, running, input);
  std::vector<int> consumed;
  mlvc::app::CallbackDataConsumer consumer(pipeline, running,
      [&](const std::shared_ptr<mlvc::DataObject>& data) {
        pipeline.RethrowIfFailed();
        auto packet = std::dynamic_pointer_cast<mlvc::app::PreparedFramePacket>(data);
        consumed.push_back(packet->frame().frame_index);
        packet->Release();
      }, [&] { input.Cancel(std::current_exception()); });
  consumer.Start();
  producer.Start();
  producer.Join();
  consumer.Join();
  pipeline.Stop();
  bool propagated = false;
  try { pipeline.RethrowIfFailed(); }
  catch (const std::runtime_error& error) {
    propagated = std::string(error.what()) == "expected motion failure";
  }
  mlvc::Check(propagated, "motion producer exception was swallowed");
  for (const int frame : consumed)
    mlvc::Check(frame == 0, "a frame following a failed motion frame was encoded");
  mlvc::Check(input.stats().prepared_frames < 1000, "motion failure did not cancel input producer");
}

void TestCancelWakesPopAndFreeSlotWait() {
  const auto spec = FrameSpec();
  mlvc::app::FrameInputArena arena(spec, 1);
  mlvc::app::AsyncFrameInputQueue input(1000, 1000, {}, {}, std::nullopt, nullptr,
                                      spec, 8, 8, &arena, nullptr, nullptr);
  auto held = input.Pop();
  auto waiting = std::async(std::launch::async, [&] { return input.Pop(); });
  const bool was_blocked = waiting.wait_for(30ms) == std::future_status::timeout;
  input.Cancel();
  input.Cancel();  // Cancellation must be idempotent.
  const bool did_wake = waiting.wait_for(2s) == std::future_status::ready;
  mlvc::Check(did_wake && waiting.get().eof, "Cancel did not wake Pop with EOF");
  input.Release(&held);
  mlvc::Check(was_blocked, "Pop unexpectedly bypassed the held arena lease");
}

void TestInputFailurePropagatesThroughProducer() {
  const auto spec = FrameSpec();
  const auto directory = std::filesystem::temp_directory_path() /
      ("mlvc-prefetch-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directory(directory);
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
  } cleanup{directory};
  mlvc::codec::WriteTensorFile(directory / "frame_0.fp16",
                              mlvc::codec::MakeFp16Tensor(spec.shape, 0.5f));
  mlvc::app::FrameInputArena arena(spec, 3);
  mlvc::app::AsyncFrameInputQueue input(4, 4, {}, directory, std::nullopt, nullptr,
                                      spec, 8, 8, &arena, nullptr, nullptr);
  std::atomic<bool> running{true};
  mlvc::StreamingPipeline pipeline(1, 3);
  pipeline.SetProcessor([](const std::shared_ptr<mlvc::DataObject>& data) { return data; });
  pipeline.Start();
  mlvc::app::PreparedFrameProducer producer(pipeline, running, input);
  mlvc::app::CallbackDataConsumer consumer(pipeline, running,
      [](const std::shared_ptr<mlvc::DataObject>& data) {
        auto packet = std::dynamic_pointer_cast<mlvc::app::PreparedFramePacket>(data);
        mlvc::Check(packet->frame().frame_index == 0, "input failure allowed later frames");
        packet->Release();
      }, [&] { input.Cancel(std::current_exception()); });
  consumer.Start();
  producer.Start();
  producer.Join();
  consumer.Join();
  bool propagated = false;
  try { producer.RethrowIfFailed(); }
  catch (const std::exception& error) {
    propagated = std::string(error.what()).find("missing input frame:") != std::string::npos;
  }
  input.Cancel();
  bool preserved = false;
  try { (void)input.Pop(); }
  catch (const std::exception& error) {
    preserved = std::string(error.what()).find("missing input frame:") != std::string::npos;
  }
  mlvc::Check(propagated && preserved, "input producer failure was replaced by cancellation EOF");
  pipeline.Stop();
}

void TestSourceEofDrainsAcceptedPackets() {
  const auto spec = FrameSpec();
  const auto directory = std::filesystem::temp_directory_path() /
      ("mlvc-prefetch-eof-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directory(directory);
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
  } cleanup{directory};
  for (int frame = 0; frame < 7; ++frame) {
    mlvc::codec::WriteTensorFile(directory / ("frame_" + std::to_string(frame) + ".fp16"),
        mlvc::codec::MakeFp16Tensor(spec.shape, frame == 0 ? 0.5f : 0.51f));
  }
  // An unknown-length source ends at frame 7, before the queue's attempt limit.
  // The producer must close input only after all seven accepted packets drain.
  TestOrderedEof(7, 1, directory);
  TestOrderedEof(7, 2, directory);
}

void CheckMotionCache(const mlvc::app::InputFrame& frame) {
  mlvc::Check(frame.motion_nv12 != nullptr &&
                  frame.motion_nv12->size() == static_cast<std::size_t>(64 + frame.frame_index),
              "motion cache is missing or belongs to another frame");
  for (const auto byte : *frame.motion_nv12)
    mlvc::Check(byte == frame.frame_index, "leased motion cache was overwritten");
}

void TestMotionCacheUsesIndependentLeasedSlots() {
  const auto spec = FrameSpec();
  mlvc::app::FrameInputArena arena(spec, 3);
  std::atomic<int> prepared{0};
  mlvc::app::AsyncFrameInputQueue input(5, 5, {}, {}, std::nullopt, nullptr,
      spec, 8, 8, &arena, nullptr, nullptr,
      [&](const mlvc::codec::TensorData&, std::vector<uint8_t>* output) {
        const int frame = prepared++;
        output->assign(static_cast<std::size_t>(64 + frame), static_cast<uint8_t>(frame));
      });
  auto first = input.Pop();
  auto second = input.Pop();
  auto third = input.Pop();
  CheckMotionCache(first);
  CheckMotionCache(second);
  CheckMotionCache(third);
  mlvc::Check(first.motion_nv12 != second.motion_nv12 && first.motion_nv12 != third.motion_nv12 &&
                  second.motion_nv12 != third.motion_nv12,
              "in-flight frames share a mutable motion cache");
  const auto* first_slot_cache = first.motion_nv12;
  input.Release(&first);
  auto fourth = input.Pop();
  mlvc::Check(fourth.frame_index == 3 && fourth.motion_nv12 == first_slot_cache,
              "motion cache slot was not reused after its lease ended");
  CheckMotionCache(fourth);
  CheckMotionCache(second);
  CheckMotionCache(third);
  input.Release(&fourth);
  auto fifth = input.Pop();
  mlvc::Check(fifth.frame_index == 4 && fifth.motion_nv12 == first_slot_cache,
              "unreleased cache slots were reused ahead of the released slot");
  CheckMotionCache(fifth);
  CheckMotionCache(second);
  CheckMotionCache(third);
  input.Release(&fifth);
  input.Release(&second);
  input.Release(&third);
  mlvc::Check(input.Pop().eof && prepared == 5, "cached motion input did not drain at EOF");
}

void TestReadinessIncludesMotionPreparationWait() {
  const auto spec = FrameSpec();
  mlvc::app::FrameInputArena arena(spec, 1);
  std::promise<void> callback_started, release_callback;
  auto started = callback_started.get_future();
  auto release = release_callback.get_future().share();
  std::chrono::steady_clock::time_point callback_entry;
  mlvc::app::AsyncFrameInputQueue input(1, 1, {}, {}, std::nullopt, nullptr,
      spec, 8, 8, &arena, nullptr, nullptr,
      [&](const mlvc::codec::TensorData&, std::vector<uint8_t>* output) {
        callback_entry = std::chrono::steady_clock::now();
        callback_started.set_value();
        release.wait();
        output->assign(64, 0);
      });
  auto popped = std::async(std::launch::async, [&] { return input.Pop(); });
  const bool did_start = started.wait_for(2s) == std::future_status::ready;
  const bool not_published = popped.wait_for(30ms) == std::future_status::timeout;
  const auto before_release = std::chrono::steady_clock::now();
  release_callback.set_value();
  auto frame = popped.get();
  const auto source_ready_at = frame.ready_at;
  const auto stats = input.stats();
  input.Release(&frame);
  mlvc::Check(did_start && not_published, "motion preparation published an incomplete frame");
  mlvc::Check(source_ready_at != std::chrono::steady_clock::time_point{} &&
                  source_ready_at <= callback_entry,
              "readiness timestamp excluded motion preparation latency");
  const double held_ms = std::chrono::duration<double, std::milli>(
      before_release - callback_entry).count();
  mlvc::Check(held_ms >= 20 && stats.motion_prepare_ms >= held_ms &&
                  stats.motion_prepare_max_ms >= held_ms,
              "motion preparation metrics omitted time spent in the callback");
}

void TestMotionPreparationFailurePreservesCause() {
  const auto spec = FrameSpec();
  mlvc::app::FrameInputArena arena(spec, 1);
  int callbacks = 0;
  mlvc::app::AsyncFrameInputQueue input(3, 3, {}, {}, std::nullopt, nullptr,
      spec, 8, 8, &arena, nullptr, nullptr,
      [&](const mlvc::codec::TensorData&, std::vector<uint8_t>* output) {
        if (callbacks++ == 1) throw std::runtime_error("expected motion preparation failure");
        output->assign(64, 0);
      });
  auto first = input.Pop();
  CheckMotionCache(first);
  input.Release(&first);
  bool propagated = false;
  try { (void)input.Pop(); }
  catch (const std::runtime_error& error) {
    propagated = std::string(error.what()) == "expected motion preparation failure";
  }
  input.Cancel(std::make_exception_ptr(std::runtime_error("later cancellation")));
  bool preserved = false;
  try { input.RethrowIfFailed(); }
  catch (const std::runtime_error& error) {
    preserved = std::string(error.what()) == "expected motion preparation failure";
  }
  mlvc::Check(propagated && preserved && callbacks == 2,
              "motion preparation failure was lost or later frames were prepared");
}
}  // namespace

int main() {
  Watchdog watchdog;
  try {
    for (const int prefetch : {1, 2})
      for (const int frames : {0, 1, 7}) TestOrderedEof(frames, prefetch);
    TestTwoFutureFramesWhileEncodingIsBlocked();
    TestConsumerFailureCancelsBackpressuredProducer();
    TestMotionFailureDoesNotEncodeLaterFrames();
    TestCancelWakesPopAndFreeSlotWait();
    TestInputFailurePropagatesThroughProducer();
    TestSourceEofDrainsAcceptedPackets();
    TestMotionCacheUsesIndependentLeasedSlots();
    TestReadinessIncludesMotionPreparationWait();
    TestMotionPreparationFailurePreservesCause();
    std::cout << "motion prefetch order, bounds, EOF and cancellation tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_motion_prefetch failed: " << error.what() << '\n';
    return 1;
  }
}
