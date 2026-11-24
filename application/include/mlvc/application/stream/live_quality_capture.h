#ifndef MLVC_APPLICATION_STREAM_LIVE_QUALITY_CAPTURE_H_
#define MLVC_APPLICATION_STREAM_LIVE_QUALITY_CAPTURE_H_

#include <mlvc/codec/tensor_data.h>
#include <mlvc/core/tensor_handle.h>

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace mlvc::app {

// Calibration is opt-in through a bounded request in tmpfs. No host copy is
// performed during normal streaming. Frame IDs, rather than preview arrival
// times, identify the matching input and reconstructed pictures.
class LiveQualityCapture {
 public:
  LiveQualityCapture() : worker_([this] { WriteLoop(); }) {}
  ~LiveQualityCapture() {
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
    condition_.notify_all();
    worker_.join();
  }

  bool Wants(int frame) {
    if (frame % 15 == 0) ReadRequest();
    if (run_.empty() || frame < first_ || frame >= first_ + count_) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size() < 3;
  }

  void Reference(int frame, const codec::TensorData& tensor,
                 const std::vector<uint8_t>* nv12) {
    if (!Wants(frame)) return;
    const auto view = tensor.View();
    if (nv12 == nullptr || nv12->size() != 1920u * 1080u * 3u / 2u ||
        tensor.dtype != DataType::kFloat16 || tensor.shape.rank() != 4 ||
        tensor.shape.dim(2) != 1088 || tensor.shape.dim(3) != 1920 ||
        view.location() == MemoryLocation::kAcl) return;
    Put(frame, "source.nv12", nv12->data(), nv12->size());
    Put(frame, "reference.yf16", view.data(), 1920u * 1088u * 2u);
  }

  void Reconstruction(int frame, const codec::TensorData& tensor, TensorHandle* handle) {
    if (!Wants(frame)) return;
    if (tensor.dtype != DataType::kFloat16 || tensor.shape.rank() != 4 ||
        tensor.shape.dim(2) != 1088 || tensor.shape.dim(3) != 1920) return;
    if (handle != nullptr) {
      handle->MaterializeToCpu("psnr_calibration");
      Put(frame, "reconstruction.yf16", handle->CpuView().data(), 1920u * 1088u * 2u);
    } else if (tensor.View().location() != MemoryLocation::kAcl) {
      Put(frame, "reconstruction.yf16", tensor.View().data(), 1920u * 1088u * 2u);
    }
  }

 private:
  struct Item { std::filesystem::path path; std::vector<uint8_t> bytes; };
  void ReadRequest() {
    std::ifstream input("/dev/shm/mlvc-psnr/request.txt");
    std::string run; int first = -1, count = 0;
    if (!(input >> run >> first >> count) || run.size() != 32 || first < 0 ||
        count < 16 || count > 96 ||
        !std::all_of(run.begin(), run.end(), [](char c) {
          return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        })) return;
    run_ = run; first_ = first; count_ = count;
  }
  void Put(int frame, const char* suffix, const void* data, std::size_t bytes) {
    if (data == nullptr) return;
    Item item;
    item.path = std::filesystem::path("/dev/shm/mlvc-psnr") / run_ /
                ("frame_" + std::to_string(frame) + "." + suffix);
    item.bytes.resize(bytes);
    std::memcpy(item.bytes.data(), data, bytes);
    { std::lock_guard<std::mutex> lock(mutex_); queue_.push_back(std::move(item)); }
    condition_.notify_one();
  }
  void WriteLoop() {
    for (;;) {
      Item item;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
        if (queue_.empty() && stopping_) return;
        item = std::move(queue_.front()); queue_.pop_front();
      }
      try {
        std::filesystem::create_directories(item.path.parent_path());
        const auto temporary = item.path.string() + ".tmp";
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(item.bytes.data()), item.bytes.size());
        output.close();
        if (output.good()) std::filesystem::rename(temporary, item.path);
      } catch (const std::exception&) {
        // A failed calibration must not terminate the live codec pipeline.
      }
    }
  }
  std::string run_;
  int first_ = -1, count_ = 0;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<Item> queue_;
  bool stopping_ = false;
  std::thread worker_;
};
}  // namespace mlvc::app
#endif
