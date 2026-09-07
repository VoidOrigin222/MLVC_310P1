#ifndef MLVC_FRAMEWORK_CODEC_GRAPH_EXECUTOR_H_
#define MLVC_FRAMEWORK_CODEC_GRAPH_EXECUTOR_H_

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

#include "mlvc/framework/profiler.h"
namespace mlvc {

enum class CodecGraphNodeType {
  kAclStage,
  kDeviceCopy,
  kCpuEntropy,
  kBitstreamIo,
  kVideoIo,
};

const char* CodecGraphNodeTypeName(CodecGraphNodeType type);

class CodecFramePacket {
 public:
  explicit CodecFramePacket(int slot_index);

  int frame_index() const { return frame_index_; }
  int slot_index() const { return slot_index_; }
  bool device_ready_recorded() const { return device_ready_recorded_; }

  void RecordDeviceReady(void* stream);
  void WaitDeviceReady(void* stream) const;

 private:
  friend class BoundedFramePacketPool;

  void Reset(int frame_index);

  int slot_index_ = 0;
  int frame_index_ = -1;
  bool device_ready_recorded_ = false;
};

class BoundedFramePacketPool {
 public:
  explicit BoundedFramePacketPool(std::size_t capacity);

  CodecFramePacket* Acquire(int frame_index, Profiler* profiler);
  void Release(CodecFramePacket* packet, Profiler* profiler);

  std::size_t capacity() const { return packets_.size(); }
  std::size_t active_count() const;

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  std::vector<CodecFramePacket> packets_;
  std::vector<bool> in_use_;
  std::size_t active_count_ = 0;
};

class CodecGraphExecutor {
 public:
  explicit CodecGraphExecutor(std::size_t packet_pool_capacity = 2);

  std::size_t packet_pool_capacity() const { return packet_pool_.capacity(); }

  void RecordTemplate(Profiler* profiler) const;
  CodecFramePacket* AcquireFramePacket(int frame_index, Profiler* profiler);
  void ReleaseFramePacket(CodecFramePacket* packet, Profiler* profiler);
  void RecordNode(Profiler* profiler, CodecGraphNodeType type, const std::string& name,
                  const std::string& detail, std::chrono::steady_clock::time_point begin,
                  std::chrono::steady_clock::time_point end, int frame_index = -1) const;

 private:
  BoundedFramePacketPool packet_pool_;
};

class ScopedCodecGraphNode {
 public:
  ScopedCodecGraphNode(const CodecGraphExecutor* executor, Profiler* profiler,
                       CodecGraphNodeType type, std::string name, std::string detail,
                       int frame_index = -1);
  ~ScopedCodecGraphNode();

  ScopedCodecGraphNode(const ScopedCodecGraphNode&) = delete;
  ScopedCodecGraphNode& operator=(const ScopedCodecGraphNode&) = delete;

 private:
  const CodecGraphExecutor* executor_ = nullptr;
  Profiler* profiler_ = nullptr;
  CodecGraphNodeType type_ = CodecGraphNodeType::kAclStage;
  std::string name_;
  std::string detail_;
  int frame_index_ = -1;
  std::chrono::steady_clock::time_point begin_;
};

}  // namespace mlvc

#endif  // MLVC_FRAMEWORK_CODEC_GRAPH_EXECUTOR_H_
