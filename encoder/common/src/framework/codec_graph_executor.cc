#include "mlvc/framework/codec_graph_executor.h"

#include <utility>

#include "mlvc/core/status.h"

namespace mlvc {

const char* CodecGraphNodeTypeName(CodecGraphNodeType type) {
  switch (type) {
    case CodecGraphNodeType::kAclStage:
      return "acl_stage";
    case CodecGraphNodeType::kDeviceCopy:
      return "device_copy";
    case CodecGraphNodeType::kCpuEntropy:
      return "cpu_entropy";
    case CodecGraphNodeType::kBitstreamIo:
      return "bitstream_io";
    case CodecGraphNodeType::kVideoIo:
      return "video_io";
  }
  return "acl_stage";
}

CodecFramePacket::CodecFramePacket(int slot_index) : slot_index_(slot_index) {}

void CodecFramePacket::Reset(int frame_index) {
  frame_index_ = frame_index;
  device_ready_recorded_ = false;
}

void CodecFramePacket::RecordDeviceReady(void*) { device_ready_recorded_ = true; }

void CodecFramePacket::WaitDeviceReady(void*) const {}

BoundedFramePacketPool::BoundedFramePacketPool(std::size_t capacity) {
  Check(capacity > 0, "frame packet pool capacity must be positive");
  packets_.reserve(capacity);
  in_use_.resize(capacity, false);
  for (std::size_t i = 0; i < capacity; ++i) {
    packets_.emplace_back(static_cast<int>(i));
  }
}

CodecFramePacket* BoundedFramePacketPool::Acquire(int frame_index, Profiler* profiler) {
  const auto begin = std::chrono::steady_clock::now();
  std::unique_lock<std::mutex> lock(mutex_);
  condition_.wait(lock, [this] { return active_count_ < packets_.size(); });
  std::size_t slot = packets_.size();
  for (std::size_t i = 0; i < in_use_.size(); ++i) {
    if (!in_use_[i]) {
      slot = i;
      break;
    }
  }
  Check(slot < packets_.size(), "frame packet pool accounting mismatch");
  in_use_[slot] = true;
  ++active_count_;
  packets_[slot].Reset(frame_index);
  const std::size_t active = active_count_;
  lock.unlock();
  const auto end = std::chrono::steady_clock::now();
  if (profiler != nullptr) {
    profiler->AddEventWithArgs(
        "pipeline.packet.acquire", "pipeline", "pipeline", profiler->StartMs(begin),
        profiler->DurationMs(begin, end),
        {Profiler::Arg("frame_index", frame_index), Profiler::Arg("slot", static_cast<int>(slot)),
         Profiler::Arg("capacity", static_cast<int>(packets_.size())),
         Profiler::Arg("active", static_cast<int>(active)),
         Profiler::Arg("node_type", "resource_pool")});
  }
  return &packets_[slot];
}

void BoundedFramePacketPool::Release(CodecFramePacket* packet, Profiler* profiler) {
  Check(packet != nullptr, "frame packet release requires a packet");
  const auto begin = std::chrono::steady_clock::now();
  const int slot_index = packet->slot_index();
  int frame_index = packet->frame_index();
  std::size_t active = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    Check(slot_index >= 0 && slot_index < static_cast<int>(in_use_.size()),
          "invalid frame packet slot");
    Check(in_use_[static_cast<std::size_t>(slot_index)], "frame packet was not acquired");
    in_use_[static_cast<std::size_t>(slot_index)] = false;
    --active_count_;
    active = active_count_;
  }
  condition_.notify_one();
  const auto end = std::chrono::steady_clock::now();
  if (profiler != nullptr) {
    profiler->AddEventWithArgs(
        "pipeline.packet.release", "pipeline", "pipeline", profiler->StartMs(begin),
        profiler->DurationMs(begin, end),
        {Profiler::Arg("frame_index", frame_index), Profiler::Arg("slot", slot_index),
         Profiler::Arg("capacity", static_cast<int>(packets_.size())),
         Profiler::Arg("active", static_cast<int>(active)),
         Profiler::Arg("node_type", "resource_pool")});
  }
}

std::size_t BoundedFramePacketPool::active_count() const { return active_count_; }

CodecGraphExecutor::CodecGraphExecutor(std::size_t packet_pool_capacity)
    : packet_pool_(packet_pool_capacity) {}

void CodecGraphExecutor::RecordTemplate(Profiler* profiler) const {
  if (profiler == nullptr) {
    return;
  }
  profiler->AddEventWithArgs(
      "pipeline.graph_template", "pipeline", "pipeline", 0.0, 0.0,
      {Profiler::Arg("scheduler", "dag"), Profiler::Arg("node_type", "graph_template"),
       Profiler::Arg("node_type_0", CodecGraphNodeTypeName(CodecGraphNodeType::kAclStage)),
       Profiler::Arg("node_type_1", CodecGraphNodeTypeName(CodecGraphNodeType::kDeviceCopy)),
       Profiler::Arg("node_type_2", CodecGraphNodeTypeName(CodecGraphNodeType::kCpuEntropy)),
       Profiler::Arg("node_type_3", CodecGraphNodeTypeName(CodecGraphNodeType::kBitstreamIo)),
       Profiler::Arg("node_type_4", CodecGraphNodeTypeName(CodecGraphNodeType::kVideoIo)),
       Profiler::Arg("packet_pool_capacity", static_cast<int>(packet_pool_.capacity()))});
}

CodecFramePacket* CodecGraphExecutor::AcquireFramePacket(int frame_index, Profiler* profiler) {
  return packet_pool_.Acquire(frame_index, profiler);
}

void CodecGraphExecutor::ReleaseFramePacket(CodecFramePacket* packet, Profiler* profiler) {
  packet_pool_.Release(packet, profiler);
}

void CodecGraphExecutor::RecordNode(Profiler* profiler, CodecGraphNodeType type,
                                    const std::string& name, const std::string& detail,
                                    std::chrono::steady_clock::time_point begin,
                                    std::chrono::steady_clock::time_point end,
                                    int frame_index) const {
  if (profiler == nullptr) {
    return;
  }
  profiler->AddEventWithArgs(
      "pipeline.node." + std::string(CodecGraphNodeTypeName(type)) + "." + name, "pipeline",
      "pipeline", profiler->StartMs(begin), profiler->DurationMs(begin, end),
      {Profiler::Arg("scheduler", "dag"), Profiler::Arg("node_type", CodecGraphNodeTypeName(type)),
       Profiler::Arg("name", name), Profiler::Arg("detail", detail),
       Profiler::Arg("frame_index", frame_index)});
}

ScopedCodecGraphNode::ScopedCodecGraphNode(const CodecGraphExecutor* executor, Profiler* profiler,
                                           CodecGraphNodeType type, std::string name,
                                           std::string detail, int frame_index)
    : executor_(executor),
      profiler_(profiler),
      type_(type),
      name_(std::move(name)),
      detail_(std::move(detail)),
      frame_index_(frame_index),
      begin_(std::chrono::steady_clock::now()) {}

ScopedCodecGraphNode::~ScopedCodecGraphNode() {
  if (executor_ == nullptr || profiler_ == nullptr) {
    return;
  }
  executor_->RecordNode(profiler_, type_, name_, detail_, begin_, std::chrono::steady_clock::now(),
                        frame_index_);
}

}  // namespace mlvc
