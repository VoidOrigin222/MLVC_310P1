#ifndef ULBVC_APPLICATION_INPUT_FRAME_BUFFER_POOL_H_
#define ULBVC_APPLICATION_INPUT_FRAME_BUFFER_POOL_H_

#include <ulbvc/codec/tensor_data.h>

#include <cstddef>
#include <vector>

namespace ulbvc::app {

class FrameBufferPool {
 public:
  virtual ~FrameBufferPool() = default;
  virtual codec::TensorData& Get(std::size_t slot_index) = 0;
  virtual const codec::TensorData& Get(std::size_t slot_index) const = 0;
  virtual std::size_t capacity() const = 0;
};

class FrameInputArena final : public FrameBufferPool {
 public:
  FrameInputArena(const ulbvc::TensorSpec& frame_spec, int slot_count,
                  bool allocate_host_buffers = true);

  codec::TensorData& Get(std::size_t slot_index) override { return slots_.at(slot_index); }
  const codec::TensorData& Get(std::size_t slot_index) const override {
    return slots_.at(slot_index);
  }
  std::size_t capacity() const override { return slots_.size(); }

  std::vector<codec::TensorData>& slots() { return slots_; }
  const std::vector<codec::TensorData>& slots() const { return slots_; }

 private:
  std::vector<codec::TensorData> slots_;
};

}  // namespace ulbvc::app

#endif  // ULBVC_APPLICATION_INPUT_FRAME_BUFFER_POOL_H_
