#include <mlvc/codec/detail/entropy/entropy_buffers.h>

#include <cstring>
#include <string>

#include <mlvc/codec/detail/profile/allocation_tracking.h>
#include <mlvc/codec/detail/profile/codec_profile.h>
#include <mlvc/codec/detail/stage/constants.h>
#include "mlvc/core/status.h"
#include "mlvc/framework/profile_range.h"

namespace mlvc::codec {

std::size_t TensorBytes(const TensorData& tensor) { return tensor.bytes.size(); }

std::size_t EntropyInputBytes(const TensorData& z_symbols, const RunOutput& spatial,
                              int part_count) {
  std::size_t bytes = TensorBytes(z_symbols);
  for (int part = 0; part < part_count; ++part) {
    bytes += TensorBytes(spatial.At(kYResidualPartNames[part]));
    bytes += TensorBytes(spatial.At(kScalePartNames[part]));
  }
  return bytes;
}

void CopyTensorToPinned(const TensorData& source, mlvc::PinnedHostBuffer* buffer,
                        std::size_t* offset, TensorBytesView* view) {
  const std::size_t next_offset = *offset + source.bytes.size();
  mlvc::Check(next_offset <= buffer->bytes(), "async pinned entropy slot is too small");
  auto* destination = static_cast<uint8_t*>(buffer->data()) + *offset;
  std::memcpy(destination, source.bytes.data(), source.bytes.size());
  *view = TensorBytesView{source.shape, source.dtype, destination, source.bytes.size()};
  *offset = next_offset;
}

AsyncEntropyInputs CopyEntropyInputsToPinned(const TensorData& z_symbols, const RunOutput& spatial,
                                             int part_count, mlvc::PinnedHostBuffer* buffer) {
  MLVC_PROFILE_RANGE_FUNCTION();
  AsyncEntropyInputs inputs;
  inputs.part_count = part_count;
  std::size_t offset = 0;
  CopyTensorToPinned(z_symbols, buffer, &offset, &inputs.z_symbols);
  for (int part = 0; part < part_count; ++part) {
    CopyTensorToPinned(spatial.At(kYResidualPartNames[part]), buffer, &offset,
                       &inputs.y_symbols[part]);
    CopyTensorToPinned(spatial.At(kScalePartNames[part]), buffer, &offset, &inputs.scales[part]);
  }
  return inputs;
}

void RecordPinnedCopy(mlvc::AsyncFramePlan* plan, mlvc::Profiler* profiler, const std::string& name,
                      std::size_t bytes, const PinnedCopySpan& copy_span,
                      const std::string& timing_source) {
  ScopedRepositoryAllocationTrackingPause allocation_pause;
  if (profiler != nullptr) {
    plan->RecordCopy(name, bytes, copy_span.start_ms, copy_span.duration_ms, timing_source);
    AddProfileEventWithArgs(
        profiler, "copy.pinned_handoff." + name, "copy", "copy", copy_span.start_ms,
        copy_span.duration_ms,
        {mlvc::Profiler::Arg("name", name), mlvc::Profiler::Arg("direction", "D2H"),
         mlvc::Profiler::Arg("reason", "entropy_encode"),
         mlvc::Profiler::Arg("bytes", static_cast<uint64_t>(bytes)),
         mlvc::Profiler::Arg("timing_source", timing_source),
         mlvc::Profiler::Arg("memory", timing_source == "device_d2h"
                                           ? "device_to_pinned_cpu"
                                           : "pageable_cpu_to_pinned_cpu")});
  } else {
    plan->RecordCopy(name, bytes);
  }
}

}  // namespace mlvc::codec
