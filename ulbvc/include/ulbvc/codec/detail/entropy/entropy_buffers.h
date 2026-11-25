#ifndef ULBVC_CODEC_DETAIL_ENTROPY_BUFFERS_H_
#define ULBVC_CODEC_DETAIL_ENTROPY_BUFFERS_H_

#include <ulbvc/codec/detail/stage/stage_types.h>
#include <ulbvc/codec/tensor_data.h>
#include <ulbvc/core/buffer.h>
#include <ulbvc/framework/async_plan.h>
#include <ulbvc/framework/profiler.h>

#include <cstddef>
#include <string>

namespace ulbvc::codec {

std::size_t EntropyInputBytes(const TensorData& z_symbols, const RunOutput& spatial,
                              int part_count);
AsyncEntropyInputs CopyEntropyInputsToPinned(const TensorData& z_symbols, const RunOutput& spatial,
                                             int part_count, ulbvc::PinnedHostBuffer* buffer);
void RecordPinnedCopy(ulbvc::AsyncFramePlan* plan, ulbvc::Profiler* profiler, const std::string& name,
                      std::size_t bytes, const PinnedCopySpan& copy_span,
                      const std::string& timing_source);

}  // namespace ulbvc::codec

#endif  // ULBVC_CODEC_DETAIL_ENTROPY_BUFFERS_H_
