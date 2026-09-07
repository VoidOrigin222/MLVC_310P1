#ifndef MLVC_CODEC_DETAIL_ENTROPY_BUFFERS_H_
#define MLVC_CODEC_DETAIL_ENTROPY_BUFFERS_H_

#include <mlvc/codec/tensor_data.h>
#include <mlvc/core/buffer.h>
#include <mlvc/framework/async_plan.h>
#include <mlvc/framework/profiler.h>

#include <cstddef>
#include <string>

#include <mlvc/codec/detail/stage/stage_types.h>

namespace mlvc::codec {

std::size_t EntropyInputBytes(const TensorData& z_symbols, const RunOutput& spatial,
                              int part_count);
AsyncEntropyInputs CopyEntropyInputsToPinned(const TensorData& z_symbols, const RunOutput& spatial,
                                             int part_count, mlvc::PinnedHostBuffer* buffer);
void RecordPinnedCopy(mlvc::AsyncFramePlan* plan, mlvc::Profiler* profiler, const std::string& name,
                      std::size_t bytes, const PinnedCopySpan& copy_span,
                      const std::string& timing_source);

}  // namespace mlvc::codec

#endif  // MLVC_CODEC_DETAIL_ENTROPY_BUFFERS_H_
