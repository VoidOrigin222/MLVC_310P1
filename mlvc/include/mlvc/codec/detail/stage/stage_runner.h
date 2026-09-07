#ifndef MLVC_CODEC_DETAIL_STAGE_RUNNER_H_
#define MLVC_CODEC_DETAIL_STAGE_RUNNER_H_

#include <mlvc/codec/tensor_data.h>
#include <mlvc/framework/profiler.h>
#include <mlvc/runtime/stage_runtime.h>

#include <initializer_list>
#include <string_view>

#include <mlvc/codec/detail/stage/stage_types.h>

namespace mlvc::codec {

RunOutput RunStage(mlvc::StageModelSet* models, std::string_view name,
                   std::initializer_list<StageInput> inputs, mlvc::Profiler* profiler = nullptr);
RunOutput RunIHyperPrior(mlvc::StageModelSet* models, const TensorData& y_latent,
                         RunOutput* hyper_output, mlvc::Profiler* profiler = nullptr);
RunOutput RunPHyperPrior(mlvc::StageModelSet* models, const TensorData& y_latent,
                         const TensorData& temporal_prior_context, RunOutput* hyper_output,
                         mlvc::Profiler* profiler = nullptr);

}  // namespace mlvc::codec

#endif  // MLVC_CODEC_DETAIL_STAGE_RUNNER_H_
