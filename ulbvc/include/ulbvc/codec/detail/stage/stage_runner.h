#ifndef ULBVC_CODEC_DETAIL_STAGE_RUNNER_H_
#define ULBVC_CODEC_DETAIL_STAGE_RUNNER_H_

#include <ulbvc/codec/detail/stage/stage_types.h>
#include <ulbvc/codec/tensor_data.h>
#include <ulbvc/framework/profiler.h>
#include <ulbvc/runtime/stage_runtime.h>

#include <initializer_list>
#include <string_view>

namespace ulbvc::codec {

RunOutput RunStage(ulbvc::StageModelSet* models, std::string_view name,
                   std::initializer_list<StageInput> inputs, ulbvc::Profiler* profiler = nullptr);
RunOutput RunIHyperPrior(ulbvc::StageModelSet* models, const TensorData& y_latent,
                         RunOutput* hyper_output, ulbvc::Profiler* profiler = nullptr);
RunOutput RunPHyperPrior(ulbvc::StageModelSet* models, const TensorData& y_latent,
                         const TensorData& temporal_prior_context, RunOutput* hyper_output,
                         ulbvc::Profiler* profiler = nullptr);

}  // namespace ulbvc::codec

#endif  // ULBVC_CODEC_DETAIL_STAGE_RUNNER_H_
