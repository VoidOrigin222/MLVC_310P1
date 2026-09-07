#include <mlvc/codec/detail/frame/encoder.h>

#include <chrono>
#include <cstdint>
#include <future>

#include <mlvc/codec/detail/profile/allocation_tracking.h>
#include <mlvc/codec/detail/profile/codec_profile.h>
#include <mlvc/codec/detail/stage/constants.h>
#include <mlvc/codec/detail/entropy/entropy_buffers.h>
#include <mlvc/codec/detail/frame/reference.h>
#include <mlvc/codec/detail/stage/stage_runner.h>
#include <mlvc/codec/detail/stage/stage_runtime_state.h>
#include <mlvc/codec/detail/tensor/tensor_utils.h>
#include "mlvc/core/status.h"
#include "mlvc/framework/profile_range.h"

namespace mlvc::codec {

namespace {

void CompactPrebuiltYIndexes(const TensorBytesView& combined, const TensorBytesView& keep_mask,
                             std::vector<int16_t>* output) {
  MLVC_PROFILE_RANGE_FUNCTION();
  mlvc::Check(output != nullptr, "prebuilt Y index output is required");
  mlvc::Check(combined.dtype == mlvc::DataType::kInt16, "prebuilt Y indexes must be int16");
  mlvc::Check(keep_mask.dtype == mlvc::DataType::kUInt8, "prebuilt Y keep mask must be uint8");
  mlvc::Check(combined.Elements() == keep_mask.Elements(),
              "prebuilt Y index and keep mask size mismatch");
  const auto* combined_values = reinterpret_cast<const int16_t*>(combined.bytes);
  const auto* keep_values = reinterpret_cast<const uint8_t*>(keep_mask.bytes);
  output->clear();
  output->reserve(combined.Elements());
  for (std::size_t index = 0; index < combined.Elements(); ++index) {
    if (keep_values[index] != 0) {
      output->push_back(combined_values[index]);
    }
  }
}

std::size_t AsyncEntropyPinnedBytes(const AsyncEntropyInputs& inputs) {
  std::size_t bytes = inputs.z_symbols.byte_count;
  for (int part = 0; part < inputs.part_count; ++part) {
    if (inputs.y_indexes_prebuilt) {
      bytes += inputs.y_combined_indexes[part].byte_count + inputs.y_keep_masks[part].byte_count;
    } else {
      bytes += inputs.y_symbols[part].byte_count + inputs.scales[part].byte_count;
    }
  }
  return bytes;
}

}  // namespace

void BuildIEntropyPayloadFromPinnedInto(mlvc::EntropyEncoder* entropy,
                                        const mlvc::RuntimeSidecar& sidecar,
                                        const AsyncEntropyInputs& inputs,
                                        AsyncEntropyScratch* scratch,
                                        std::vector<uint8_t>* payload) {
  MLVC_PROFILE_RANGE_FUNCTION();
  mlvc::Check(entropy != nullptr, "I entropy encoder is required");
  mlvc::Check(scratch != nullptr, "I async entropy scratch is required");
  mlvc::Check(payload != nullptr, "I entropy payload output is required");
  {
    ScopedRepositoryAllocationTrackingPause allocation_pause;
    entropy->Reset();
  }
  const int z_height = static_cast<int>(inputs.z_symbols.shape.dim(2));
  const int z_width = static_cast<int>(inputs.z_symbols.shape.dim(3));
  TensorToInt8Symbols(inputs.z_symbols, &scratch->z_symbol_scratch);
  {
    ScopedRepositoryAllocationTrackingPause allocation_pause;
    entropy->EncodeZView(scratch->z_symbol_scratch, kBaseQp, z_height, z_width);
  }
  for (int part = 0; part < inputs.part_count; ++part) {
    if (inputs.y_indexes_prebuilt) {
      CompactPrebuiltYIndexes(inputs.y_combined_indexes[part], inputs.y_keep_masks[part],
                              &scratch->y_index_scratch);
      {
        ScopedRepositoryAllocationTrackingPause allocation_pause;
        entropy->EncodeYIndexesView(scratch->y_index_scratch);
      }
    } else {
      TensorToInt8Symbols(inputs.y_symbols[part], &scratch->y_symbol_scratch);
      Fp16ToFloat(inputs.scales[part], &scratch->scale_scratch);
      {
        ScopedRepositoryAllocationTrackingPause allocation_pause;
        entropy->EncodeY(scratch->y_symbol_scratch, scratch->scale_scratch,
                         sidecar.python_fast_force_zero_thres());
      }
    }
  }
  {
    ScopedRepositoryAllocationTrackingPause allocation_pause;
    entropy->FlushInto(payload);
  }
}

void EncodeIAsync(mlvc::StageModelSet* models, const mlvc::RuntimeSidecar& sidecar,
                  mlvc::EntropyEncoder* entropy, mlvc::AsyncFramePlan* plan, QScaleCache* q_scales,
                  const TensorData& frame, TensorData* x_hat, mlvc::TensorHandle* x_hat_handle,
                  bool /*use_two_parts*/, AsyncEntropyScratch* entropy_scratch,
                  std::vector<uint8_t>* payload_output, mlvc::Profiler* profiler) {
  MLVC_PROFILE_RANGE_FUNCTION();
  mlvc::Check(entropy != nullptr, "I async entropy encoder is required");
  mlvc::Check(entropy_scratch != nullptr, "I async entropy scratch is required");
  mlvc::Check(payload_output != nullptr, "I async payload output is required");
  ScopedProfileTimer frame_timer(profiler, "encode.i.async_total");
  ScopedAsyncEncodeDeviceOnlyMirrorSkip mirror_skip(g_skip_async_entropy_cpu_mirror);

  const TensorData& encoder_q_step = q_scales->IEncoder();
  RunOutput encoder =
      RunStage(models, "i_encoder",
               {TensorInput("current_frame", frame),
                TensorOrHandleInput("encoder_q_step", encoder_q_step, q_scales->IEncoderHandle())},
               profiler);

  RunOutput hyper;
  RunOutput prior = RunIHyperPrior(models, encoder.At("y_latent"), &hyper, profiler);
  RunOutput spatial = RunStage(models, "i_spatial_prior",
                               {{"y_latent", &encoder.At("y_latent")},
                                {"fused_prior_params", &prior.At("fused_prior_params")}},
                               profiler);

  std::size_t entropy_ready_bytes = EntropyInputBytes(hyper.At("z_symbols"), spatial, 4);
  PinnedCopySpan copy_span;
  AsyncEntropyInputs entropy_inputs;
  std::string timing_source = "host_memcpy";
  if (g_stage_output_workspace != nullptr &&
      g_stage_output_workspace->CanCopyStageTensorFromDevice("i_hyper_encoder", "z_symbols")) {
    entropy_inputs = CopyEntropyInputsToPinnedFromDevice(
        "i_hyper_encoder", "i_spatial_prior", 4, &plan->PinnedSlot(0),
        sidecar.python_fast_force_zero_thres(), &copy_span, profiler);
    entropy_ready_bytes = AsyncEntropyPinnedBytes(entropy_inputs);
    timing_source = "device_d2h";
  } else {
    const auto host_begin = std::chrono::steady_clock::now();
    entropy_inputs =
        CopyEntropyInputsToPinned(hyper.At("z_symbols"), spatial, 4, &plan->PinnedSlot(0));
    const auto host_end = std::chrono::steady_clock::now();
    if (profiler != nullptr) {
      copy_span.start_ms = profiler->StartMs(host_begin);
      copy_span.duration_ms = profiler->DurationMs(host_begin, host_end);
    }
  }
  RecordPinnedCopy(plan, profiler, "i_entropy_inputs_pinned_handoff", entropy_ready_bytes,
                   copy_span, timing_source);
  std::chrono::steady_clock::time_point entropy_worker_begin;
  std::chrono::steady_clock::time_point entropy_worker_end;
  auto future = plan->entropy_worker().SubmitVoid([entropy, &sidecar, entropy_inputs,
                                                   entropy_scratch, payload_output,
                                                   &entropy_worker_begin, &entropy_worker_end]() {
    MLVC_PROFILE_RANGE("EncodeIAsync.entropy_worker");
    entropy_worker_begin = std::chrono::steady_clock::now();
    BuildIEntropyPayloadFromPinnedInto(entropy, sidecar, entropy_inputs, entropy_scratch,
                                       payload_output);
    entropy_worker_end = std::chrono::steady_clock::now();
  });

  const TensorData& decoder_q_step = q_scales->IDecoder(kBaseQp);
  RunOutput decoder = RunStage(
      models, "i_decoder",
      {TensorInput("reconstructed_y_latent", spatial.At("reconstructed_y_latent")),
       TensorOrHandleInput("decoder_q_step", decoder_q_step, q_scales->IDecoderHandle(kBaseQp))},
      profiler);
  (void)CloneRunOutputHandle(decoder, "x_hat", x_hat_handle, "reference_frame", profiler);
  CloneTensorInto(decoder.At("x_hat"), x_hat);
  ScopedProfileTimer wait_timer(profiler, "entropy.i.async_wait");
  future.get();
  if (profiler != nullptr) {
    AddProfileDuration(profiler, "entropy.i.async_encode_worker", entropy_worker_begin,
                       entropy_worker_end);
    if (g_codec_graph_executor != nullptr) {
      g_codec_graph_executor->RecordNode(profiler, mlvc::CodecGraphNodeType::kCpuEntropy,
                                         "i_entropy_encode_worker", "EntropyWorker::SubmitVoid",
                                         entropy_worker_begin, entropy_worker_end);
    }
  }
}

void BuildPEntropyPayloadFromPinnedInto(mlvc::EntropyEncoder* entropy,
                                        const mlvc::RuntimeSidecar& sidecar,
                                        const AsyncEntropyInputs& inputs, int qp,
                                        AsyncEntropyScratch* scratch,
                                        std::vector<uint8_t>* payload) {
  MLVC_PROFILE_RANGE_FUNCTION();
  mlvc::Check(entropy != nullptr, "P entropy encoder is required");
  mlvc::Check(scratch != nullptr, "P async entropy scratch is required");
  mlvc::Check(payload != nullptr, "P entropy payload output is required");
  {
    ScopedRepositoryAllocationTrackingPause allocation_pause;
    entropy->Reset();
  }
  const int z_height = static_cast<int>(inputs.z_symbols.shape.dim(2));
  const int z_width = static_cast<int>(inputs.z_symbols.shape.dim(3));
  TensorToInt8Symbols(inputs.z_symbols, &scratch->z_symbol_scratch);
  {
    ScopedRepositoryAllocationTrackingPause allocation_pause;
    entropy->EncodeZView(scratch->z_symbol_scratch, qp, z_height, z_width);
  }
  for (int part = 0; part < inputs.part_count; ++part) {
    if (inputs.y_indexes_prebuilt) {
      CompactPrebuiltYIndexes(inputs.y_combined_indexes[part], inputs.y_keep_masks[part],
                              &scratch->y_index_scratch);
      {
        ScopedRepositoryAllocationTrackingPause allocation_pause;
        entropy->EncodeYIndexesView(scratch->y_index_scratch);
      }
    } else {
      TensorToInt8Symbols(inputs.y_symbols[part], &scratch->y_symbol_scratch);
      Fp16ToFloat(inputs.scales[part], &scratch->scale_scratch);
      {
        ScopedRepositoryAllocationTrackingPause allocation_pause;
        entropy->EncodeY(scratch->y_symbol_scratch, scratch->scale_scratch,
                         sidecar.python_fast_force_zero_thres());
      }
    }
  }
  {
    ScopedRepositoryAllocationTrackingPause allocation_pause;
    entropy->FlushInto(payload);
  }
}

void EncodePAsync(mlvc::StageModelSet* models, const mlvc::RuntimeSidecar& sidecar,
                  mlvc::EntropyEncoder* entropy, mlvc::AsyncFramePlan* plan, QScaleCache* q_scales,
                  const TensorData& frame, const ReferenceState& state, int qp,
                  bool /*use_two_parts*/, AsyncEntropyScratch* entropy_scratch,
                  TensorData* decoded_feature, mlvc::TensorHandle* decoded_feature_handle,
                  CodecScratch* scratch, std::vector<uint8_t>* payload_output,
                  mlvc::Profiler* profiler) {
  MLVC_PROFILE_RANGE_FUNCTION();
  mlvc::Check(entropy != nullptr, "P async entropy encoder is required");
  mlvc::Check(entropy_scratch != nullptr, "P async entropy scratch is required");
  mlvc::Check(decoded_feature != nullptr, "P async decoded feature output is required");
  mlvc::Check(scratch != nullptr, "codec scratch is required");
  mlvc::Check(payload_output != nullptr, "P async payload output is required");
  ScopedProfileTimer total_timer(profiler, "encode.p.async_total");
  ScopedAsyncEncodeDeviceOnlyMirrorSkip mirror_skip(g_skip_async_entropy_cpu_mirror);
  ReferenceFeatureBinding reference_feature =
      ReferenceFeature(models, state, &scratch->reference_feature, profiler);

  const TensorData& feature_q_step = q_scales->PFeature(qp);
  RunOutput contexts =
      reference_feature.handle != nullptr
          ? RunStage(models, "p_reference_context",
                     {StageInput{"reference_feature", nullptr, reference_feature.handle},
                      TensorOrHandleInput("feature_q_step", feature_q_step,
                                          q_scales->PFeatureHandle(qp))},
                     profiler)
          : RunStage(models, "p_reference_context",
                     {StageInput{"reference_feature", reference_feature.tensor, nullptr},
                      TensorOrHandleInput("feature_q_step", feature_q_step,
                                          q_scales->PFeatureHandle(qp))},
                     profiler);

  const TensorData& encoder_q_step = q_scales->PEncoder(qp);
  RunOutput analysis = RunStage(
      models, "p_analysis_encoder",
      {TensorInput("current_frame", frame),
       TensorInput("reference_context", contexts.At("reference_context")),
       TensorOrHandleInput("encoder_q_step", encoder_q_step, q_scales->PEncoderHandle(qp))},
      profiler);
  RunOutput hyper;
  RunOutput prior = RunPHyperPrior(models, analysis.At("y_latent"),
                                   contexts.At("temporal_prior_context"), &hyper, profiler);
  RunOutput spatial = RunStage(models, "p_spatial_prior",
                               {{"y_latent", &analysis.At("y_latent")},
                                {"fused_prior_params", &prior.At("fused_prior_params")}},
                               profiler);

  std::size_t entropy_ready_bytes = EntropyInputBytes(hyper.At("z_symbols"), spatial, 2);
  PinnedCopySpan copy_span;
  AsyncEntropyInputs entropy_inputs;
  std::string timing_source = "host_memcpy";
  if (g_stage_output_workspace != nullptr &&
      g_stage_output_workspace->CanCopyStageTensorFromDevice("p_hyper_encoder", "z_symbols")) {
    entropy_inputs = CopyEntropyInputsToPinnedFromDevice(
        "p_hyper_encoder", "p_spatial_prior", 2, &plan->PinnedSlot(1),
        sidecar.python_fast_force_zero_thres(), &copy_span, profiler);
    entropy_ready_bytes = AsyncEntropyPinnedBytes(entropy_inputs);
    timing_source = "device_d2h";
  } else {
    const auto host_begin = std::chrono::steady_clock::now();
    entropy_inputs =
        CopyEntropyInputsToPinned(hyper.At("z_symbols"), spatial, 2, &plan->PinnedSlot(1));
    const auto host_end = std::chrono::steady_clock::now();
    if (profiler != nullptr) {
      copy_span.start_ms = profiler->StartMs(host_begin);
      copy_span.duration_ms = profiler->DurationMs(host_begin, host_end);
    }
  }
  RecordPinnedCopy(plan, profiler, "p_entropy_inputs_pinned_handoff", entropy_ready_bytes,
                   copy_span, timing_source);
  std::chrono::steady_clock::time_point entropy_worker_begin;
  std::chrono::steady_clock::time_point entropy_worker_end;
  auto future = plan->entropy_worker().SubmitVoid([entropy, &sidecar, entropy_inputs, qp,
                                                   entropy_scratch, payload_output,
                                                   &entropy_worker_begin, &entropy_worker_end]() {
    MLVC_PROFILE_RANGE("EncodePAsync.entropy_worker");
    entropy_worker_begin = std::chrono::steady_clock::now();
    BuildPEntropyPayloadFromPinnedInto(entropy, sidecar, entropy_inputs, qp, entropy_scratch,
                                       payload_output);
    entropy_worker_end = std::chrono::steady_clock::now();
  });

  const TensorData& decoder_q_step = q_scales->PDecoder(qp);
  RunOutput synthesis = RunStage(
      models, "p_synthesis_decoder",
      {TensorInput("reconstructed_y_latent", spatial.At("reconstructed_y_latent")),
       TensorInput("reference_context", contexts.At("reference_context")),
       TensorOrHandleInput("decoder_q_step", decoder_q_step, q_scales->PDecoderHandle(qp))},
      profiler);
  (void)CloneRunOutputHandle(synthesis, "decoded_feature", decoded_feature_handle,
                             "decoded_feature", profiler);
  CloneTensorInto(synthesis.At("decoded_feature"), decoded_feature);
  ScopedProfileTimer wait_timer(profiler, "entropy.p.async_wait");
  future.get();
  if (profiler != nullptr) {
    AddProfileDuration(profiler, "entropy.p.async_encode_worker", entropy_worker_begin,
                       entropy_worker_end);
    if (g_codec_graph_executor != nullptr) {
      g_codec_graph_executor->RecordNode(profiler, mlvc::CodecGraphNodeType::kCpuEntropy,
                                         "p_entropy_encode_worker", "EntropyWorker::SubmitVoid",
                                         entropy_worker_begin, entropy_worker_end);
    }
  }
}

}  // namespace mlvc::codec
