#include <mlvc/codec/detail/frame/decoder.h>

#include <acl/acl.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>

#include <mlvc/codec/detail/profile/allocation_tracking.h>
#include <mlvc/codec/detail/profile/codec_profile.h>
#include <mlvc/codec/detail/stage/constants.h>
#include <mlvc/codec/detail/prior/prior_layout.h>
#include <mlvc/codec/detail/frame/reference.h>
#include <mlvc/codec/detail/stage/stage_runner.h>
#include <mlvc/codec/detail/stage/stage_runtime_state.h>
#include <mlvc/codec/detail/tensor/tensor_utils.h>
#include "mlvc/core/status.h"
#include "mlvc/framework/profile_range.h"
#include "mlvc/runtime/decode_prior_acl.h"

namespace mlvc::codec {

namespace {

void CheckAclStatus(aclError status, const char* operation) {
  if (status != ACL_ERROR_NONE) {
    throw mlvc::Error(std::string(operation) + " failed: ret=" + std::to_string(status));
  }
}

bool CanUseAclDecodePrior() {
  return g_stage_output_workspace != nullptr &&
         g_stage_output_workspace->binding_mode() == StageOutputBindingMode::kAclMirror &&
         mlvc::DecodePriorAclAvailable();
}

void RecordAclPriorValidation(mlvc::Profiler* profiler, const char* name, bool implemented) {
  if (!g_validate_acl_decode_prior || profiler == nullptr) {
    return;
  }
  profiler->AddEventWithArgs(
      "acl_prior.validation." + std::string(name), "acl_prior", "acl_prior", 0.0, 0.0,
      {mlvc::Profiler::Arg("kernel", name), mlvc::Profiler::BoolArg("implemented", implemented)});
}

void EnsureOutputAclBuffers(PriorScratch* scratch) {
  const std::size_t bytes =
      scratch->latent_shape.NumElements() * mlvc::ElementSize(mlvc::DataType::kFloat16);
  if (scratch->output_handle.bytes() != bytes) {
    scratch->output_handle.Reset(scratch->latent_shape, mlvc::DataType::kFloat16);
    scratch->output_handle.UseOwnedCpuBuffer(false);
    scratch->output_handle.AllocateAclBuffer(false);
  }
  if (scratch->reconstructed_fp16_handle.bytes() != bytes) {
    scratch->reconstructed_fp16_handle.Reset(scratch->latent_shape, mlvc::DataType::kFloat16);
    scratch->reconstructed_fp16_handle.UseOwnedCpuBuffer(false);
    scratch->reconstructed_fp16_handle.AllocateAclBuffer(false);
  }
}

void UploadYSymbols(const std::vector<int8_t>& symbols, PriorScratch* scratch) {
  const std::size_t bytes = symbols.size() * sizeof(int8_t);
  if (scratch->y_symbols_acl.bytes() < bytes) {
    scratch->y_symbols_acl.Allocate(bytes);
  }
  CheckAclStatus(aclrtMemcpy(scratch->y_symbols_acl.data(), bytes, symbols.data(), bytes,
                             ACL_MEMCPY_HOST_TO_DEVICE),
                 "aclrtMemcpy decode prior y symbols H2D");
}

void SinglePartToCpuFromAcl(const mlvc::TensorHandle* scale_params_handle, int parts,
                            int mask_index, int channels, int height, int width,
                            PriorScratch* scratch, mlvc::Profiler* profiler) {
  mlvc::Check(scale_params_handle != nullptr && scale_params_handle->has_acl_buffer() &&
                  scale_params_handle->acl_valid(),
              "decode prior single_part requires ACL-resident scale params");
  mlvc::Check(scale_params_handle->dtype() == mlvc::DataType::kFloat16,
              "decode prior single_part currently expects FP16 scale params");
  const int part_channels = channels / parts;
  const std::size_t elements = static_cast<std::size_t>(part_channels) * height * width;
  const std::size_t bytes = elements * mlvc::ElementSize(mlvc::DataType::kFloat16);
  if (scratch->part_scales_acl.bytes() < bytes) {
    scratch->part_scales_acl.Allocate(bytes);
  }

  mlvc::DecodePriorSinglePartAcl(scale_params_handle->AclView().data(),
                                 scale_params_handle->dtype(), parts, mask_index, channels, height,
                                 width, scratch->part_scales_acl.data(), g_acl_user_compute_stream,
                                 profiler);

  scratch->part_scales_fp16.resize(elements);
  const auto begin = std::chrono::steady_clock::now();
  CheckAclStatus(aclrtMemcpy(scratch->part_scales_fp16.data(), bytes,
                             scratch->part_scales_acl.data(), bytes, ACL_MEMCPY_DEVICE_TO_HOST),
                 "aclrtMemcpy decode prior single_part D2H");
  const auto end = std::chrono::steady_clock::now();
  scratch->part_scales.resize(elements);
  for (std::size_t i = 0; i < elements; ++i) {
    scratch->part_scales[i] = HalfBitsToFloat(scratch->part_scales_fp16[i]);
  }
  if (profiler != nullptr) {
    AddProfileEventWithArgs(
        profiler,
        std::string("copy.d2h.acl_prior.") + (parts == 4 ? "single_part4x" : "single_part2x"),
        "copy", "copy", profiler->StartMs(begin), profiler->DurationMs(begin, end),
        {mlvc::Profiler::Arg("kernel", parts == 4 ? "single_part4x" : "single_part2x"),
         mlvc::Profiler::Arg("direction", "D2H"),
         mlvc::Profiler::Arg("reason", "entropy_decode_part_scales"),
         mlvc::Profiler::Arg("bytes", static_cast<uint64_t>(bytes)),
         mlvc::Profiler::Arg("elements", static_cast<uint64_t>(elements))});
  }
  RecordAclPriorValidation(profiler, parts == 4 ? "single_part4x" : "single_part2x", true);
}

void RestoreYToAcl(const std::vector<int8_t>& y_symbols, const mlvc::TensorHandle* means_handle,
                   int parts, int mask_index, int channels, int height, int width,
                   PriorScratch* scratch, mlvc::Profiler* profiler) {
  mlvc::Check(
      means_handle != nullptr && means_handle->has_acl_buffer() && means_handle->acl_valid(),
      "decode prior restore_y requires ACL-resident means");
  UploadYSymbols(y_symbols, scratch);
  EnsureOutputAclBuffers(scratch);
  if (mask_index == 0) {
    CheckAclStatus(aclrtMemset(scratch->reconstructed_fp16_handle.AclView().data(),
                               scratch->reconstructed_fp16_handle.bytes(), 0,
                               scratch->reconstructed_fp16_handle.bytes()),
                   "aclrtMemset decode prior reconstructed");
  } else if (!scratch->reconstructed_fp16_handle.acl_valid()) {
    FloatToFp16Tensor(scratch->reconstructed, scratch->latent_shape, &scratch->reconstructed_fp16);
    CheckAclStatus(aclrtMemcpy(scratch->reconstructed_fp16_handle.AclView().data(),
                               scratch->reconstructed_fp16_handle.bytes(),
                               scratch->reconstructed_fp16.bytes.data(),
                               scratch->reconstructed_fp16.bytes.size(), ACL_MEMCPY_HOST_TO_DEVICE),
                   "aclrtMemcpy decode prior reconstructed H2D");
  }

  mlvc::DecodePriorRestoreYAcl(static_cast<const int8_t*>(scratch->y_symbols_acl.data()),
                               means_handle->AclView().data(), parts, mask_index, channels, height,
                               width, scratch->reconstructed_fp16_handle.AclView().data(),
                               g_acl_user_compute_stream, profiler);
  scratch->reconstructed_fp16_handle.MarkAclModified();
  scratch->reconstructed_fp16_handle.RecordReady(g_acl_user_compute_stream);
  RecordAclPriorValidation(profiler, parts == 4 ? "restore_y4x" : "restore_y2x", true);
}

void ApplyQuantToAcl(const mlvc::TensorHandle* decoder_q_handle, int channels, int height,
                     int width, PriorScratch* scratch, mlvc::Profiler* profiler) {
  mlvc::Check(decoder_q_handle != nullptr && decoder_q_handle->has_acl_buffer() &&
                  decoder_q_handle->acl_valid(),
              "decode prior apply_channel_quant_step requires ACL-resident quant step");
  EnsureOutputAclBuffers(scratch);
  const int quant_channels =
      static_cast<int>(decoder_q_handle->shape().NumElements()) / (height * width);
  mlvc::DecodePriorApplyChannelQuantStepAcl(
      decoder_q_handle->AclView().data(), quant_channels,
      scratch->reconstructed_fp16_handle.AclView().data(), channels, height, width,
      scratch->output_handle.AclView().data(), g_acl_user_compute_stream, profiler);
  scratch->output_handle.MarkAclModified();
  scratch->output_handle.RecordReady(g_acl_user_compute_stream);
  RecordAclPriorValidation(profiler, "apply_channel_quant_step", true);
}

void QuantizedZToAcl(const std::vector<int8_t>& symbols, PriorScratch* scratch,
                     mlvc::Profiler* profiler) {
  if (!CanUseAclDecodePrior()) {
    return;
  }
  const std::size_t symbol_bytes = symbols.size() * sizeof(int8_t);
  if (scratch->z_symbols_acl.bytes() < symbol_bytes) {
    scratch->z_symbols_acl.Allocate(symbol_bytes);
  }
  CheckAclStatus(aclrtMemcpy(scratch->z_symbols_acl.data(), symbol_bytes, symbols.data(),
                             symbol_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
                 "aclrtMemcpy decode prior z symbols H2D");

  const std::size_t fp16_bytes =
      scratch->z_shape.NumElements() * mlvc::ElementSize(mlvc::DataType::kFloat16);
  if (scratch->quantized_z_handle.bytes() != fp16_bytes) {
    scratch->quantized_z_handle.Reset(scratch->z_shape, mlvc::DataType::kFloat16);
    scratch->quantized_z_handle.UseOwnedCpuBuffer(false);
    scratch->quantized_z_handle.AllocateAclBuffer(false);
  }
  mlvc::DecodePriorInt8ToFp16Acl(static_cast<const int8_t*>(scratch->z_symbols_acl.data()),
                                 scratch->z_shape.dims().data(), scratch->z_shape.dims().size(),
                                 symbols.size(), scratch->quantized_z_handle.AclView().data(),
                                 g_acl_user_compute_stream, profiler);
  scratch->quantized_z_handle.MarkAclModified();
  scratch->quantized_z_handle.RecordReady(g_acl_user_compute_stream);
  RecordAclPriorValidation(profiler, "int8_to_fp16", true);
}

}  // namespace

RunOutput::Entry DecompressImagePrior(mlvc::StageModelSet* models, mlvc::EntropyDecoder* entropy,
                                      const TensorData& fused_prior,
                                      const mlvc::TensorHandle* fused_prior_handle,
                                      float force_zero_thres, PriorScratch* scratch,
                                      mlvc::Profiler* profiler = nullptr) {
  MLVC_PROFILE_RANGE_FUNCTION();
  mlvc::Check(scratch != nullptr, "image prior scratch is required");
  const bool use_acl_prior = CanUseAclDecodePrior();
  RunOutput init =
      fused_prior_handle != nullptr
          ? RunStage(models, "i_spatial_prior_decode_init",
                     {TensorOrHandleInput("fused_prior_params", fused_prior, fused_prior_handle)},
                     profiler)
          : RunStage(models, "i_spatial_prior_decode_init", {{"fused_prior_params", &fused_prior}},
                     profiler);
  TensorData& decoder_q = init.At("decoder_y_quant_step");
  TensorData& scale_params = init.At("scale_params");
  TensorData& mean_params = init.At("mean_params");
  TensorData& reduced_prior = init.At("reduced_prior_params");
  mlvc::TensorHandle* decoder_q_handle = init.Handle("decoder_y_quant_step");
  mlvc::TensorHandle* scale_params_handle = init.Handle("scale_params");
  mlvc::TensorHandle* mean_params_handle = init.Handle("mean_params");
  mlvc::TensorHandle* reduced_prior_handle = init.Handle("reduced_prior_params");

  const int channels = static_cast<int>(mean_params.shape.dim(1));
  const int height = static_cast<int>(mean_params.shape.dim(2));
  const int width = static_cast<int>(mean_params.shape.dim(3));
  Fp16ToFloat(mean_params, &scratch->means);
  scratch->reconstructed.assign(static_cast<std::size_t>(channels * height * width), 0.0f);

  if (use_acl_prior) {
    SinglePartToCpuFromAcl(scale_params_handle, 4, 0, channels, height, width, scratch, profiler);
  } else {
    Fp16ToFloat(scale_params, &scratch->scales);
    SinglePart4x(scratch->scales, 0, channels, height, width, &scratch->part_scales);
    RecordAclPriorValidation(profiler, "single_part4x", false);
  }
  {
    ScopedRepositoryAllocationTrackingPause allocation_pause;
    entropy->DecodeYInto(scratch->part_scales, force_zero_thres, &scratch->y_symbols);
  }
  {
    RestoreY4x(scratch->y_symbols, scratch->means, 0, channels, height, width,
               &scratch->reconstructed);
    if (use_acl_prior) {
      RestoreYToAcl(scratch->y_symbols, mean_params_handle, 4, 0, channels, height, width, scratch,
                    profiler);
    } else {
      RecordAclPriorValidation(profiler, "restore_y4x", false);
    }
  }

  for (int step = 1; step <= 3; ++step) {
    RunOutput prior_step =
        use_acl_prior
            ? RunStage(models, kISpatialPriorDecodeStepNames[step - 1],
                       {TensorOrHandleInput("reconstructed_y_so_far", scratch->reconstructed_fp16,
                                            &scratch->reconstructed_fp16_handle),
                        TensorOrHandleInput("reduced_prior_params", reduced_prior,
                                            reduced_prior_handle)},
                       profiler)
            : [&] {
                FloatToFp16Tensor(scratch->reconstructed, scratch->latent_shape,
                                  &scratch->reconstructed_fp16);
                return RunStage(models, kISpatialPriorDecodeStepNames[step - 1],
                                {TensorInput("reconstructed_y_so_far", scratch->reconstructed_fp16),
                                 TensorOrHandleInput("reduced_prior_params", reduced_prior,
                                                     reduced_prior_handle)},
                                profiler);
              }();
    Fp16ToFloat(prior_step.At("mean_params"), &scratch->means);
    mlvc::TensorHandle* step_scale_params_handle = prior_step.Handle("scale_params");
    mlvc::TensorHandle* step_mean_params_handle = prior_step.Handle("mean_params");
    if (use_acl_prior) {
      SinglePartToCpuFromAcl(step_scale_params_handle, 4, step, channels, height, width, scratch,
                             profiler);
    } else {
      Fp16ToFloat(prior_step.At("scale_params"), &scratch->scales);
      SinglePart4x(scratch->scales, step, channels, height, width, &scratch->part_scales);
      RecordAclPriorValidation(profiler, "single_part4x", false);
    }
    {
      ScopedRepositoryAllocationTrackingPause allocation_pause;
      entropy->DecodeYInto(scratch->part_scales, force_zero_thres, &scratch->y_symbols);
    }
    {
      RestoreY4x(scratch->y_symbols, scratch->means, step, channels, height, width,
                 &scratch->reconstructed);
      if (use_acl_prior) {
        RestoreYToAcl(scratch->y_symbols, step_mean_params_handle, 4, step, channels, height, width,
                      scratch, profiler);
      } else {
        RecordAclPriorValidation(profiler, "restore_y4x", false);
      }
    }
  }

  Fp16ToFloat(decoder_q, &scratch->decoder_q_values);
  ApplyChannelQuantStep(scratch->decoder_q_values, channels, height, width,
                        &scratch->reconstructed);
  if (use_acl_prior) {
    ApplyQuantToAcl(decoder_q_handle, channels, height, width, scratch, profiler);
  } else {
    RecordAclPriorValidation(profiler, "apply_channel_quant_step", false);
  }
  FloatToFp16Tensor(scratch->reconstructed, scratch->latent_shape, &scratch->output);
  return RunOutput::Entry{nullptr, &scratch->output,
                          use_acl_prior ? &scratch->output_handle : nullptr};
}

RunOutput::Entry DecompressVideoPrior(mlvc::StageModelSet* models, mlvc::EntropyDecoder* entropy,
                                      const TensorData& fused_prior,
                                      const mlvc::TensorHandle* fused_prior_handle,
                                      float force_zero_thres, PriorScratch* scratch,
                                      mlvc::Profiler* profiler = nullptr) {
  MLVC_PROFILE_RANGE_FUNCTION();
  mlvc::Check(scratch != nullptr, "video prior scratch is required");
  const bool use_acl_prior = CanUseAclDecodePrior();
  RunOutput init =
      fused_prior_handle != nullptr
          ? RunStage(models, "p_spatial_prior_decode_init",
                     {TensorOrHandleInput("fused_prior_params", fused_prior, fused_prior_handle)},
                     profiler)
          : RunStage(models, "p_spatial_prior_decode_init", {{"fused_prior_params", &fused_prior}},
                     profiler);
  TensorData& decoder_q = init.At("decoder_y_quant_step");
  TensorData& scale_params = init.At("scale_params");
  TensorData& mean_params = init.At("mean_params");
  mlvc::TensorHandle* decoder_q_handle = init.Handle("decoder_y_quant_step");
  mlvc::TensorHandle* scale_params_handle = init.Handle("scale_params");
  mlvc::TensorHandle* mean_params_handle = init.Handle("mean_params");

  const int channels = static_cast<int>(mean_params.shape.dim(1));
  const int height = static_cast<int>(mean_params.shape.dim(2));
  const int width = static_cast<int>(mean_params.shape.dim(3));
  Fp16ToFloat(mean_params, &scratch->means);
  scratch->reconstructed.assign(static_cast<std::size_t>(channels * height * width), 0.0f);

  if (use_acl_prior) {
    SinglePartToCpuFromAcl(scale_params_handle, 2, 0, channels, height, width, scratch, profiler);
  } else {
    Fp16ToFloat(scale_params, &scratch->scales);
    SinglePart2x(scratch->scales, 0, channels, height, width, &scratch->part_scales);
    RecordAclPriorValidation(profiler, "single_part2x", false);
  }
  {
    ScopedRepositoryAllocationTrackingPause allocation_pause;
    entropy->DecodeYInto(scratch->part_scales, force_zero_thres, &scratch->y_symbols);
  }
  {
    RestoreY2x(scratch->y_symbols, scratch->means, 0, channels, height, width,
               &scratch->reconstructed);
    if (use_acl_prior) {
      RestoreYToAcl(scratch->y_symbols, mean_params_handle, 2, 0, channels, height, width, scratch,
                    profiler);
    } else {
      RecordAclPriorValidation(profiler, "restore_y2x", false);
    }
  }

  RunOutput prior_step =
      use_acl_prior
          ? RunStage(models, "p_spatial_prior_decode_step",
                     {TensorOrHandleInput("reconstructed_y_part_0", scratch->reconstructed_fp16,
                                          &scratch->reconstructed_fp16_handle),
                      TensorOrHandleInput("fused_prior_params", fused_prior, fused_prior_handle)},
                     profiler)
          : [&] {
              FloatToFp16Tensor(scratch->reconstructed, scratch->latent_shape,
                                &scratch->reconstructed_fp16);
              return RunStage(
                  models, "p_spatial_prior_decode_step",
                  {TensorInput("reconstructed_y_part_0", scratch->reconstructed_fp16),
                   TensorOrHandleInput("fused_prior_params", fused_prior, fused_prior_handle)},
                  profiler);
            }();
  Fp16ToFloat(prior_step.At("mean_params"), &scratch->means);
  mlvc::TensorHandle* step_scale_params_handle = prior_step.Handle("scale_params");
  mlvc::TensorHandle* step_mean_params_handle = prior_step.Handle("mean_params");
  if (use_acl_prior) {
    SinglePartToCpuFromAcl(step_scale_params_handle, 2, 1, channels, height, width, scratch,
                           profiler);
  } else {
    Fp16ToFloat(prior_step.At("scale_params"), &scratch->scales);
    SinglePart2x(scratch->scales, 1, channels, height, width, &scratch->part_scales);
    RecordAclPriorValidation(profiler, "single_part2x", false);
  }
  {
    ScopedRepositoryAllocationTrackingPause allocation_pause;
    entropy->DecodeYInto(scratch->part_scales, force_zero_thres, &scratch->y_symbols);
  }
  {
    RestoreY2x(scratch->y_symbols, scratch->means, 1, channels, height, width,
               &scratch->reconstructed);
    if (use_acl_prior) {
      RestoreYToAcl(scratch->y_symbols, step_mean_params_handle, 2, 1, channels, height, width,
                    scratch, profiler);
    } else {
      RecordAclPriorValidation(profiler, "restore_y2x", false);
    }
  }

  Fp16ToFloat(decoder_q, &scratch->decoder_q_values);
  ApplyChannelQuantStep(scratch->decoder_q_values, channels, height, width,
                        &scratch->reconstructed);
  if (use_acl_prior) {
    ApplyQuantToAcl(decoder_q_handle, channels, height, width, scratch, profiler);
  } else {
    RecordAclPriorValidation(profiler, "apply_channel_quant_step", false);
  }
  FloatToFp16Tensor(scratch->reconstructed, scratch->latent_shape, &scratch->output);
  return RunOutput::Entry{nullptr, &scratch->output,
                          use_acl_prior ? &scratch->output_handle : nullptr};
}

void DecodeI(mlvc::StageModelSet* models, const mlvc::RuntimeSidecar& sidecar,
             mlvc::EntropyDecoder* entropy, QScaleCache* q_scales,
             const std::vector<uint8_t>& bitstream, int qp, const SourceFrameGeometry& geometry,
             CodecScratch* scratch, TensorData* x_hat, mlvc::Profiler* profiler) {
  MLVC_PROFILE_RANGE_FUNCTION();
  mlvc::Check(scratch != nullptr, "codec scratch is required");
  mlvc::Check(x_hat != nullptr, "I decode x_hat output is required");
  ScopedProfileTimer total_timer(profiler, "decode.i.total");
  entropy->SetStream(bitstream);
  const int z_height = (geometry.padded_height + 63) / 64;
  const int z_width = (geometry.padded_width + 63) / 64;
  {
    ScopedProfileTimer entropy_timer(profiler, "entropy.i.decode_z");
    ScopedRepositoryAllocationTrackingPause allocation_pause;
    entropy->DecodeZInto(qp, z_height, z_width, &scratch->image_prior.z_symbols);
  }
  FloatToFp16TensorFromInt8(scratch->image_prior.z_symbols, scratch->image_prior.z_shape,
                            &scratch->image_prior.quantized_z);
  QuantizedZToAcl(scratch->image_prior.z_symbols, &scratch->image_prior, profiler);
  RunOutput prior =
      RunStage(models, "i_hyper_decoder_prior",
               {TensorOrHandleInput("quantized_z", scratch->image_prior.quantized_z,
                                    scratch->image_prior.quantized_z_handle.acl_valid()
                                        ? &scratch->image_prior.quantized_z_handle
                                        : nullptr)},
               profiler);
  RunOutput::Entry reconstructed_y = DecompressImagePrior(
      models, entropy, prior.At("fused_prior_params"), prior.Handle("fused_prior_params"),
      sidecar.python_fast_force_zero_thres(), &scratch->image_prior, profiler);

  const TensorData& decoder_q_step = q_scales->IDecoder(qp);
  RunOutput decoder = RunStage(
      models, "i_decoder",
      {TensorOrHandleInput("reconstructed_y_latent", *reconstructed_y.tensor,
                           reconstructed_y.handle),
       TensorOrHandleInput("decoder_q_step", decoder_q_step, q_scales->IDecoderHandle(qp))},
      profiler);
  CloneTensorInto(decoder.At("x_hat"), x_hat);
}

void DecodeP(mlvc::StageModelSet* models, const mlvc::RuntimeSidecar& sidecar,
             mlvc::EntropyDecoder* entropy, QScaleCache* q_scales,
             const std::vector<uint8_t>& bitstream, const ReferenceState& state, int qp,
             const SourceFrameGeometry& geometry, CodecScratch* scratch, TensorData* x_hat,
             TensorData* decoded_feature, mlvc::TensorHandle* decoded_feature_handle,
             mlvc::Profiler* profiler) {
  MLVC_PROFILE_RANGE_FUNCTION();
  mlvc::Check(scratch != nullptr, "codec scratch is required");
  mlvc::Check(x_hat != nullptr, "P decode x_hat output is required");
  mlvc::Check(decoded_feature != nullptr, "P decode decoded feature output is required");
  ScopedProfileTimer total_timer(profiler, "decode.p.total");
  entropy->SetStream(bitstream);
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
  const int z_height = (geometry.padded_height + 63) / 64;
  const int z_width = (geometry.padded_width + 63) / 64;
  {
    ScopedProfileTimer entropy_timer(profiler, "entropy.p.decode_z");
    ScopedRepositoryAllocationTrackingPause allocation_pause;
    entropy->DecodeZInto(qp, z_height, z_width, &scratch->video_prior.z_symbols);
  }
  FloatToFp16TensorFromInt8(scratch->video_prior.z_symbols, scratch->video_prior.z_shape,
                            &scratch->video_prior.quantized_z);
  QuantizedZToAcl(scratch->video_prior.z_symbols, &scratch->video_prior, profiler);
  RunOutput prior =
      RunStage(models, "p_hyper_temporal_prior",
               {TensorOrHandleInput("quantized_z", scratch->video_prior.quantized_z,
                                    scratch->video_prior.quantized_z_handle.acl_valid()
                                        ? &scratch->video_prior.quantized_z_handle
                                        : nullptr),
                {"temporal_prior_context", &contexts.At("temporal_prior_context")}},
               profiler);
  RunOutput::Entry reconstructed_y = DecompressVideoPrior(
      models, entropy, prior.At("fused_prior_params"), prior.Handle("fused_prior_params"),
      sidecar.python_fast_force_zero_thres(), &scratch->video_prior, profiler);

  const TensorData& decoder_q_step = q_scales->PDecoder(qp);
  RunOutput synthesis = RunStage(
      models, "p_synthesis_decoder",
      {TensorOrHandleInput("reconstructed_y_latent", *reconstructed_y.tensor,
                           reconstructed_y.handle),
       TensorInput("reference_context", contexts.At("reference_context")),
       TensorOrHandleInput("decoder_q_step", decoder_q_step, q_scales->PDecoderHandle(qp))},
      profiler);
  (void)CloneRunOutputHandle(synthesis, "decoded_feature", decoded_feature_handle,
                             "decoded_feature", profiler);
  CloneTensorInto(synthesis.At("decoded_feature"), decoded_feature);

  const TensorData& reconstruction_q_step = q_scales->PReconstruction(qp);
  RunOutput reconstruction =
      RunStage(models, "p_reconstruction",
               {TensorOrHandleInput("decoded_feature", *decoded_feature, decoded_feature_handle),
                TensorOrHandleInput("reconstruction_q_step", reconstruction_q_step,
                                    q_scales->PReconstructionHandle(qp))},
               profiler);
  CloneTensorInto(reconstruction.At("x_hat"), x_hat);
}

}  // namespace mlvc::codec
