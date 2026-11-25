#include <ulbvc/codec/detail/profile/allocation_tracking.h>
#include <ulbvc/codec/detail/profile/codec_profile.h>
#include <ulbvc/codec/detail/stage/constants.h>
#include <ulbvc/codec/detail/stage/stage_runner.h>
#include <ulbvc/codec/detail/stage/stage_runtime_state.h>
#include <ulbvc/codec/detail/stage/stage_workspace.h>

#include <array>
#include <chrono>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ulbvc/core/status.h"
#include "ulbvc/framework/profile_range.h"

namespace ulbvc::codec {

namespace {

struct StageProfileName {
  const char* stage = nullptr;
  const char* event = nullptr;
};

constexpr std::array<StageProfileName, 22> kAclProfileNames = {{
    {"i_decoder", "acl.i_decoder"},
    {"i_encoder", "acl.i_encoder"},
    {"i_hyper_fused", "acl.i_hyper_fused"},
    {"i_hyper_decoder_prior", "acl.i_hyper_decoder_prior"},
    {"i_hyper_encoder", "acl.i_hyper_encoder"},
    {"i_spatial_prior", "acl.i_spatial_prior"},
    {"i_spatial_prior_decode_init", "acl.i_spatial_prior_decode_init"},
    {"i_spatial_prior_decode_step_1", "acl.i_spatial_prior_decode_step_1"},
    {"i_spatial_prior_decode_step_2", "acl.i_spatial_prior_decode_step_2"},
    {"i_spatial_prior_decode_step_3", "acl.i_spatial_prior_decode_step_3"},
    {"p_analysis_encoder", "acl.p_analysis_encoder"},
    {"p_hyper_fused", "acl.p_hyper_fused"},
    {"p_hyper_encoder", "acl.p_hyper_encoder"},
    {"p_hyper_temporal_prior", "acl.p_hyper_temporal_prior"},
    {"p_reconstruction", "acl.p_reconstruction"},
    {"p_reference_context", "acl.p_reference_context"},
    {"p_reference_feature_adaptor", "acl.p_reference_feature_adaptor"},
    {"p_reference_frame_adaptor", "acl.p_reference_frame_adaptor"},
    {"p_spatial_prior", "acl.p_spatial_prior"},
    {"p_spatial_prior_decode_init", "acl.p_spatial_prior_decode_init"},
    {"p_spatial_prior_decode_step", "acl.p_spatial_prior_decode_step"},
    {"p_synthesis_decoder", "acl.p_synthesis_decoder"},
}};

const char* AclProfileName(std::string_view stage_name) {
  for (const StageProfileName& entry : kAclProfileNames) {
    if (stage_name == entry.stage) {
      return entry.event;
    }
  }
  return "acl.unknown";
}

}  // namespace

RunOutput RunStage(ulbvc::StageModelSet* models, std::string_view name,
                   std::initializer_list<StageInput> inputs, ulbvc::Profiler* profiler) {
  ULBVC_PROFILE_RANGE(std::string("RunStage.") + std::string(name));
  const auto stage_begin = std::chrono::steady_clock::now();
  ulbvc::StageModel& stage = models->GetStage(name);
  ulbvc::Check(inputs.size() <= kMaxStageInputs, "stage input count exceeds fixed storage");
  ulbvc::Check(stage.record().outputs.size() <= kMaxStageOutputs,
              "stage output count exceeds fixed storage");
  RunOutput output;
  std::array<ulbvc::NamedTensorView, kMaxStageInputs> input_views;
  std::array<ulbvc::NamedTensorView, kMaxStageOutputs> output_views;
  ulbvc::ScopedCodecGraphNode graph_node(g_codec_graph_executor, profiler,
                                        ulbvc::CodecGraphNodeType::kAclStage, std::string(name),
                                        "RunStage");

  std::size_t input_count = 0;
  {
    ULBVC_PROFILE_RANGE(std::string("RunStage.prepare_inputs.") + std::string(name));
    for (const StageInput& input : inputs) {
      std::optional<ulbvc::ScopedProfileRange> profile_input;
      if (input.name != nullptr) {
        profile_input.emplace(std::string("RunStage.input.") + std::string(name) + "." +
                              input.name);
      }
      ulbvc::Check(input.name != nullptr && (input.tensor != nullptr || input.handle != nullptr),
                  "stage input name and tensor or handle must be set");
      ulbvc::Check(!(input.tensor != nullptr && input.handle != nullptr),
                  "stage input cannot provide both tensor and handle");
      std::optional<InputView> input_view;
      if (input.handle != nullptr) {
        input.handle->WaitReady(g_acl_user_compute_stream);
        const bool use_acl_view = input.handle->acl_valid() && input.handle->has_acl_buffer();
        input_views[input_count++] = ulbvc::NamedTensorView{
            input.name, use_acl_view ? input.handle->AclView() : input.handle->CpuView()};
        if (profiler != nullptr) {
          ScopedRepositoryAllocationTrackingPause allocation_pause;
          profiler->AddCounter(
              "io.acl_input_handle." + std::string(name) + "." + input.name + ".bytes", "copy",
              "copy", 0.0,
              {ulbvc::Profiler::Arg("stage", std::string(name)),
               ulbvc::Profiler::Arg("tensor", input.name),
               ulbvc::Profiler::Arg(
                   "reason", use_acl_view ? "acl_tensor_handle" : "host_staged_tensor_handle"),
               ulbvc::Profiler::Arg("residency", input.handle->ResidencyString()),
               ulbvc::Profiler::Arg("location", use_acl_view ? "acl" : "cpu"),
               ulbvc::Profiler::Arg("bytes", static_cast<uint64_t>(input.handle->bytes()))});
        }
        continue;
      }
      if (input.tensor->has_external_buffer()) {
        const ulbvc::TensorView view = input.tensor->View();
        input_views[input_count++] = ulbvc::NamedTensorView{input.name, view};
        if (profiler != nullptr) {
          ScopedRepositoryAllocationTrackingPause allocation_pause;
          profiler->AddCounter(
              "io.acl_input_external." + std::string(name) + "." + input.name + ".bytes", "copy",
              "copy", 0.0,
              {ulbvc::Profiler::Arg("stage", std::string(name)),
               ulbvc::Profiler::Arg("tensor", input.name),
               ulbvc::Profiler::Arg("reason", "external_device_buffer"),
               ulbvc::Profiler::Arg("location", view.location() == ulbvc::MemoryLocation::kAcl
                                                       ? "acl"
                                                       : "pinned_cpu"),
               ulbvc::Profiler::Arg("bytes", static_cast<uint64_t>(input.tensor->ByteSize()))});
        }
        continue;
      }
      if (g_stage_output_workspace != nullptr) {
        input_view = g_stage_output_workspace->InputViewForCpuMirror(input.tensor);
      }
      if (input_view.has_value()) {
        input_views[input_count++] = ulbvc::NamedTensorView{input.name, input_view->view};
        if (profiler != nullptr && input_view->device_reuse) {
          ScopedRepositoryAllocationTrackingPause allocation_pause;
          profiler->AddCounter(
              "io.acl_input_reuse." + std::string(name) + "." + input.name + ".bytes", "copy",
              "copy", 0.0,
              {ulbvc::Profiler::Arg("stage", std::string(name)),
               ulbvc::Profiler::Arg("tensor", input.name),
                ulbvc::Profiler::Arg("bytes", static_cast<uint64_t>(input.tensor->ByteSize()))});
        }
      } else {
        if (g_stage_output_workspace != nullptr) {
          input_view = g_stage_output_workspace->UploadInputToDevice(stage.record(), input.name,
                                                                     *input.tensor, profiler);
        }
        if (input_view.has_value()) {
          input_views[input_count++] = ulbvc::NamedTensorView{input.name, input_view->view};
        } else {
          input_views[input_count++] = ulbvc::NamedTensorView{input.name, input.tensor->View()};
        }
      }
    }
  }
  std::size_t output_count = 0;
  {
    ULBVC_PROFILE_RANGE(std::string("RunStage.bind_outputs.") + std::string(name));
    if (g_stage_output_workspace != nullptr) {
      output = g_stage_output_workspace->Bind(stage.record());
    } else {
      static thread_local std::array<TensorData, kMaxStageOutputs> fallback_tensors;
      output.size = stage.record().outputs.size();
      for (std::size_t i = 0; i < output.size; ++i) {
        fallback_tensors[i] = MakeTensorLike(stage.record().outputs[i]);
        output.tensors[i] =
            RunOutput::Entry{&stage.record().outputs[i].name, &fallback_tensors[i], nullptr};
      }
    }
    if (g_stage_output_workspace != nullptr) {
      g_stage_output_workspace->BindOutputViews(stage.record(), output, &output_views,
                                                &output_count);
    } else {
      for (const ulbvc::TensorSpec& spec : stage.record().outputs) {
        output_views[output_count] =
            ulbvc::NamedTensorView{spec.name.c_str(), output.tensors[output_count].tensor->View()};
        ++output_count;
      }
    }
  }

  {
    ULBVC_PROFILE_RANGE(std::string("RunStage.acl_run.") + std::string(name));
    ScopedRepositoryAllocationTrackingPause allocation_pause;
    stage.RunNamed(input_views.data(), input_count, output_views.data(), output_count);
  }
  const auto stage_end = std::chrono::steady_clock::now();
  if (profiler != nullptr) {
    std::vector<ulbvc::ProfileArgument> args;
    args.reserve(8 + input_count + output_count);
    args.push_back(ulbvc::Profiler::Arg("stage", std::string(name)));
    args.push_back(ulbvc::Profiler::Arg("input_count", static_cast<int>(input_count)));
    args.push_back(ulbvc::Profiler::Arg("output_count", static_cast<int>(output_count)));
    args.push_back(ulbvc::Profiler::Arg(
        "binding_mode",
        g_stage_output_workspace != nullptr &&
                g_stage_output_workspace->binding_mode() == StageOutputBindingMode::kAclDeviceWithCpuMirror
            ? "acl-device-with-cpu-mirror"
            : "cpu"));
    args.push_back(ulbvc::Profiler::Arg(
        "residency_model",
        g_stage_output_workspace != nullptr &&
                g_stage_output_workspace->binding_mode() == StageOutputBindingMode::kAclDeviceWithCpuMirror
            ? "tensor_handle"
            : "tensor_data"));
    for (std::size_t i = 0; i < input_count; ++i) {
      args.push_back(ulbvc::Profiler::Arg("input_" + std::to_string(i), input_views[i].name));
      args.push_back(ulbvc::Profiler::Arg("input_" + std::to_string(i) + "_bytes",
                                         static_cast<uint64_t>(input_views[i].view.bytes())));
      args.push_back(ulbvc::Profiler::Arg(
          "input_" + std::to_string(i) + "_location",
          input_views[i].view.location() == ulbvc::MemoryLocation::kAcl
              ? "acl"
              : (input_views[i].view.location() == ulbvc::MemoryLocation::kPinnedCpu ? "pinned_cpu"
                                                                                    : "cpu")));
    }
    for (std::size_t i = 0; i < output_count; ++i) {
      args.push_back(ulbvc::Profiler::Arg("output_" + std::to_string(i), output_views[i].name));
      args.push_back(ulbvc::Profiler::Arg("output_" + std::to_string(i) + "_bytes",
                                         static_cast<uint64_t>(output_views[i].view.bytes())));
      args.push_back(ulbvc::Profiler::Arg(
          "output_" + std::to_string(i) + "_location",
          output_views[i].view.location() == ulbvc::MemoryLocation::kAcl
              ? "acl"
              : (output_views[i].view.location() == ulbvc::MemoryLocation::kPinnedCpu ? "pinned_cpu"
                                                                                     : "cpu")));
      if (i < output.size && output.tensors[i].handle != nullptr) {
        const ulbvc::TensorHandleState state = output.tensors[i].handle->State();
        args.push_back(ulbvc::Profiler::Arg("output_" + std::to_string(i) + "_residency",
                                           ulbvc::TensorResidencyName(state.residency)));
        args.push_back(
            ulbvc::Profiler::BoolArg("output_" + std::to_string(i) + "_cpu_valid", state.cpu_valid));
        args.push_back(
            ulbvc::Profiler::BoolArg("output_" + std::to_string(i) + "_acl_valid", state.acl_valid));
      }
    }
    AddProfileEventWithArgs(profiler, AclProfileName(name), "acl", "acl",
                            profiler->StartMs(stage_begin),
                            profiler->DurationMs(stage_begin, stage_end), std::move(args));
  }
  if (g_stage_output_workspace != nullptr) {
    ULBVC_PROFILE_RANGE(std::string("RunStage.mirror_outputs.") + std::string(name));
    g_stage_output_workspace->MirrorOutputsToCpu(stage.record(), output, profiler);
  }
  return output;
}

RunOutput RunIHyperPrior(ulbvc::StageModelSet* models, const TensorData& y_latent,
                         RunOutput* hyper_output, ulbvc::Profiler* profiler) {
  ULBVC_PROFILE_RANGE_FUNCTION();
  ulbvc::Check(models != nullptr, "model set is required");
  ulbvc::Check(hyper_output != nullptr, "I hyper output storage is required");
  const ulbvc::ModelRecord* fused =
      models->manifest().FindFusionCandidate({"i_hyper_encoder", "i_hyper_decoder_prior"});
  if (g_enable_stage_fusion && fused != nullptr && models->HasStage(fused->name)) {
    RunOutput fused_output = RunStage(models, fused->name, {{"y_latent", &y_latent}}, profiler);
    *hyper_output = fused_output;
    return fused_output;
  }
  *hyper_output = RunStage(models, "i_hyper_encoder", {{"y_latent", &y_latent}}, profiler);
  return RunStage(models, "i_hyper_decoder_prior",
                  {{"quantized_z", &hyper_output->At("quantized_z")}}, profiler);
}

RunOutput RunPHyperPrior(ulbvc::StageModelSet* models, const TensorData& y_latent,
                         const TensorData& temporal_prior_context, RunOutput* hyper_output,
                         ulbvc::Profiler* profiler) {
  ULBVC_PROFILE_RANGE_FUNCTION();
  ulbvc::Check(models != nullptr, "model set is required");
  ulbvc::Check(hyper_output != nullptr, "P hyper output storage is required");
  const ulbvc::ModelRecord* fused =
      models->manifest().FindFusionCandidate({"p_hyper_encoder", "p_hyper_temporal_prior"});
  if (g_enable_stage_fusion && fused != nullptr && models->HasStage(fused->name)) {
    RunOutput fused_output = RunStage(
        models, fused->name,
        {{"y_latent", &y_latent}, {"temporal_prior_context", &temporal_prior_context}}, profiler);
    *hyper_output = fused_output;
    return fused_output;
  }
  *hyper_output = RunStage(models, "p_hyper_encoder", {{"y_latent", &y_latent}}, profiler);
  return RunStage(models, "p_hyper_temporal_prior",
                  {{"quantized_z", &hyper_output->At("quantized_z")},
                   {"temporal_prior_context", &temporal_prior_context}},
                  profiler);
}

}  // namespace ulbvc::codec
