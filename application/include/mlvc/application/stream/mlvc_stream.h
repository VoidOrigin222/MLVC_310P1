#ifndef MLVC_APPLICATION_STREAM_MLVC_STREAM_H_
#define MLVC_APPLICATION_STREAM_MLVC_STREAM_H_

#include <mlvc/application/stream/stream_encoder.h>
#include <mlvc/application/runtime/mlvc_codec_runtime.h>
#include <mlvc/framework/codec_graph_executor.h>
#include <mlvc/framework/entropy_worker.h>
#include <mlvc/framework/profiler.h>
#include <mlvc/runtime/model_manifest.h>

#include <filesystem>
#include <optional>
#include <string>

namespace mlvc::codec {

bool IsMlvcManifest(const mlvc::ModelManifest& manifest);

struct EncodePipelineServices {
  mlvc::Profiler* profiler = nullptr;
  mlvc::CodecGraphExecutor* graph_executor = nullptr;
  mlvc::EntropyWorker* entropy_worker = nullptr;
  mlvc::app::MlvcCodecRuntime* codec_runtime = nullptr;
};

int RunEncodeStream(const EncodeStreamOptions& options,
                    EncodePipelineServices* services = nullptr);

struct DecodePipelineServices {
  mlvc::Profiler* profiler = nullptr;
  mlvc::CodecGraphExecutor* graph_executor = nullptr;
  mlvc::EntropyWorker* entropy_worker = nullptr;
  mlvc::app::MlvcCodecRuntime* codec_runtime = nullptr;
};

struct DecodeStreamOptions {
  std::filesystem::path manifest_path;
  std::filesystem::path input_bitstream_path;
  std::filesystem::path output_video_path;
  std::filesystem::path profile_output_path;
  std::string output_format;
  std::string bitrate = "6500k";
  std::string preset = "medium";
  std::string execution_profile;
  std::string forward_host;
  std::string forward_mode = "jpeg";
  double fps = 30.0;
  int crf = 23;
  int device = 0;
  int frame_num = -1;
  int drop_frame_index = -1;
  int forced_ltr_reference_frame = -1;
  int forced_ltr_recovery_frame = -1;
  int udp_port = 0;
  int forward_port = 0;
  int forward_queue_capacity = 3;
  int forward_jpeg_quality = 75;
  PipelineOptions pipeline;
};

int RunDecodeStream(const DecodeStreamOptions& options,
                    DecodePipelineServices* services = nullptr);

}  // namespace mlvc::codec

#endif  // MLVC_APPLICATION_STREAM_MLVC_STREAM_H
