#ifndef ULBVC_APPLICATION_STREAM_ULBVC_STREAM_H_
#define ULBVC_APPLICATION_STREAM_ULBVC_STREAM_H_

#include <ulbvc/application/runtime/ulbvc_codec_runtime.h>
#include <ulbvc/application/stream/stream_encoder.h>
#include <ulbvc/framework/codec_graph_executor.h>
#include <ulbvc/framework/entropy_worker.h>
#include <ulbvc/framework/profiler.h>
#include <ulbvc/runtime/model_manifest.h>

#include <filesystem>
#include <optional>
#include <string>

namespace ulbvc::codec {

bool IsUlbvcManifest(const ulbvc::ModelManifest& manifest);

struct EncodePipelineServices {
  ulbvc::Profiler* profiler = nullptr;
  ulbvc::CodecGraphExecutor* graph_executor = nullptr;
  ulbvc::EntropyWorker* entropy_worker = nullptr;
  ulbvc::app::UlbvcCodecRuntime* codec_runtime = nullptr;
};

int RunEncodeStream(const EncodeStreamOptions& options, EncodePipelineServices* services = nullptr);

struct DecodePipelineServices {
  ulbvc::Profiler* profiler = nullptr;
  ulbvc::CodecGraphExecutor* graph_executor = nullptr;
  ulbvc::EntropyWorker* entropy_worker = nullptr;
  ulbvc::app::UlbvcCodecRuntime* codec_runtime = nullptr;
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
  std::string output_transport_mode = "none";
  std::string output_transport_rtsp_url;
  std::string output_transport_rtsp_encoder = "dvpp";
  std::string output_transport_rtsp_preset = "ultrafast";
  std::string output_transport_rtsp_transport = "udp";
  int output_transport_rtsp_crf = 0;
  double fps = 30.0;
  int crf = 23;
  int device = 0;
  int frame_num = -1;
  int drop_frame_index = -1;
  int forced_ltr_reference_frame = -1;
  int forced_ltr_recovery_frame = -1;
  int input_transport_port = 0;
  std::string input_transport_mode = "udp";
  uint8_t input_transport_payload_type = 96;
  int output_transport_queue_capacity = 3;
  PipelineOptions pipeline;
};

int RunDecodeStream(const DecodeStreamOptions& options, DecodePipelineServices* services = nullptr);

}  // namespace ulbvc::codec

#endif  // ULBVC_APPLICATION_STREAM_ULBVC_STREAM_H
