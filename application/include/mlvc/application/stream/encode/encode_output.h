#ifndef MLVC_APPLICATION_STREAM_ENCODE_ENCODE_OUTPUT_H_
#define MLVC_APPLICATION_STREAM_ENCODE_ENCODE_OUTPUT_H_

#include <mlvc/application/stream/mlvc_stream.h>
#include <mlvc/application/stream/mlvc_internal.h>
#include <mlvc/codec/mlvc_rate_control.h>
#include <mlvc/framework/profiler.h>
#include <mlvc/io/mlvc_bitstream.h>
#include <mlvc/io/udp_frame_transport.h>

#include <cstdint>
#include <filesystem>
#include <optional>

namespace mlvc::codec {

class EncodeOutput {
 public:
  EncodeOutput(const EncodeStreamOptions& options, const mlvc::io::MlvcBitstreamHeader& header,
               const MlvcRateControlOptions& rate_options, mlvc::Profiler* profiler);
  ~EncodeOutput();

  EncodeOutput(const EncodeOutput&) = delete;
  EncodeOutput& operator=(const EncodeOutput&) = delete;

  void Flush(PendingEncodedFrame pending);
  void Close();
  void SendEnd();

  MlvcRateController& rate_controller() { return rate_controller_; }

  uint64_t payload_bytes() const { return payload_bytes_; }
  uint64_t file_bytes() const;

 private:
  const EncodeStreamOptions& options_;
  mlvc::Profiler* profiler_ = nullptr;
  double fps_ = 30.0;
  std::optional<mlvc::io::MlvcBitstreamWriter> writer_;
  std::optional<mlvc::io::UdpMlvcSender> udp_sender_;
  MlvcRateController rate_controller_;
  uint64_t payload_bytes_ = 0;
};

}  // namespace mlvc::codec

#endif  // MLVC_APPLICATION_STREAM_ENCODE_ENCODE_OUTPUT_H_
