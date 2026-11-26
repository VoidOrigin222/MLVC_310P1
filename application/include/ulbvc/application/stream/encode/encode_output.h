#ifndef ULBVC_APPLICATION_STREAM_ENCODE_ENCODE_OUTPUT_H_
#define ULBVC_APPLICATION_STREAM_ENCODE_ENCODE_OUTPUT_H_

#include <ulbvc/application/stream/ulbvc_internal.h>
#include <ulbvc/application/stream/ulbvc_stream.h>
#include <ulbvc/codec/ulbvc_rate_control.h>
#include <ulbvc/framework/profiler.h>
#include <ulbvc/io/udp_frame_transport.h>
#include <ulbvc/io/ulbvc_bitstream.h>
#include <ulbvc/transport/network_rate_controller.h>

#include <cstdint>
#include <exception>
#include <filesystem>
#include <optional>

namespace ulbvc::codec {

class EncodeOutput {
 public:
  EncodeOutput(const EncodeStreamOptions& options, const ulbvc::io::UlbvcBitstreamHeader& header,
               const UlbvcRateControlOptions& rate_options, ulbvc::Profiler* profiler);
  ~EncodeOutput();

  EncodeOutput(const EncodeOutput&) = delete;
  EncodeOutput& operator=(const EncodeOutput&) = delete;

  void Flush(PendingEncodedFrame pending);
  void Close();
  void SendEnd();
  bool ConsumeRandomAccessRequest();
  bool PopUlbvcControl(ulbvc::transport::UlbvcControlMessage* message);
  void SendUlbvcControlResponse(const ulbvc::transport::UlbvcControlMessage& request, bool accepted,
                               const std::string& reason);

  UlbvcRateController& rate_controller() { return rate_controller_; }

  const LatencyStats& ready_to_output() const { return ready_to_output_; }
  uint64_t payload_bytes() const { return payload_bytes_; }
  ulbvc::transport::RtpTransportStats transport_stats() const;
  uint64_t file_bytes() const;
  void PrepareRateControl();
  bool CanApplyRateCommand(const ulbvc::transport::RateCommand& command) const;
  void ApplyRateCommand(const ulbvc::transport::RateCommand& command);

 private:
  const EncodeStreamOptions& options_;
  ulbvc::Profiler* profiler_ = nullptr;
  double fps_ = 30.0;
  std::optional<ulbvc::io::UlbvcBitstreamWriter> writer_;
  std::optional<ulbvc::io::UdpUlbvcSender> udp_sender_;
  std::optional<ulbvc::io::RtpUlbvcSender> rtp_sender_;
  UlbvcRateController rate_controller_;
  uint64_t payload_bytes_ = 0;
  LatencyStats ready_to_output_;
  bool closed_ = false;
  ulbvc::transport::RateControlMode rate_mode_ = ulbvc::transport::RateControlMode::kFixedQ;
  std::optional<ulbvc::transport::NetworkRateController> network_rate_;
  double manual_target_bps_ = 0;
  double frame_wire_overhead_ = 256;
  uint64_t last_rate_log_ms_ = 0;
};

}  // namespace ulbvc::codec

#endif  // ULBVC_APPLICATION_STREAM_ENCODE_ENCODE_OUTPUT_H_
