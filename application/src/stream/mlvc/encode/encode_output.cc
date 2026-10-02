#include <mlvc/application/stream/encode/encode_output.h>

#include <filesystem>

#include "mlvc/core/status.h"
#include "mlvc/framework/profile_range.h"

namespace mlvc::codec {

EncodeOutput::EncodeOutput(const EncodeStreamOptions& options,
                           const mlvc::io::MlvcBitstreamHeader& header,
                           const MlvcRateControlOptions& rate_options, mlvc::Profiler* profiler)
    : options_(options),
      profiler_(profiler),
      fps_(rate_options.fps),
      rate_controller_(rate_options) {
  const bool udp_output = options_.output_transport_port > 0;
  Check(udp_output || !options_.output_bitstream_path.empty(),
        "encode requires output_bitstream_path or udp_port");
  if (udp_output) {
    Check(!options_.output_transport_host.empty(), "udp_host is required when udp_port is set");
    if (options_.output_transport_mode == "rtp") {
      rtp_sender_.emplace(
          options_.output_transport_host, static_cast<uint16_t>(options_.output_transport_port),
          options_.output_transport_pacing_rate_bps, options_.output_transport_max_burst_bytes,
          options_.output_transport_max_queue_bytes,
          options_.output_transport_max_queue_delay_ms,
          options_.output_transport_payload_type);
      rtp_sender_->SendHeader(header);
    } else {
      mlvc::transport::UdpSendOptions send_options;
      send_options.pacing_rate_bps = options_.output_transport_pacing_rate_bps;
      send_options.max_burst_bytes = options_.output_transport_max_burst_bytes;
      send_options.max_queue_bytes = options_.output_transport_max_queue_bytes;
      send_options.max_queue_delay_ms = options_.output_transport_max_queue_delay_ms;
      udp_sender_.emplace(options_.output_transport_host,
                          static_cast<uint16_t>(options_.output_transport_port), send_options);
      udp_sender_->SendHeader(header);
    }
  } else {
    writer_.emplace(options_.output_bitstream_path, header);
  }
}

EncodeOutput::~EncodeOutput() {
  try {
    Close();
  } catch (...) {
  }
}

void EncodeOutput::Flush(PendingEncodedFrame pending) {
  std::vector<uint8_t> payload = pending.payload.get();
  {
    mlvc::ScopedCpuTimer timer(profiler_, "bitstream.frame_write");
    if (writer_.has_value()) {
      writer_->WriteFrame(pending.frame_index, pending.frame_type, pending.q_index,
                          pending.metadata, payload);
    }
    if (udp_sender_.has_value()) {
      mlvc::ScopedCpuTimer udp_timer(profiler_, "udp.send_enqueue");
      udp_sender_->SendFrame(pending.frame_index, pending.frame_type, pending.q_index, payload);
    }
    if (rtp_sender_.has_value()) {
      mlvc::ScopedCpuTimer rtp_timer(profiler_, "rtp.send");
      rtp_sender_->SendFrame(pending.frame_index, pending.frame_type, pending.q_index,
                             pending.metadata, payload);
    }
  }
  payload_bytes_ += payload.size();
  rate_controller_.Update(static_cast<double>(pending.frame_index) / fps_, pending.frame_type,
                          pending.q_index, payload.size(),
                          mlvc::io::kMlvcBitstreamFrameOverheadBytes +
                              (pending.metadata.translation_warp &&
                               pending.frame_type != MlvcFrameType::kIFrame ? 2 : 0));
}

void EncodeOutput::Close() {
  if (closed_) return;
  std::exception_ptr first_error;
  const auto close_one = [&first_error](auto& resource) {
    if (!resource.has_value()) return;
    try {
      resource->Close();
    } catch (...) {
      if (first_error == nullptr) first_error = std::current_exception();
    }
  };
  close_one(writer_);
  close_one(udp_sender_);
  close_one(rtp_sender_);
  closed_ = true;
  if (first_error != nullptr) std::rethrow_exception(first_error);
}

void EncodeOutput::SendEnd() {
  Check(!closed_, "encode output is closed");
  if (udp_sender_.has_value()) udp_sender_->SendEnd();
  if (rtp_sender_.has_value()) rtp_sender_->SendEnd();
}

bool EncodeOutput::ConsumeRandomAccessRequest() {
  return rtp_sender_.has_value() && rtp_sender_->ConsumeRandomAccessRequest();
}
bool EncodeOutput::PopMlvcControl(mlvc::transport::MlvcControlMessage* message) {
  return rtp_sender_.has_value() && rtp_sender_->PopMlvcControl(message);
}
void EncodeOutput::SendMlvcControlResponse(
    const mlvc::transport::MlvcControlMessage& request, bool accepted,
    const std::string& reason) {
  if (rtp_sender_.has_value()) rtp_sender_->SendMlvcControlResponse(request, accepted, reason);
}

uint64_t EncodeOutput::file_bytes() const {
  if (!options_.output_bitstream_path.empty() &&
      std::filesystem::exists(options_.output_bitstream_path)) {
    return std::filesystem::file_size(options_.output_bitstream_path);
  }
  return payload_bytes_;
}

}  // namespace mlvc::codec
