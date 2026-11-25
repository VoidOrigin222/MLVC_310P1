#include <ulbvc/application/stream/encode/encode_output.h>

#include <filesystem>

#include "ulbvc/core/status.h"
#include "ulbvc/framework/profile_range.h"

namespace ulbvc::codec {

EncodeOutput::EncodeOutput(const EncodeStreamOptions& options,
                           const ulbvc::io::UlbvcBitstreamHeader& header,
                           const UlbvcRateControlOptions& rate_options, ulbvc::Profiler* profiler)
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
      ulbvc::transport::UdpSendOptions send_options;
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
  const auto entropy_wait_begin = std::chrono::steady_clock::now();
  std::vector<uint8_t> payload = pending.payload.get();
  if (profiler_) profiler_->AddDurationNow("sync.entropy_output_wait", entropy_wait_begin,
                                          std::chrono::steady_clock::now());
  if (pending.detection.valid()) {
    const auto detection_wait_begin = std::chrono::steady_clock::now();
    const auto& detection = pending.detection.get();
    if (profiler_) profiler_->AddDurationNow("sync.detection_output_wait", detection_wait_begin,
                                            std::chrono::steady_clock::now());
    Check(detection.frame_id == static_cast<uint32_t>(pending.frame_index),
          "detection result is associated with a different codec frame");
    pending.metadata.tlvs.push_back(ulbvc::transport::SerializeDetections(detection));
  }
  {
    ulbvc::ScopedCpuTimer timer(profiler_, "bitstream.frame_write");
    if (writer_.has_value()) {
      writer_->WriteFrame(pending.frame_index, pending.frame_type, pending.q_index,
                          pending.metadata, payload);
    }
    if (udp_sender_.has_value()) {
      ulbvc::ScopedCpuTimer udp_timer(profiler_, "udp.send_enqueue");
      udp_sender_->SendFrame(pending.frame_index, pending.frame_type, pending.q_index, payload);
    }
    if (rtp_sender_.has_value()) {
      ulbvc::ScopedCpuTimer rtp_timer(profiler_, "rtp.send");
      rtp_sender_->SendFrame(pending.frame_index, pending.frame_type, pending.q_index,
                             pending.metadata, payload);
    }
  }
  if (pending.ready_at != std::chrono::steady_clock::time_point{})
    ready_to_output_.Add(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - pending.ready_at).count());
  payload_bytes_ += payload.size();
  rate_controller_.Update(static_cast<double>(pending.frame_index) / fps_, pending.frame_type,
                          pending.q_index, payload.size(),
                          ulbvc::io::kUlbvcBitstreamFrameOverheadBytes +
                              (pending.metadata.translation_warp &&
                               pending.frame_type != UlbvcFrameType::kIFrame ? 2 : 0));
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
bool EncodeOutput::PopUlbvcControl(ulbvc::transport::UlbvcControlMessage* message) {
  return rtp_sender_.has_value() && rtp_sender_->PopUlbvcControl(message);
}
void EncodeOutput::SendUlbvcControlResponse(
    const ulbvc::transport::UlbvcControlMessage& request, bool accepted,
    const std::string& reason) {
  if (rtp_sender_.has_value()) rtp_sender_->SendUlbvcControlResponse(request, accepted, reason);
}

uint64_t EncodeOutput::file_bytes() const {
  if (!options_.output_bitstream_path.empty() &&
      std::filesystem::exists(options_.output_bitstream_path)) {
    return std::filesystem::file_size(options_.output_bitstream_path);
  }
  return payload_bytes_;
}

ulbvc::transport::RtpTransportStats EncodeOutput::transport_stats() const {
  if (rtp_sender_.has_value()) return rtp_sender_->Stats();
  return {};
}

}  // namespace ulbvc::codec
