#include <mlvc/application/stream/encode/encode_output.h>

#include <filesystem>

#include "mlvc/core/status.h"
#include "mlvc/framework/profile_range.h"

namespace mlvc::codec {

EncodeOutput::EncodeOutput(const EncodeStreamOptions& options,
                           const mlvc::io::MlvcBitstreamHeader& header,
                           const MlvcRateControlOptions& rate_options,
                           mlvc::Profiler* profiler)
    : options_(options), profiler_(profiler), fps_(rate_options.fps), rate_controller_(rate_options) {
  const bool udp_output = options_.udp_port > 0;
  Check(udp_output || !options_.output_bitstream_path.empty(),
        "encode requires output_bitstream_path or udp_port");
  if (udp_output) {
    Check(!options_.udp_host.empty(), "udp_host is required when udp_port is set");
    udp_sender_.emplace(options_.udp_host, static_cast<uint16_t>(options_.udp_port));
    udp_sender_->SendHeader(header);
  } else {
    writer_.emplace(options_.output_bitstream_path, header);
  }
}

EncodeOutput::~EncodeOutput() { Close(); }

void EncodeOutput::Flush(PendingEncodedFrame pending) {
  std::vector<uint8_t> payload = pending.payload.get();
  {
    mlvc::ScopedCpuTimer timer(profiler_, "bitstream.frame_write");
    if (writer_.has_value()) {
      writer_->WriteFrame(pending.frame_index, pending.frame_type, pending.q_index, payload);
    }
    if (udp_sender_.has_value()) {
      mlvc::ScopedCpuTimer udp_timer(profiler_, "udp.send_enqueue");
      udp_sender_->SendFrame(pending.frame_index, pending.frame_type, pending.q_index, payload);
    }
  }
  payload_bytes_ += payload.size();
  rate_controller_.Update(static_cast<double>(pending.frame_index) / fps_, pending.frame_type,
                          pending.q_index, payload.size(),
                          mlvc::io::kMlvcBitstreamFrameOverheadBytes);
}

void EncodeOutput::Close() {
  if (writer_.has_value()) writer_->Close();
}

void EncodeOutput::SendEnd() {
  if (udp_sender_.has_value()) udp_sender_->SendEnd();
}

uint64_t EncodeOutput::file_bytes() const {
  if (!options_.output_bitstream_path.empty() &&
      std::filesystem::exists(options_.output_bitstream_path)) {
    return std::filesystem::file_size(options_.output_bitstream_path);
  }
  return payload_bytes_;
}

}  // namespace mlvc::codec
