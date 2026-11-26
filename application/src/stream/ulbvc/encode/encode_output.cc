#include <ulbvc/application/stream/encode/encode_output.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>

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
  if (options.rate_control_mode == "adaptive_rate")
    rate_mode_ = ulbvc::transport::RateControlMode::kAdaptiveRate;
  else if (options.rate_control_mode == "target_rate")
    rate_mode_ = ulbvc::transport::RateControlMode::kTargetRate;
  manual_target_bps_ = options.target_bitrate_bps > 0 ? options.target_bitrate_bps
                                                      : options.adaptive_max_bitrate_bps * 0.8;
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
      rtp_sender_->EnableRateFeedback(options.rtcp_feedback_interval_ms);
      if (rate_mode_ == ulbvc::transport::RateControlMode::kAdaptiveRate) PrepareRateControl();
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
      const auto before = rtp_sender_->Stats().serialized_wire_bytes;
      rtp_sender_->SendFrame(pending.frame_index, pending.frame_type, pending.q_index,
                             pending.metadata, payload);
      const auto after = rtp_sender_->Stats().serialized_wire_bytes;
      if (after >= before && after - before >= payload.size())
        frame_wire_overhead_ =
            frame_wire_overhead_ * 0.9 + double(after - before - payload.size()) * 0.1;
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

bool EncodeOutput::CanApplyRateCommand(const ulbvc::transport::RateCommand& command) const {
  if (command.fixed_q && (*command.fixed_q < options_.min_qp || *command.fixed_q > options_.max_qp))
    return false;
  const auto mode = command.mode.value_or(
      command.fixed_q ? ulbvc::transport::RateControlMode::kFixedQ
                      : (command.target_bitrate_bps ? ulbvc::transport::RateControlMode::kTargetRate
                                                    : rate_mode_));
  if (mode == ulbvc::transport::RateControlMode::kAdaptiveRate &&
      (!rtp_sender_ || options_.output_transport_max_queue_delay_ms == 0 ||
       options_.output_transport_max_queue_delay_ms > 60000))
    return false;
  if (mode == ulbvc::transport::RateControlMode::kAdaptiveRate &&
      options_.adaptive_max_bitrate_bps <
          ulbvc::transport::MinimumAdaptiveWireRate(
              static_cast<uint32_t>(options_.output_transport_max_queue_delay_ms)))
    return false;
  if (command.target_bitrate_bps && *command.target_bitrate_bps > options_.adaptive_max_bitrate_bps)
    return false;
  return true;
}
void EncodeOutput::ApplyRateCommand(const ulbvc::transport::RateCommand& command) {
  if (command.target_bitrate_bps) manual_target_bps_ = *command.target_bitrate_bps;
  const auto next = command.mode.value_or(
      command.fixed_q ? ulbvc::transport::RateControlMode::kFixedQ
                      : (command.target_bitrate_bps ? ulbvc::transport::RateControlMode::kTargetRate
                                                    : rate_mode_));
  if (next != rate_mode_) network_rate_.reset();
  rate_mode_ = next;
  if (rate_mode_ == ulbvc::transport::RateControlMode::kFixedQ)
    rate_controller_.SetTargetBitrate(0);
  else if (rate_mode_ == ulbvc::transport::RateControlMode::kTargetRate)
    rate_controller_.SetTargetBitrate(manual_target_bps_);
  if (rtp_sender_ && rate_mode_ != ulbvc::transport::RateControlMode::kAdaptiveRate)
    rtp_sender_->SetPacingRate(
        std::max<uint64_t>(1, options_.adaptive_max_bitrate_bps * 975 / 1000));
  PrepareRateControl();
}
void EncodeOutput::PrepareRateControl() {
  using namespace ulbvc::transport;
  if (rate_mode_ != RateControlMode::kAdaptiveRate || !rtp_sender_) return;
  if (!network_rate_)
    network_rate_.emplace(
        NetworkRateOptions{options_.adaptive_max_bitrate_bps, 0,
                           static_cast<uint32_t>(options_.output_transport_max_queue_delay_ms),
                           options_.adaptive_feedback_timeout_ms});
  const auto now = std::chrono::steady_clock::now();
  const uint64_t now_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  const auto sent = rtp_sender_->Stats();
  const auto feedback = rtp_sender_->Feedback();
  const auto wire_budget = network_rate_->Update(
      now_ms,
      {sent.wire_bytes + sent.rtcp_wire_bytes, sent.queued_wire_bytes, sent.oldest_queue_delay_us},
      feedback);
  // Reserve room for forward RTCP and bounded repair. Exact EFU/detection/RTP
  // overhead is learned from each fully admitted frame, including SCU refreshes.
  rtp_sender_->SetPacingRate(std::max<uint64_t>(1, wire_budget * 975 / 1000));
  const bool encoder_pressure = network_rate_->state() == NetworkRateState::kEncoderLimited;
  const double codec_budget = std::max(
      1.0, wire_budget * (encoder_pressure ? 0.65 : 0.85) - frame_wire_overhead_ * 8 * fps_);
  if (std::abs(rate_controller_.target_bitrate_bps() - codec_budget) >= 1)
    rate_controller_.SetTargetBitrate(codec_budget);
  if (now_ms - last_rate_log_ms_ >= 1000) {
    std::cerr << "rate_control mode=adaptive_rate network_state="
              << NetworkRateStateName(network_rate_->state()) << " wire_budget_bps=" << wire_budget
              << " codec_target_bps=" << codec_budget
              << " received_bps=" << network_rate_->received_bps()
              << " loss_fraction=" << network_rate_->loss_fraction()
              << " queue_ms=" << sent.oldest_queue_delay_us / 1000.0
              << " feedback_valid=" << feedback.valid
              << " qp_saturated=" << rate_controller_.saturated() << '\n';
    last_rate_log_ms_ = now_ms;
  }
}

}  // namespace ulbvc::codec
