#include "mlvc/io/udp_frame_transport.h"

#include <chrono>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <utility>

#include "mlvc/core/status.h"
#include "mlvc/transport/mlvc_media_unit.h"

namespace mlvc::io {
namespace {
constexpr uint8_t kHeaderMessage = 1;
constexpr uint8_t kFrameMessage = 2;
constexpr uint8_t kEndMessage = 3;
constexpr std::size_t kMaxPendingRtpFrames = 64;
constexpr std::size_t kMaxPendingRtpFrameBytes = 128u * 1024u * 1024u;
constexpr auto kRtpFrameReorderWait = std::chrono::milliseconds(100);

bool IsRandomAccessEfu(const mlvc::transport::MlvcEfu& efu) {
  return efu.frame_type == static_cast<uint8_t>(mlvc::codec::MlvcFrameType::kIFrame) &&
         (efu.unit_flags & (mlvc::transport::kEfuRandomAccess |
                            mlvc::transport::kEfuResetReference)) ==
             (mlvc::transport::kEfuRandomAccess | mlvc::transport::kEfuResetReference);
}

std::string FormatBundleHash(const std::array<uint8_t, 32>& hash) {
  std::ostringstream output;
  output << std::hex;
  for (uint8_t byte : hash) output << static_cast<unsigned>(byte >> 4)
                                    << static_cast<unsigned>(byte & 0x0f);
  return output.str();
}

void PutU32(std::vector<uint8_t>* out, uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    out->push_back(static_cast<uint8_t>(value >> shift));
  }
}

void PutI32(std::vector<uint8_t>* out, int32_t value) { PutU32(out, static_cast<uint32_t>(value)); }

void PutU64(std::vector<uint8_t>* out, uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8) {
    out->push_back(static_cast<uint8_t>(value >> shift));
  }
}

void PutDouble(std::vector<uint8_t>* out, double value) {
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  PutU64(out, bits);
}

uint32_t ReadU32(const uint8_t* data) {
  return static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8) |
         (static_cast<uint32_t>(data[2]) << 16) | (static_cast<uint32_t>(data[3]) << 24);
}

int32_t ReadI32(const uint8_t* data) { return static_cast<int32_t>(ReadU32(data)); }

uint64_t ReadU64(const uint8_t* data) {
  uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 8) {
    value |= static_cast<uint64_t>(data[shift / 8]) << shift;
  }
  return value;
}

double ReadDouble(const uint8_t* data) {
  const uint64_t bits = ReadU64(data);
  double value = 0.0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

void PutBe32(std::vector<uint8_t>* out, uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8) out->push_back(static_cast<uint8_t>(value >> shift));
}

mlvc::transport::MlvcScu MakeScu(const MlvcBitstreamHeader& header, uint32_t config_id) {
  mlvc::transport::MlvcScu scu;
  scu.config_id = config_id;
  scu.codec_bundle_sha256 = header.codec_bundle_sha256;
  scu.coded_width = static_cast<uint32_t>(header.width);
  scu.coded_height = static_cast<uint32_t>(header.height);
  if (header.coded_width > 0) scu.coded_width = static_cast<uint32_t>(header.coded_width);
  if (header.coded_height > 0) scu.coded_height = static_cast<uint32_t>(header.coded_height);
  scu.visible_width = static_cast<uint32_t>(header.width);
  scu.visible_height = static_cast<uint32_t>(header.height);
  scu.max_frame_bytes = static_cast<uint32_t>(std::min<uint64_t>(
      mlvc::io::MaxMlvcFramePayloadBytes(header.width, header.height),
      std::numeric_limits<uint32_t>::max()));
  const uint64_t fps_scaled = static_cast<uint64_t>(std::llround(header.fps * 1000.0));
  const uint32_t fps_num = static_cast<uint32_t>(std::max<uint64_t>(1, fps_scaled));
  const uint32_t fps_den = 1000;
  const uint32_t divisor = std::gcd(fps_num, fps_den);
  scu.nominal_fps_num = fps_num / divisor;
  scu.nominal_fps_den = fps_den / divisor;
  // A verified model bundle hash is carried in the mandatory critical SCU TLV;
  // SerializeScu rejects an unset hash so a receiver never guesses a model.
  std::vector<uint8_t> policy;
  PutBe32(&policy, static_cast<uint32_t>(header.gop));
  PutBe32(&policy, static_cast<uint32_t>(header.reset_interval));
  PutBe32(&policy, static_cast<uint32_t>(header.ltr_start_idx));
  PutBe32(&policy, static_cast<uint32_t>(header.ltr_period));
  scu.tlvs.push_back({mlvc::transport::kMlvcTlvNominalPolicy, std::move(policy)});
  std::vector<uint8_t> forced;
  PutBe32(&forced, static_cast<uint32_t>(header.forced_ltr_recovery_frame));
  PutBe32(&forced, static_cast<uint32_t>(header.forced_ltr_reference_frame));
  scu.tlvs.push_back({mlvc::transport::kMlvcTlvVendorData, std::move(forced)});
  return scu;
}

uint32_t RtpTimestamp(uint32_t offset, int frame_index, double fps) {
  const long double ticks = std::round(static_cast<long double>(frame_index) * 90000.0L /
                                       static_cast<long double>(fps));
  if (!std::isfinite(ticks) || ticks < 0.0L) throw mlvc::Error("invalid RTP timestamp");
  const long double modulo = std::fmod(ticks, 4294967296.0L);
  return offset + static_cast<uint32_t>(modulo);
}

uint32_t RtpTimestampForPts(uint32_t offset, int64_t pts) {
  Check(pts >= 0, "RTP presentation timestamp must be non-negative");
  return offset + static_cast<uint32_t>(static_cast<uint64_t>(pts));
}

void CheckCompatibleRtpConfiguration(const mlvc::transport::MlvcScu& current,
                                     const mlvc::transport::MlvcScu& next) {
  Check(mlvc::transport::MlvcScuDecoderCompatible(current, next),
        "RTP configuration switch changes static decoder compatibility fields");
}

}  // namespace

UdpMlvcSender::UdpMlvcSender(const std::string& host, uint16_t port,
                             mlvc::transport::UdpSendOptions options)
    : sender_(host, port, "MLVC", options) {}

UdpMlvcSender::~UdpMlvcSender() = default;

void UdpMlvcSender::SendHeader(const MlvcBitstreamHeader& header) {
  Check(!closed_, "MLVC UDP sender is closed");
  Check(!header_sent_, "MLVC UDP header has already been sent");
  MlvcBitstreamHeader normalized = header;
  if (normalized.version == 0) normalized.version = 3;
  ValidateMlvcBitstreamHeader(normalized);
  max_payload_size_ = MaxMlvcFramePayloadBytes(normalized.width, normalized.height);
  std::vector<uint8_t> message;
  message.reserve(65);
  message.push_back(kHeaderMessage);
  PutU32(&message, normalized.version);
  PutI32(&message, normalized.width);
  PutI32(&message, normalized.height);
  PutDouble(&message, normalized.fps);
  PutI32(&message, normalized.q_index);
  PutI32(&message, normalized.gop);
  PutI32(&message, normalized.reset_interval);
  PutI32(&message, normalized.ltr_start_idx);
  PutI32(&message, normalized.ltr_period);
  PutI32(&message, normalized.ltr_qp_shift);
  PutDouble(&message, normalized.target_bitrate_bps);
  PutU32(&message, normalized.flags);
  if (normalized.version >= 4) {
    PutI32(&message, normalized.forced_ltr_recovery_frame);
    PutI32(&message, normalized.forced_ltr_reference_frame);
  }
  sender_.Send(message);
  header_sent_ = true;
  expected_frame_index_ = 0;
}

void UdpMlvcSender::SendFrame(int frame_index, mlvc::codec::MlvcFrameType frame_type, int q_index,
                              const std::vector<uint8_t>& payload) {
  Check(!closed_, "MLVC UDP sender is closed");
  Check(header_sent_, "MLVC UDP header must be sent before frames");
  ValidateMlvcFrameMetadata(frame_index, frame_type, q_index, expected_frame_index_,
                            expected_frame_index_ == 0);
  Check(payload.size() <= max_payload_size_,
        "MLVC frame payload exceeds the configured resolution limit");
  Check(payload.size() <= std::numeric_limits<uint32_t>::max(), "MLVC frame payload is too large");
  std::vector<uint8_t> message;
  message.reserve(14 + payload.size());
  message.push_back(kFrameMessage);
  PutI32(&message, frame_index);
  message.push_back(static_cast<uint8_t>(frame_type));
  PutI32(&message, q_index);
  PutU32(&message, static_cast<uint32_t>(payload.size()));
  message.insert(message.end(), payload.begin(), payload.end());
  sender_.Send(message);
  ++expected_frame_index_;
}

void UdpMlvcSender::SendEnd() {
  Check(!closed_, "MLVC UDP sender is closed");
  Check(header_sent_, "MLVC UDP header must be sent before end of stream");
  sender_.Send(std::vector<uint8_t>{kEndMessage});
  sender_.Flush();
}

void UdpMlvcSender::Close() {
  if (closed_) return;
  sender_.Flush();
  closed_ = true;
}

UdpMlvcReceiver::UdpMlvcReceiver(uint16_t port) : receiver_(port, "MLVC") {}

UdpMlvcReceiver::~UdpMlvcReceiver() = default;

MlvcBitstreamHeader UdpMlvcReceiver::ReceiveHeader() {
  const std::vector<uint8_t> message = receiver_.Receive();
  Check((message.size() == 57 || message.size() == 65) && message[0] == kHeaderMessage,
        "invalid MLVC UDP header message");
  const uint8_t* cursor = message.data() + 1;
  MlvcBitstreamHeader header;
  header.version = ReadU32(cursor);
  cursor += 4;
  header.width = ReadI32(cursor);
  cursor += 4;
  header.height = ReadI32(cursor);
  cursor += 4;
  header.fps = ReadDouble(cursor);
  cursor += 8;
  header.q_index = ReadI32(cursor);
  cursor += 4;
  header.gop = ReadI32(cursor);
  cursor += 4;
  header.reset_interval = ReadI32(cursor);
  cursor += 4;
  header.ltr_start_idx = ReadI32(cursor);
  cursor += 4;
  header.ltr_period = ReadI32(cursor);
  cursor += 4;
  header.ltr_qp_shift = ReadI32(cursor);
  cursor += 4;
  header.target_bitrate_bps = ReadDouble(cursor);
  cursor += 8;
  header.flags = ReadU32(cursor);
  cursor += 4;
  if (header.version >= 4) {
    Check(message.size() == 65, "invalid MLVC UDP v4 header size");
    header.forced_ltr_recovery_frame = ReadI32(cursor);
    cursor += 4;
    header.forced_ltr_reference_frame = ReadI32(cursor);
  } else {
    Check(message.size() == 57, "invalid MLVC UDP legacy header size");
  }
  mlvc::io::ValidateMlvcBitstreamHeader(header);
  Check(header.version >= 2 && header.version <= 4, "unsupported MLVC UDP header version");
  header_ = header;
  max_payload_size_ = mlvc::io::MaxMlvcFramePayloadBytes(header.width, header.height);
  expected_frame_index_ = 0;
  header_received_ = true;
  return header;
}

bool UdpMlvcReceiver::ReceiveFrame(int* frame_index, mlvc::codec::MlvcFrameType* frame_type,
                                   int* q_index, std::vector<uint8_t>* payload) {
  Check(frame_index && frame_type && q_index && payload, "UDP frame outputs are required");
  Check(header_received_, "MLVC UDP header must be received before frames");
  const std::vector<uint8_t> message = receiver_.Receive();
  Check(!message.empty(), "empty MLVC UDP message");
  if (message[0] == kEndMessage) {
    Check(message.size() == 1, "invalid MLVC UDP end message");
    return false;
  }
  Check(message[0] == kFrameMessage && message.size() >= 14, "invalid MLVC UDP frame message");
  const uint8_t* cursor = message.data() + 1;
  *frame_index = ReadI32(cursor);
  cursor += 4;
  const uint8_t raw_type = *cursor++;
  Check(raw_type <= static_cast<uint8_t>(mlvc::codec::MlvcFrameType::kLtrRecovery),
        "invalid MLVC UDP frame type");
  *frame_type = static_cast<mlvc::codec::MlvcFrameType>(raw_type);
  *q_index = ReadI32(cursor);
  cursor += 4;
  mlvc::io::ValidateMlvcFrameMetadata(*frame_index, *frame_type, *q_index, expected_frame_index_,
                                      expected_frame_index_ == 0);
  const uint32_t payload_size = ReadU32(cursor);
  cursor += 4;
  Check(payload_size == message.size() - 14, "invalid MLVC UDP frame payload size");
  Check(payload_size <= max_payload_size_, "MLVC UDP frame payload exceeds maximum for resolution");
  payload->assign(cursor, cursor + payload_size);
  ++expected_frame_index_;
  return true;
}

RtpMlvcSender::RtpMlvcSender(const std::string& host, uint16_t port, uint64_t pacing_rate_bps,
                             std::size_t max_burst_bytes, std::size_t max_queue_bytes,
                             uint64_t max_queue_delay_ms, uint8_t payload_type)
    : sender_(host, port, 0, pacing_rate_bps, max_burst_bytes, max_queue_bytes,
              max_queue_delay_ms, payload_type) {
  std::random_device rd;
  timestamp_offset_ = (static_cast<uint32_t>(rd()) << 16) ^ static_cast<uint32_t>(rd());
}
RtpMlvcSender::~RtpMlvcSender() = default;
void RtpMlvcSender::SendHeader(const MlvcBitstreamHeader& header) {
  Check(!closed_, "MLVC RTP sender is closed");
  Check(!header_sent_, "MLVC RTP header has already been sent");
  MlvcBitstreamHeader normalized = header;
  if (normalized.version == 0) normalized.version = 4;
  ValidateMlvcBitstreamHeader(normalized);
  const auto scu = MakeScu(normalized, config_id_);
  session_config_ = mlvc::transport::SerializeScu(scu);
  sender_.SetSessionConfig(config_id_, session_config_, 0, timestamp_offset_);
  fps_ = normalized.fps;
  max_payload_size_ = std::min<uint64_t>(scu.max_frame_bytes,
                                         MaxMlvcFramePayloadBytes(normalized.width, normalized.height));
  header_sent_ = true;
  expected_frame_index_ = 0;
  configuration_switch_pending_ = false;
}

void RtpMlvcSender::SendConfiguration(uint32_t config_id,
                                      const MlvcBitstreamHeader& header) {
  Check(!closed_, "MLVC RTP sender is closed");
  Check(header_sent_, "MLVC RTP initial header must be sent before a configuration switch");
  Check(!configuration_switch_pending_, "MLVC RTP configuration switch is already pending");
  const auto config_order = mlvc::transport::CompareMlvcSerial32(config_id, config_id_);
  Check(config_order == mlvc::transport::MlvcSerial32Order::kNewer && config_id != 0,
        "MLVC RTP configuration IDs must increase monotonically");
  MlvcBitstreamHeader normalized = header;
  if (normalized.version == 0) normalized.version = 4;
  ValidateMlvcBitstreamHeader(normalized);
  const auto scu = MakeScu(normalized, config_id);
  const auto session_config = mlvc::transport::SerializeScu(scu);
  CheckCompatibleRtpConfiguration(mlvc::transport::ParseScu(session_config_),
                                  mlvc::transport::ParseScu(session_config));
  session_config_ = session_config;
  config_id_ = config_id;
  fps_ = normalized.fps;
  max_payload_size_ = std::min<uint64_t>(scu.max_frame_bytes,
                                         MaxMlvcFramePayloadBytes(normalized.width, normalized.height));
  configuration_switch_pending_ = true;
}
void RtpMlvcSender::SendFrame(int frame_index, mlvc::codec::MlvcFrameType frame_type, int q_index,
                              const std::vector<uint8_t>& payload) {
  SendFrame(frame_index, frame_type, q_index, MlvcFrameMetadata{}, payload);
}

void RtpMlvcSender::SendFrame(int frame_index, mlvc::codec::MlvcFrameType frame_type, int q_index,
                              const MlvcFrameMetadata& metadata,
                              const std::vector<uint8_t>& payload) {
  Check(!closed_, "MLVC RTP sender is closed");
  Check(header_sent_, "MLVC RTP header must be sent before frames");
  if (configuration_switch_pending_) {
    Check(frame_type == mlvc::codec::MlvcFrameType::kIFrame,
          "MLVC RTP configuration switch must begin with a random-access I-frame");
    if (metadata.explicit_metadata) {
      Check((metadata.unit_flags & (mlvc::transport::kEfuRandomAccess |
                                    mlvc::transport::kEfuResetReference)) ==
                (mlvc::transport::kEfuRandomAccess | mlvc::transport::kEfuResetReference),
            "MLVC RTP configuration switch I-frame is not marked random access");
    }
  }
  ValidateMlvcFrameMetadata(frame_index, frame_type, q_index, expected_frame_index_,
                            expected_frame_index_ == 0);
  Check(metadata.explicit_metadata || frame_type == mlvc::codec::MlvcFrameType::kIFrame,
        "RTP inter frames require explicit reference metadata");
  Check(payload.size() <= max_payload_size_,
        "RTP MLVC frame payload exceeds the configured resolution limit");
  Check(payload.size() <= std::numeric_limits<uint32_t>::max(),
        "RTP MLVC frame payload is too large");
  Check(static_cast<uint64_t>(frame_index) + 1 <= std::numeric_limits<uint32_t>::max(),
        "RTP MLVC frame index exceeds the unit identifier range");
  const int64_t pts = metadata.explicit_metadata
                          ? metadata.pts
                          : static_cast<int64_t>(std::llround(
                                static_cast<long double>(frame_index) * 90000.0L /
                                static_cast<long double>(fps_)));
  const uint32_t timestamp = RtpTimestampForPts(timestamp_offset_, pts);
  if (configuration_switch_pending_) {
    sender_.SetSessionConfig(config_id_, session_config_, 0, timestamp);
  } else if (frame_type == mlvc::codec::MlvcFrameType::kIFrame && frame_index != 0) {
    sender_.ResendSessionConfig(timestamp);
  }
  mlvc::transport::MlvcEfu efu;
  efu.config_id = config_id_;
  efu.frame_id = static_cast<uint32_t>(frame_index);
  efu.frame_type = static_cast<uint8_t>(frame_type);
  efu.entropy_q_index = static_cast<uint8_t>(q_index);
  efu.model_q_index = static_cast<uint8_t>(q_index);
  efu.pts = pts;
  if (metadata.explicit_metadata) {
    efu.unit_flags = metadata.unit_flags | mlvc::transport::kEfuCrcPresent;
    efu.short_ref_frame_id = metadata.short_ref_frame_id;
    efu.long_ref_frame_id = metadata.long_ref_frame_id;
    const int model_q_index = metadata.model_q_index < 0 ? q_index : metadata.model_q_index;
    Check(model_q_index >= 0 && model_q_index < 64,
          "RTP explicit model Q index must be in [0, 63]");
    efu.model_q_index = static_cast<uint8_t>(model_q_index);
  } else if (frame_type == mlvc::codec::MlvcFrameType::kIFrame) {
    efu.unit_flags |= mlvc::transport::kEfuRandomAccess | mlvc::transport::kEfuResetReference;
    efu.short_ref_frame_id = mlvc::transport::kMlvcNoReference;
    efu.long_ref_frame_id = mlvc::transport::kMlvcNoReference;
  }
  efu.entropy_payload = payload;
  const auto unit = mlvc::transport::SerializeEfu(efu);
  sender_.SendUnit(mlvc::transport::RtpUnitType::kEfu, 0, config_id_,
                   static_cast<uint32_t>(frame_index), timestamp, unit);
  configuration_switch_pending_ = false;
  ++expected_frame_index_;
}
void RtpMlvcSender::SendEnd() {
  Check(!closed_, "MLVC RTP sender is closed");
  Check(header_sent_, "MLVC RTP header must be sent before end of stream");
  Check(!configuration_switch_pending_,
        "MLVC RTP configuration switch has no random-access frame");
  const uint32_t next_frame_id = static_cast<uint32_t>(expected_frame_index_);
  const auto unit = mlvc::transport::SerializeEos(config_id_, next_frame_id);
  sender_.SendUnit(mlvc::transport::RtpUnitType::kEos, 0, config_id_,
                   next_frame_id,
                   RtpTimestamp(timestamp_offset_, expected_frame_index_, fps_), unit);
}

void RtpMlvcSender::Close() {
  if (closed_) return;
  Check(!configuration_switch_pending_,
        "MLVC RTP configuration switch has no random-access frame");
  sender_.Close();
  closed_ = true;
}

RtpMlvcReceiver::RtpMlvcReceiver(uint16_t port, std::array<uint8_t, 32> expected_bundle_hash,
                                 uint8_t payload_type)
    : receiver_(port, "MLVC RTP", payload_type), expected_bundle_hash_(expected_bundle_hash) {}
RtpMlvcReceiver::~RtpMlvcReceiver() = default;

void RtpMlvcReceiver::HandleConfigurationUnit(const std::vector<uint8_t>& unit) {
  const auto scu = mlvc::transport::ParseScu(unit);
  if (std::any_of(expected_bundle_hash_.begin(), expected_bundle_hash_.end(),
                  [](uint8_t byte) { return byte != 0; })) {
      Check(scu.codec_bundle_sha256 == expected_bundle_hash_,
            "RTP configuration uses a different model bundle (received=" +
                FormatBundleHash(scu.codec_bundle_sha256) + ", expected=" +
                FormatBundleHash(expected_bundle_hash_) + ")");
  }
  const auto config_order = mlvc::transport::CompareMlvcSerial32(scu.config_id, config_id_);
  if (config_order == mlvc::transport::MlvcSerial32Order::kAmbiguous) {
    throw Error("RTP SCU configuration ID is serial-number ambiguous");
  }
  if (config_order == mlvc::transport::MlvcSerial32Order::kSame) {
    Check(unit == active_config_unit_, "RTP configuration ID was reused with different SCU bytes");
    return;
  }
  Check(config_order == mlvc::transport::MlvcSerial32Order::kNewer,
        "RTP SCU configuration ID moved backwards");
  Check(header_received_, "RTP configuration switch arrived before the initial SCU");
  Check(pending_config_id_ == 0, "RTP configuration switch is already pending");
  CheckCompatibleRtpConfiguration(mlvc::transport::ParseScu(active_config_unit_), scu);
  config_id_ = scu.config_id;
  active_config_unit_ = unit;
  pending_config_id_ = config_id_;
  reorder_deadline_.reset();
  waiting_for_recovery_ = false;
  eos_frame_count_.reset();
  max_payload_size_ = std::min<uint64_t>(scu.max_frame_bytes,
                                         MaxMlvcFramePayloadBytes(header_.width, header_.height));
  available_short_reference_id_.reset();
  ltr_frame_ids_.clear();
}

void RtpMlvcReceiver::QueueConfigurationUnit(std::vector<uint8_t> unit, uint32_t config_id) {
  for (const auto& queued : pending_config_units_) {
    if (mlvc::transport::ParseScu(queued).config_id == config_id) {
      Check(queued == unit, "conflicting RTP SCU with the same configuration ID");
      return;
    }
  }
  Check(pending_config_units_.size() < 8, "RTP configuration reorder window is full");
  pending_config_units_.push_back(std::move(unit));
}

void RtpMlvcReceiver::HandleEndUnit(const std::vector<uint8_t>& unit) {
  const auto header = mlvc::transport::ParseMediaUnitHeader(unit);
  Check(header.unit_type == mlvc::transport::MlvcMediaUnitType::kEos,
        "RTP end unit has an invalid media-unit type");
  mlvc::transport::ValidateMediaUnit(unit);
  Check(header.unit_id != mlvc::transport::kMlvcNoReference,
        "RTP EOS is missing its exclusive frame boundary");
  Check(header.unit_id >= static_cast<uint32_t>(expected_frame_index_),
        "RTP EOS frame boundary precedes an already decoded frame");
  if (eos_frame_count_.has_value()) {
    Check(*eos_frame_count_ == header.unit_id,
          "conflicting RTP EOS frame boundaries");
    return;
  }
  for (const auto& entry : pending_frames_) {
    const auto frame_header = mlvc::transport::ParseMediaUnitHeader(entry.second);
    if (frame_header.config_id == config_id_) {
      Check(entry.first < header.unit_id,
            "RTP frame is at or beyond the EOS frame boundary");
    }
  }
  eos_frame_count_ = header.unit_id;
}

void RtpMlvcReceiver::BufferFrameUnit(const std::vector<uint8_t>& unit, uint32_t frame_id) {
  const auto common = mlvc::transport::ParseMediaUnitHeader(unit);
  Check(common.unit_type == mlvc::transport::MlvcMediaUnitType::kEfu &&
            common.unit_id == frame_id,
        "RTP pending frame metadata does not match its frame ID");
  Check(frame_id <= static_cast<uint32_t>(std::numeric_limits<int>::max()),
        "RTP frame ID exceeds the application range");
  const auto efu = mlvc::transport::ParseEfu(unit);
  if (efu.config_id == config_id_ && eos_frame_count_.has_value()) {
    Check(frame_id < *eos_frame_count_, "RTP frame is at or beyond the EOS frame boundary");
  }
  if (!header_received_) {
    // A live receiver can join well after frame zero.  Before the matching
    // SCU and random-access EFU arrive, discard dependent frames instead of
    // growing the reorder window toward an arbitrary frame number.
    const bool is_random_access = IsRandomAccessEfu(efu);
    if (!is_random_access) return;
    const auto existing = pending_frames_.find(frame_id);
    if (existing != pending_frames_.end()) {
      Check(existing->second == unit, "conflicting duplicate RTP random-access EFU");
      return;
    }
    Check(pending_frames_.size() < kMaxPendingRtpFrames &&
              unit.size() <= kMaxPendingRtpFrameBytes - pending_frame_bytes_,
          "RTP initial random-access window is full");
    pending_frame_bytes_ += unit.size();
    pending_frames_.emplace(frame_id, unit);
    return;
  }
  // A late frame from an older configuration may be discarded only after it
  // is behind the decode cursor.  It must not enter the reorder map ahead of
  // the cursor, where it could be mistaken for the next configuration.
  // A delayed frame from an older, superseded configuration is never
  // eligible for decoding, regardless of its frame number.
  const auto frame_config_order =
      mlvc::transport::CompareMlvcSerial32(common.config_id, config_id_);
  if (frame_config_order == mlvc::transport::MlvcSerial32Order::kOlder) return;
  Check(frame_config_order != mlvc::transport::MlvcSerial32Order::kAmbiguous,
        "RTP EFU configuration ID is serial-number ambiguous");
  if (frame_id < static_cast<uint32_t>(expected_frame_index_)) {
    const auto recent = recent_frames_.find(frame_id);
    if (recent != recent_frames_.end()) {
      Check(recent->second.first == unit.size() &&
                recent->second.second == mlvc::transport::MlvcCrc32c(unit),
            "conflicting duplicate RTP media unit");
    }
    return;
  }
  const uint64_t frame_distance = static_cast<uint64_t>(frame_id) -
                                  static_cast<uint64_t>(expected_frame_index_);
  const bool random_access = IsRandomAccessEfu(efu);
  const bool ltr_recovery = efu.config_id == config_id_ &&
      efu.frame_type == static_cast<uint8_t>(mlvc::codec::MlvcFrameType::kLtrRecovery) &&
      ltr_frame_ids_.find(efu.long_ref_frame_id) != ltr_frame_ids_.end();
  const bool can_recover_gap = random_access || ltr_recovery;
  if (waiting_for_recovery_ && frame_id > static_cast<uint32_t>(expected_frame_index_) &&
      !can_recover_gap) {
    return;
  }
  if (!can_recover_gap && frame_distance >= kMaxPendingRtpFrames) {
    MarkDependencyGap();
    return;
  }
  const auto existing = pending_frames_.find(frame_id);
  if (existing != pending_frames_.end()) {
    Check(existing->second == unit, "conflicting duplicate RTP media unit");
    return;
  }
  if (pending_frames_.size() >= kMaxPendingRtpFrames ||
      unit.size() > kMaxPendingRtpFrameBytes - pending_frame_bytes_) {
    MarkDependencyGap();
    if (!can_recover_gap && frame_id > static_cast<uint32_t>(expected_frame_index_)) return;
  }
  Check(pending_frames_.size() < kMaxPendingRtpFrames &&
            unit.size() <= kMaxPendingRtpFrameBytes - pending_frame_bytes_,
        "RTP pending frame resource limit exceeded");
  pending_frame_bytes_ += unit.size();
  pending_frames_.emplace(frame_id, unit);
}

bool RtpMlvcReceiver::PopReadyFrame(std::vector<uint8_t>* unit) {
  const auto it = pending_frames_.find(static_cast<uint32_t>(expected_frame_index_));
  if (it == pending_frames_.end()) return false;
  const auto header = mlvc::transport::ParseMediaUnitHeader(it->second);
  // A future-configuration EFU may complete before its SCU because RTP
  // packets can be reordered.  Keep it buffered until the SCU arrives and is
  // applied; popping it here would turn a harmless reorder into a false
  // "unknown configuration" failure.
  if (header.config_id != config_id_) return false;
  pending_frame_bytes_ -= it->second.size();
  *unit = std::move(it->second);
  pending_frames_.erase(it);
  reorder_deadline_.reset();
  waiting_for_recovery_ = false;
  return true;
}

bool RtpMlvcReceiver::HasPendingCurrentFrameAfterExpected() const {
  const auto expected = static_cast<uint32_t>(expected_frame_index_);
  for (auto it = pending_frames_.upper_bound(expected); it != pending_frames_.end(); ++it) {
    const auto header = mlvc::transport::ParseMediaUnitHeader(it->second);
    if (header.config_id == config_id_) return true;
  }
  return false;
}

bool RtpMlvcReceiver::TryStartAtRandomAccess() {
  if (config_id_ == 0 || pending_frames_.empty()) return false;
  for (auto it = pending_frames_.begin(); it != pending_frames_.end();) {
    const auto efu = mlvc::transport::ParseEfu(it->second);
    const auto config_order =
        mlvc::transport::CompareMlvcSerial32(efu.config_id, config_id_);
    if (config_order == mlvc::transport::MlvcSerial32Order::kOlder) {
      pending_frame_bytes_ -= it->second.size();
      it = pending_frames_.erase(it);
      continue;
    }
    Check(config_order != mlvc::transport::MlvcSerial32Order::kAmbiguous,
          "RTP recovery EFU configuration ID is serial-number ambiguous");
    if (config_order == mlvc::transport::MlvcSerial32Order::kNewer) {
      ++it;
      continue;
    }
    if (IsRandomAccessEfu(efu)) {
      expected_frame_index_ = static_cast<int>(efu.frame_id);
      for (auto stale = pending_frames_.begin(); stale != it;) {
        pending_frame_bytes_ -= stale->second.size();
        stale = pending_frames_.erase(stale);
      }
      header_received_ = true;
      pending_config_id_ = 0;
      reorder_deadline_.reset();
      waiting_for_recovery_ = false;
      return true;
    }
    pending_frame_bytes_ -= it->second.size();
    it = pending_frames_.erase(it);
  }
  return false;
}

std::optional<uint32_t> RtpMlvcReceiver::FindRecoveryFrame() const {
  const uint32_t expected = static_cast<uint32_t>(expected_frame_index_);
  for (auto it = pending_frames_.upper_bound(expected); it != pending_frames_.end(); ++it) {
    const auto efu = mlvc::transport::ParseEfu(it->second);
    const auto config_order =
        mlvc::transport::CompareMlvcSerial32(efu.config_id, config_id_);
    Check(config_order != mlvc::transport::MlvcSerial32Order::kAmbiguous,
          "RTP recovery EFU configuration ID is serial-number ambiguous");
    bool config_available = config_order == mlvc::transport::MlvcSerial32Order::kSame;
    if (!config_available && config_order == mlvc::transport::MlvcSerial32Order::kNewer) {
      config_available = std::any_of(
          pending_config_units_.begin(), pending_config_units_.end(),
          [&efu](const std::vector<uint8_t>& queued) {
            return mlvc::transport::ParseScu(queued).config_id == efu.config_id;
          });
    }
    if (!config_available) continue;
    const bool random_access = IsRandomAccessEfu(efu);
    const bool ltr_recovery = efu.config_id == config_id_ &&
        efu.frame_type == static_cast<uint8_t>(mlvc::codec::MlvcFrameType::kLtrRecovery) &&
        ltr_frame_ids_.find(efu.long_ref_frame_id) != ltr_frame_ids_.end();
    if (random_access || ltr_recovery) return it->first;
  }
  return std::nullopt;
}

void RtpMlvcReceiver::MarkDependencyGap() {
  waiting_for_recovery_ = true;
  reorder_deadline_.reset();
  const uint32_t expected = static_cast<uint32_t>(expected_frame_index_);
  for (auto it = pending_frames_.begin(); it != pending_frames_.end();) {
    if (it->first == expected) {
      ++it;
      continue;
    }
    const auto efu = mlvc::transport::ParseEfu(it->second);
    const bool random_access = IsRandomAccessEfu(efu);
    const bool usable_ltr_recovery =
        efu.config_id == config_id_ &&
        efu.frame_type == static_cast<uint8_t>(mlvc::codec::MlvcFrameType::kLtrRecovery) &&
        ltr_frame_ids_.find(efu.long_ref_frame_id) != ltr_frame_ids_.end();
    if (random_access || usable_ltr_recovery) {
      ++it;
    } else {
      pending_frame_bytes_ -= it->second.size();
      it = pending_frames_.erase(it);
    }
  }
}

void RtpMlvcReceiver::DiscardPendingFramesBefore(uint32_t frame_id) {
  for (auto it = pending_frames_.begin(); it != pending_frames_.end() && it->first < frame_id;) {
    pending_frame_bytes_ -= it->second.size();
    it = pending_frames_.erase(it);
  }
}

void RtpMlvcReceiver::RememberDecodedFrame(uint32_t frame_id, bool store_as_ltr) {
  constexpr std::size_t kMaxLtrFrames = 64;
  available_short_reference_id_ = frame_id;
  if (store_as_ltr) {
    ltr_frame_ids_.insert(frame_id);
    while (ltr_frame_ids_.size() > kMaxLtrFrames) {
      ltr_frame_ids_.erase(ltr_frame_ids_.begin());
    }
  }
}

MlvcBitstreamHeader RtpMlvcReceiver::ReceiveHeader() {
  std::vector<uint8_t> unit;
  for (;;) {
    Check(receiver_.Receive(&unit), "RTP stream ended before SCU");
    const auto common = mlvc::transport::ParseMediaUnitHeader(unit);
    if (common.unit_type != mlvc::transport::MlvcMediaUnitType::kScu) {
      if (common.unit_type == mlvc::transport::MlvcMediaUnitType::kEfu) {
        (void)mlvc::transport::ParseEfu(unit);
        BufferFrameUnit(unit, common.unit_id);
        if (TryStartAtRandomAccess()) return header_;
      } else if (common.unit_type == mlvc::transport::MlvcMediaUnitType::kEos) {
        mlvc::transport::ValidateMediaUnit(unit);
        throw Error("RTP stream ended before its initial SCU");
      }
      continue;
    }
    const auto scu = mlvc::transport::ParseScu(unit);
    const auto config_order =
        mlvc::transport::CompareMlvcSerial32(scu.config_id, config_id_);
    Check(config_order != mlvc::transport::MlvcSerial32Order::kAmbiguous,
          "RTP SCU configuration ID is serial-number ambiguous");
    if (config_id_ != 0 && config_order == mlvc::transport::MlvcSerial32Order::kSame) {
      Check(unit == active_config_unit_,
            "RTP configuration ID was reused with different SCU bytes");
      if (TryStartAtRandomAccess()) return header_;
      continue;
    }
    if (config_id_ != 0 && config_order == mlvc::transport::MlvcSerial32Order::kOlder) continue;
    Check(config_id_ == 0 || config_order == mlvc::transport::MlvcSerial32Order::kNewer,
          "RTP initial configuration IDs did not advance");
    if (std::any_of(expected_bundle_hash_.begin(), expected_bundle_hash_.end(),
                    [](uint8_t byte) { return byte != 0; })) {
      Check(scu.codec_bundle_sha256 == expected_bundle_hash_,
            "RTP SCU model bundle hash does not match the local manifest (received=" +
                FormatBundleHash(scu.codec_bundle_sha256) + ", expected=" +
                FormatBundleHash(expected_bundle_hash_) + ")");
    }
    config_id_ = scu.config_id;
    active_config_unit_ = unit;
    MlvcBitstreamHeader header;
    header.version = 4;
    header.width = static_cast<int>(scu.visible_width);
    header.height = static_cast<int>(scu.visible_height);
    header.coded_width = static_cast<int>(scu.coded_width);
    header.coded_height = static_cast<int>(scu.coded_height);
    header.codec_bundle_sha256 = scu.codec_bundle_sha256;
    header.fps = static_cast<double>(scu.nominal_fps_num) /
                 static_cast<double>(scu.nominal_fps_den);
    header.q_index = std::min<int>(21, scu.q_index_count - 1);
    for (const auto& tlv : scu.tlvs) {
      if (tlv.type == mlvc::transport::kMlvcTlvNominalPolicy && tlv.value.size() == 16) {
        auto read_nonnegative = [&tlv](std::size_t offset) {
          const uint32_t value = (static_cast<uint32_t>(tlv.value[offset]) << 24) |
                                 (static_cast<uint32_t>(tlv.value[offset + 1]) << 16) |
                                 (static_cast<uint32_t>(tlv.value[offset + 2]) << 8) |
                                 static_cast<uint32_t>(tlv.value[offset + 3]);
          Check(value <= static_cast<uint32_t>(std::numeric_limits<int>::max()),
                "RTP SCU policy value exceeds the application range");
          return static_cast<int>(value);
        };
        header.gop = read_nonnegative(0);
        header.reset_interval = read_nonnegative(4);
        header.ltr_start_idx = read_nonnegative(8);
        header.ltr_period = read_nonnegative(12);
      }
      if (tlv.type == mlvc::transport::kMlvcTlvVendorData && tlv.value.size() == 8) {
        auto read_frame_index = [&tlv](std::size_t offset) {
          const uint32_t value = (static_cast<uint32_t>(tlv.value[offset]) << 24) |
                                 (static_cast<uint32_t>(tlv.value[offset + 1]) << 16) |
                                 (static_cast<uint32_t>(tlv.value[offset + 2]) << 8) |
                                 static_cast<uint32_t>(tlv.value[offset + 3]);
          Check(value == 0xffffffffu ||
                    value <= static_cast<uint32_t>(std::numeric_limits<int>::max()),
                "RTP forced LTR frame exceeds the application range");
          return value == 0xffffffffu ? -1 : static_cast<int>(value);
        };
        header.forced_ltr_recovery_frame = read_frame_index(0);
        header.forced_ltr_reference_frame = read_frame_index(4);
      }
    }
    header.ltr_qp_shift = 0;
    header.target_bitrate_bps = 0.0;
    header.flags = 0;
    ValidateMlvcBitstreamHeader(header);
    header_ = header;
    max_payload_size_ = std::min<uint64_t>(scu.max_frame_bytes,
                                           MaxMlvcFramePayloadBytes(header.width, header.height));
    pending_config_id_ = 0;
    available_short_reference_id_.reset();
    ltr_frame_ids_.clear();
    eos_frame_count_.reset();
    pending_eos_unit_.reset();
    if (TryStartAtRandomAccess()) return header_;
  }
}
bool RtpMlvcReceiver::ReceiveFrame(int* frame_index, mlvc::codec::MlvcFrameType* frame_type,
                                   int* q_index, std::vector<uint8_t>* payload) {
  Check(frame_index && frame_type && q_index && payload, "RTP frame outputs are required");
  Check(header_received_, "MLVC RTP header must be received before frames");
  std::vector<uint8_t> message;
  for (;;) {
    if (pending_eos_unit_.has_value()) {
      const auto eos_header = mlvc::transport::ParseMediaUnitHeader(*pending_eos_unit_);
      if (eos_header.config_id == config_id_) {
        HandleEndUnit(*pending_eos_unit_);
        pending_eos_unit_.reset();
      } else if (mlvc::transport::CompareMlvcSerial32(eos_header.config_id, config_id_) ==
                 mlvc::transport::MlvcSerial32Order::kOlder) {
        pending_eos_unit_.reset();
      }
    }
    if (pending_config_id_ == 0 && !pending_config_units_.empty()) {
      uint32_t target_config_id = 0;
      const auto exact = pending_frames_.find(static_cast<uint32_t>(expected_frame_index_));
      if (exact != pending_frames_.end()) {
        const auto exact_header = mlvc::transport::ParseMediaUnitHeader(exact->second);
        if (mlvc::transport::CompareMlvcSerial32(exact_header.config_id, config_id_) ==
            mlvc::transport::MlvcSerial32Order::kNewer) {
          target_config_id = exact_header.config_id;
        }
      } else if (const auto recovery = FindRecoveryFrame(); recovery.has_value()) {
        const auto recovery_header =
            mlvc::transport::ParseMediaUnitHeader(pending_frames_.at(*recovery));
        if (mlvc::transport::CompareMlvcSerial32(recovery_header.config_id, config_id_) ==
            mlvc::transport::MlvcSerial32Order::kNewer) {
          target_config_id = recovery_header.config_id;
        }
      }
      if (target_config_id != 0 &&
          pending_frames_.find(static_cast<uint32_t>(expected_frame_index_)) ==
              pending_frames_.end() &&
          !waiting_for_recovery_) {
        if (!reorder_deadline_.has_value()) {
          reorder_deadline_ = std::chrono::steady_clock::now() + kRtpFrameReorderWait;
        }
        if (std::chrono::steady_clock::now() < *reorder_deadline_) {
          target_config_id = 0;
        } else {
          MarkDependencyGap();
        }
      }
      const bool switching_past_missing_frame =
          target_config_id != 0 &&
          pending_frames_.find(static_cast<uint32_t>(expected_frame_index_)) ==
              pending_frames_.end();
      if (target_config_id != 0) {
        auto target = std::find_if(
            pending_config_units_.begin(), pending_config_units_.end(),
            [target_config_id](const std::vector<uint8_t>& queued) {
              return mlvc::transport::ParseScu(queued).config_id == target_config_id;
            });
        if (target != pending_config_units_.end()) {
          // If an intermediate switch and its RA frame were both lost, a
          // later complete SCU + random-access EFU is a valid recovery point.
          // Skip older queued SCUs only at that independently decodable RA.
          pending_config_units_.erase(
              std::remove_if(pending_config_units_.begin(), pending_config_units_.end(),
                             [target_config_id](const std::vector<uint8_t>& queued) {
                               const auto order = mlvc::transport::CompareMlvcSerial32(
                                   target_config_id,
                                   mlvc::transport::ParseScu(queued).config_id);
                               return order == mlvc::transport::MlvcSerial32Order::kNewer;
                             }),
              pending_config_units_.end());
          target = std::find_if(
              pending_config_units_.begin(), pending_config_units_.end(),
              [target_config_id](const std::vector<uint8_t>& queued) {
                return mlvc::transport::ParseScu(queued).config_id == target_config_id;
              });
          Check(target != pending_config_units_.end(),
                "RTP recovery SCU disappeared from the reorder queue");
          message = std::move(*target);
          pending_config_units_.erase(target);
          HandleConfigurationUnit(message);
          if (switching_past_missing_frame) waiting_for_recovery_ = true;
          continue;
        }
      }
    }
    const bool expected_frame_pending =
        pending_frames_.find(static_cast<uint32_t>(expected_frame_index_)) != pending_frames_.end();
    if (!expected_frame_pending && HasPendingCurrentFrameAfterExpected() &&
        !waiting_for_recovery_ && !reorder_deadline_.has_value()) {
      reorder_deadline_ = std::chrono::steady_clock::now() + kRtpFrameReorderWait;
    }
    if (!expected_frame_pending && reorder_deadline_.has_value() &&
        std::chrono::steady_clock::now() >= *reorder_deadline_) {
      MarkDependencyGap();
      continue;
    }
    if (!expected_frame_pending && waiting_for_recovery_) {
      const auto recovery = FindRecoveryFrame();
      if (recovery.has_value()) {
        const auto recovery_header =
            mlvc::transport::ParseMediaUnitHeader(pending_frames_.at(*recovery));
        if (recovery_header.config_id == config_id_) {
          expected_frame_index_ = static_cast<int>(*recovery);
          DiscardPendingFramesBefore(*recovery);
          waiting_for_recovery_ = false;
          reorder_deadline_.reset();
          continue;
        }
      }
    }
    if (!PopReadyFrame(&message)) {
      if (eos_frame_count_.has_value() &&
          static_cast<uint32_t>(expected_frame_index_) == *eos_frame_count_ &&
          pending_frames_.empty() && pending_config_units_.empty() && pending_config_id_ == 0) {
        return false;
      }
      bool received = false;
      if (reorder_deadline_.has_value()) {
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            *reorder_deadline_ - std::chrono::steady_clock::now());
        if (remaining <= std::chrono::milliseconds::zero()) {
          MarkDependencyGap();
          continue;
        }
        received = receiver_.ReceiveFor(&message, remaining);
        if (!received) {
          MarkDependencyGap();
          continue;
        }
      } else if (eos_frame_count_.has_value()) {
        received = receiver_.ReceiveFor(&message, std::chrono::seconds(2));
        Check(received,
              "RTP EOS frame boundary was not reached before the missing-frame timeout");
      } else {
        received = receiver_.Receive(&message);
      }
      if (!received) return false;
      Check(!message.empty(), "empty MLVC RTP media unit");
      const auto common = mlvc::transport::ParseMediaUnitHeader(message);
      if (common.unit_type == mlvc::transport::MlvcMediaUnitType::kEos) {
        mlvc::transport::ValidateMediaUnit(message);
        const auto eos_config_order =
            mlvc::transport::CompareMlvcSerial32(common.config_id, config_id_);
        Check(eos_config_order != mlvc::transport::MlvcSerial32Order::kAmbiguous,
              "RTP EOS configuration ID is serial-number ambiguous");
        if (eos_config_order == mlvc::transport::MlvcSerial32Order::kOlder) {
          continue;  // Delayed EOS from an obsolete configuration.
        } else if (eos_config_order == mlvc::transport::MlvcSerial32Order::kSame) {
          HandleEndUnit(message);
        } else {
          if (pending_eos_unit_.has_value()) {
            Check(*pending_eos_unit_ == message,
                  "conflicting duplicate RTP EOS unit");
          } else {
            pending_eos_unit_ = message;
          }
        }
        continue;
      }
      if (common.unit_type == mlvc::transport::MlvcMediaUnitType::kScu) {
        const auto scu = mlvc::transport::ParseScu(message);
        const auto config_order =
            mlvc::transport::CompareMlvcSerial32(scu.config_id, config_id_);
        Check(config_order != mlvc::transport::MlvcSerial32Order::kAmbiguous,
              "RTP SCU configuration ID is serial-number ambiguous");
        if (config_order == mlvc::transport::MlvcSerial32Order::kNewer) {
          QueueConfigurationUnit(std::move(message), scu.config_id);
        } else if (config_order == mlvc::transport::MlvcSerial32Order::kSame) {
          HandleConfigurationUnit(message);
        }
        continue;
      }
      if (common.unit_type != mlvc::transport::MlvcMediaUnitType::kEfu) continue;
      const auto efu = mlvc::transport::ParseEfu(message);
      BufferFrameUnit(message, efu.frame_id);
      continue;
    }
    const auto efu = mlvc::transport::ParseEfu(message);
    Check(efu.config_id == config_id_, "RTP EFU references an unknown SCU");
    if (pending_config_id_ != 0) {
      Check(efu.config_id == pending_config_id_ && efu.frame_type == 0 &&
                (efu.unit_flags & (mlvc::transport::kEfuRandomAccess |
                                   mlvc::transport::kEfuResetReference)) ==
                    (mlvc::transport::kEfuRandomAccess | mlvc::transport::kEfuResetReference),
            "RTP configuration switch was not followed by a random-access EFU");
      pending_config_id_ = 0;
    }
    Check(efu.frame_id <= static_cast<uint32_t>(std::numeric_limits<int>::max()),
          "RTP frame ID exceeds the application range");
    *frame_index = static_cast<int>(efu.frame_id);
    *frame_type = static_cast<mlvc::codec::MlvcFrameType>(efu.frame_type);
    *q_index = efu.entropy_q_index;
    if (*frame_index != expected_frame_index_) {
      mlvc::Check(false,
                  "MLVC RTP frame index must be consecutive: expected " +
                      std::to_string(expected_frame_index_) + ", got " +
                      std::to_string(*frame_index) + " (lost=" +
                      std::to_string(receiver_.lost_packets()) + ", duplicate=" +
                      std::to_string(receiver_.duplicate_packets()) + ", reordered=" +
                      std::to_string(receiver_.reordered_packets()) + ")");
    }
    mlvc::io::ValidateMlvcFrameMetadata(*frame_index, *frame_type, *q_index, expected_frame_index_,
                                        expected_frame_index_ == 0);
    const bool random_access = (efu.unit_flags & mlvc::transport::kEfuRandomAccess) != 0;
    const bool reset_reference = (efu.unit_flags & mlvc::transport::kEfuResetReference) != 0;
    const bool store_as_ltr = (efu.unit_flags & mlvc::transport::kEfuStoreAsLtr) != 0;
    if (random_access) {
      Check(*frame_type == mlvc::codec::MlvcFrameType::kIFrame,
            "RTP random-access flag is only valid on an I-frame");
      available_short_reference_id_.reset();
      ltr_frame_ids_.clear();
    } else if (reset_reference) {
      // A non-I reset frame uses the zero short-reference feature.  LTRs are
      // independent and remain usable for an explicit LTR recovery frame.
      available_short_reference_id_.reset();
    }
    if (*frame_type == mlvc::codec::MlvcFrameType::kPFrame && !reset_reference) {
      Check(efu.short_ref_frame_id != mlvc::transport::kMlvcNoReference &&
                available_short_reference_id_.has_value() &&
                efu.short_ref_frame_id == *available_short_reference_id_,
            "RTP P frame references an unavailable short-reference frame");
    }
    if (*frame_type == mlvc::codec::MlvcFrameType::kLtrRecovery) {
      Check(efu.long_ref_frame_id != mlvc::transport::kMlvcNoReference &&
                ltr_frame_ids_.find(efu.long_ref_frame_id) != ltr_frame_ids_.end(),
            "RTP LTR recovery references an unavailable LTR frame");
    }
    Check(efu.entropy_payload.size() <= max_payload_size_,
          "MLVC RTP frame payload exceeds maximum for resolution");
    *payload = efu.entropy_payload;
    last_frame_metadata_.explicit_metadata = true;
    last_frame_metadata_.model_q_index = efu.model_q_index;
    last_frame_metadata_.unit_flags = efu.unit_flags;
    last_frame_metadata_.short_ref_frame_id = efu.short_ref_frame_id;
    last_frame_metadata_.long_ref_frame_id = efu.long_ref_frame_id;
    last_frame_metadata_.pts = efu.pts;
    RememberDecodedFrame(efu.frame_id, store_as_ltr);
    Check(expected_frame_index_ < std::numeric_limits<int>::max(),
          "RTP frame index reached the application limit");
    ++expected_frame_index_;
    recent_frames_[efu.frame_id] = {message.size(), mlvc::transport::MlvcCrc32c(message)};
    constexpr std::size_t kMaxRecentFrameFingerprints = 4096;
    while (recent_frames_.size() > kMaxRecentFrameFingerprints) {
      recent_frames_.erase(recent_frames_.begin());
    }
    return true;
  }
}

}  // namespace mlvc::io
