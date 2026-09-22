#include "mlvc/io/udp_frame_transport.h"

#include <cstring>
#include <limits>
#include <utility>

#include "mlvc/core/status.h"

namespace mlvc::io {
namespace {
constexpr uint8_t kHeaderMessage = 1;
constexpr uint8_t kFrameMessage = 2;
constexpr uint8_t kEndMessage = 3;
constexpr uint8_t kRawYuvMessage = 4;

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

}  // namespace

UdpMlvcSender::UdpMlvcSender(const std::string& host, uint16_t port,
                             mlvc::transport::UdpSendOptions options)
    : sender_(host, port, "MLVC", options) {}

UdpMlvcSender::~UdpMlvcSender() = default;

void UdpMlvcSender::SendHeader(const MlvcBitstreamHeader& header) {
  Check(!closed_, "MLVC UDP sender is closed");
  MlvcBitstreamHeader normalized = header;
  if (normalized.version == 0) normalized.version = 3;
  ValidateMlvcBitstreamHeader(normalized);
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
}

void UdpMlvcSender::SendFrame(int frame_index, mlvc::codec::MlvcFrameType frame_type, int q_index,
                              const std::vector<uint8_t>& payload) {
  Check(!closed_, "MLVC UDP sender is closed");
  ValidateMlvcFrameMetadata(frame_index, frame_type, q_index, frame_index, frame_index == 0);
  Check(payload.size() <= MaxMlvcFramePayloadBytes(8192, 8192),
        "MLVC frame payload exceeds the maximum supported payload size");
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
}

void UdpMlvcSender::SendEnd() {
  Check(!closed_, "MLVC UDP sender is closed");
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
                             uint64_t max_queue_delay_ms)
    : sender_(host, port, 0x4d4c5643u, pacing_rate_bps, max_burst_bytes, max_queue_bytes,
              max_queue_delay_ms) {}
RtpMlvcSender::~RtpMlvcSender() = default;
void RtpMlvcSender::SendHeader(const MlvcBitstreamHeader& header) {
  Check(!closed_, "MLVC RTP sender is closed");
  MlvcBitstreamHeader normalized = header;
  if (normalized.version == 0) normalized.version = 3;
  ValidateMlvcBitstreamHeader(normalized);
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
  sender_.SendUnit(mlvc::transport::RtpUnitType::kScu, 0, 1, 0, 0, message);
}
void RtpMlvcSender::SendFrame(int frame_index, mlvc::codec::MlvcFrameType frame_type, int q_index,
                              const std::vector<uint8_t>& payload) {
  Check(!closed_, "MLVC RTP sender is closed");
  ValidateMlvcFrameMetadata(frame_index, frame_type, q_index, frame_index, frame_index == 0);
  Check(payload.size() <= MaxMlvcFramePayloadBytes(8192, 8192),
        "RTP MLVC frame payload exceeds the maximum supported payload size");
  Check(payload.size() <= std::numeric_limits<uint32_t>::max(),
        "RTP MLVC frame payload is too large");
  std::vector<uint8_t> message;
  message.reserve(14 + payload.size());
  message.push_back(kFrameMessage);
  PutI32(&message, frame_index);
  message.push_back(static_cast<uint8_t>(frame_type));
  PutI32(&message, q_index);
  PutU32(&message, static_cast<uint32_t>(payload.size()));
  message.insert(message.end(), payload.begin(), payload.end());
  sender_.SendUnit(mlvc::transport::RtpUnitType::kEfu, 0, 1, static_cast<uint32_t>(frame_index + 1),
                   static_cast<uint32_t>(frame_index * 3000), message);
}
void RtpMlvcSender::SendEnd() {
  Check(!closed_, "MLVC RTP sender is closed");
  // EOS is a single small RTP unit.  Repeat it so a late or lost terminal
  // datagram cannot leave the receiver blocked after all frames were decoded.
  for (int attempt = 0; attempt < 3; ++attempt) {
    sender_.SendUnit(mlvc::transport::RtpUnitType::kEos, 0, 1, 0xffffffffu,
                     static_cast<uint32_t>(attempt), {kEndMessage});
  }
}

void RtpMlvcSender::Close() {
  if (closed_) return;
  sender_.Close();
  closed_ = true;
}

RtpMlvcReceiver::RtpMlvcReceiver(uint16_t port) : receiver_(port, "MLVC RTP") {}
RtpMlvcReceiver::~RtpMlvcReceiver() = default;
MlvcBitstreamHeader RtpMlvcReceiver::ReceiveHeader() {
  std::vector<uint8_t> message;
  while (!receiver_.Receive(&message)) {
  }
  Check((message.size() == 57 || message.size() == 65) && message[0] == kHeaderMessage,
        "invalid MLVC RTP header message");
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
    Check(message.size() == 65, "invalid MLVC RTP v4 header size");
    header.forced_ltr_recovery_frame = ReadI32(cursor);
    cursor += 4;
    header.forced_ltr_reference_frame = ReadI32(cursor);
  } else {
    Check(message.size() == 57, "invalid MLVC RTP legacy header size");
  }
  mlvc::io::ValidateMlvcBitstreamHeader(header);
  Check(header.version >= 2 && header.version <= 4, "unsupported MLVC RTP header version");
  header_ = header;
  max_payload_size_ = mlvc::io::MaxMlvcFramePayloadBytes(header.width, header.height);
  expected_frame_index_ = 0;
  header_received_ = true;
  return header;
}
bool RtpMlvcReceiver::ReceiveFrame(int* frame_index, mlvc::codec::MlvcFrameType* frame_type,
                                   int* q_index, std::vector<uint8_t>* payload) {
  Check(frame_index && frame_type && q_index && payload, "RTP frame outputs are required");
  Check(header_received_, "MLVC RTP header must be received before frames");
  std::vector<uint8_t> message;
  for (;;) {
    while (!receiver_.Receive(&message)) {
    }
    Check(!message.empty(), "empty MLVC RTP message");
    if (message[0] == kEndMessage) {
      Check(message.size() == 1, "invalid MLVC RTP end message");
      return false;
    }
    if (message[0] != kFrameMessage) continue;
    Check(message.size() >= 14, "invalid MLVC RTP frame message");
    const uint8_t* cursor = message.data() + 1;
    *frame_index = ReadI32(cursor);
    cursor += 4;
    const uint8_t raw_type = *cursor++;
    Check(raw_type <= static_cast<uint8_t>(mlvc::codec::MlvcFrameType::kLtrRecovery),
          "invalid MLVC RTP frame type");
    *frame_type = static_cast<mlvc::codec::MlvcFrameType>(raw_type);
    *q_index = ReadI32(cursor);
    cursor += 4;
    mlvc::io::ValidateMlvcFrameMetadata(*frame_index, *frame_type, *q_index, expected_frame_index_,
                                        expected_frame_index_ == 0);
    const uint32_t payload_size = ReadU32(cursor);
    cursor += 4;
    Check(payload_size == message.size() - 14, "invalid MLVC RTP frame payload size");
    Check(payload_size <= max_payload_size_,
          "MLVC RTP frame payload exceeds maximum for resolution");
    payload->assign(cursor, cursor + payload_size);
    ++expected_frame_index_;
    return true;
  }
}

RawYuvUdpSender::RawYuvUdpSender(const std::string& host, uint16_t port)
    : sender_(host, port, "raw YUV forwarder") {
  send_thread_ = std::thread(&RawYuvUdpSender::SendLoop, this);
}

RawYuvUdpSender::~RawYuvUdpSender() {
  try {
    Close();
  } catch (...) {
  }
}

void RawYuvUdpSender::Close() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (closed_) return;
    stopping_ = true;
    closed_ = true;
  }
  queue_cv_.notify_all();
  if (send_thread_.joinable()) {
    send_thread_.join();
  }
  std::exception_ptr error;
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    error = send_error_;
  }
  if (error != nullptr) std::rethrow_exception(error);
  sender_.Flush();
}

void RawYuvUdpSender::SendFrame(const mlvc::codec::TensorData& tensor, int frame_index) {
  Check(tensor.dtype == mlvc::DataType::kFloat16, "raw YUV forward expects FP16 tensor");
  Check(tensor.shape.rank() == 4 && tensor.shape.dim(0) == 1 && tensor.shape.dim(1) == 3,
        "raw YUV forward expects NCHW 3-plane tensor");
  Check(tensor.bytes.size() <= std::numeric_limits<uint32_t>::max(), "raw YUV tensor is too large");
  std::vector<uint8_t> message;
  message.reserve(26 + tensor.bytes.size());
  message.push_back(kRawYuvMessage);
  PutI32(&message, frame_index);
  PutU32(&message, static_cast<uint32_t>(tensor.shape.dim(3)));
  PutU32(&message, static_cast<uint32_t>(tensor.shape.dim(2)));
  PutU32(&message, static_cast<uint32_t>(tensor.shape.dim(3)));
  PutU32(&message, static_cast<uint32_t>(tensor.shape.dim(2)));
  message.push_back(1);  // FP16 YUV444 NCHW.
  PutU32(&message, static_cast<uint32_t>(tensor.bytes.size()));
  message.insert(message.end(), tensor.bytes.begin(), tensor.bytes.end());
  std::lock_guard<std::mutex> lock(queue_mutex_);
  if (send_error_ != nullptr) {
    std::rethrow_exception(send_error_);
  }
  Check(!stopping_, "raw YUV forwarder is stopping");
  constexpr std::size_t kMaxQueuedFrames = 4;
  if (send_queue_.size() >= kMaxQueuedFrames) {
    ++dropped_frames_;
    return;
  }
  send_queue_.push_back(PendingFrame{std::move(message)});
  queue_cv_.notify_one();
}

void RawYuvUdpSender::SendLoop() {
  for (;;) {
    PendingFrame pending;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock, [this] { return stopping_ || !send_queue_.empty(); });
      if (send_queue_.empty()) {
        return;
      }
      pending = std::move(send_queue_.front());
      send_queue_.pop_front();
    }
    try {
      sender_.Send(pending.message);
    } catch (...) {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      send_error_ = std::current_exception();
      stopping_ = true;
      queue_cv_.notify_all();
      return;
    }
  }
}

}  // namespace mlvc::io
