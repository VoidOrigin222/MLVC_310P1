#include <mlvc/io/mlvc_bitstream.h>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "mlvc/core/status.h"

namespace mlvc::io {
namespace {

constexpr std::array<char, 8> kMagic = {'M', 'L', 'V', 'C', 'B', 'S', 'T', '1'};
constexpr uint32_t kVersion = 4;

// Payload size limits to prevent memory exhaustion
constexpr uint64_t kMaxPayloadSize720p = 8 * 1024 * 1024;    // 8MB for 720p
constexpr uint64_t kMaxPayloadSize1080p = 16 * 1024 * 1024;  // 16MB for 1080p
constexpr uint64_t kMaxPayloadSize4K = 64 * 1024 * 1024;     // 64MB for 4K

// Resolution limits
constexpr int kMinResolution = 64;
constexpr int kMaxResolution = 8192;

// QP range (0-63 for H.264/H.265 compatibility)
constexpr int kMinQp = 0;
constexpr int kMaxQp = 63;

template <typename T>
void WritePod(std::ostream& out, const T& value, const char* what) {
  out.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(T)));
  Check(out.good(), std::string("failed to write mlvc bitstream ") + what);
}

template <typename T>
T ReadPod(std::istream& in, const char* what) {
  T value{};
  in.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
  Check(in.good(), std::string("failed to read mlvc bitstream ") + what);
  return value;
}

uint64_t MaxMlvcFramePayloadBytesImpl(int width, int height) {
  const int64_t pixels = static_cast<int64_t>(width) * height;
  if (pixels <= 1280 * 720) {
    return kMaxPayloadSize720p;
  } else if (pixels <= 1920 * 1080) {
    return kMaxPayloadSize1080p;
  } else {
    return kMaxPayloadSize4K;
  }
}

void ValidateMlvcBitstreamHeaderImpl(const MlvcBitstreamHeader& header) {
  Check(header.version == 0 || (header.version >= 2 && header.version <= kVersion),
        "MLVC bitstream version must be 0, 2, 3, or 4, got " + std::to_string(header.version));
  Check(header.width >= kMinResolution && header.width <= kMaxResolution,
        "MLVC bitstream width must be in [" + std::to_string(kMinResolution) + ", " +
            std::to_string(kMaxResolution) + "], got " + std::to_string(header.width));
  Check(header.height >= kMinResolution && header.height <= kMaxResolution,
        "MLVC bitstream height must be in [" + std::to_string(kMinResolution) + ", " +
            std::to_string(kMaxResolution) + "], got " + std::to_string(header.height));
  Check(std::isfinite(header.fps) && header.fps > 0.0,
        "MLVC bitstream fps must be finite and positive, got " + std::to_string(header.fps));
  Check(header.q_index >= kMinQp && header.q_index <= kMaxQp,
        "MLVC bitstream q_index must be in [" + std::to_string(kMinQp) + ", " +
            std::to_string(kMaxQp) + "], got " + std::to_string(header.q_index));
  Check(header.gop >= 0,
        "MLVC bitstream gop must be non-negative, got " + std::to_string(header.gop));
  Check(header.reset_interval >= 0, "MLVC bitstream reset_interval must be non-negative, got " +
                                        std::to_string(header.reset_interval));
  Check(header.ltr_start_idx >= 0, "MLVC bitstream ltr_start_idx must be non-negative, got " +
                                       std::to_string(header.ltr_start_idx));
  Check(header.ltr_period >= 0,
        "MLVC bitstream ltr_period must be non-negative, got " + std::to_string(header.ltr_period));
  Check(header.ltr_qp_shift >= kMinQp && header.ltr_qp_shift <= kMaxQp,
        "MLVC bitstream ltr_qp_shift must be in [" + std::to_string(kMinQp) + ", " +
            std::to_string(kMaxQp) + "], got " + std::to_string(header.ltr_qp_shift));
  Check(std::isfinite(header.target_bitrate_bps) && header.target_bitrate_bps >= 0.0,
        "MLVC bitstream target_bitrate_bps must be finite and non-negative, got " +
            std::to_string(header.target_bitrate_bps));
  Check(header.forced_ltr_recovery_frame >= -1 && header.forced_ltr_reference_frame >= -1,
        "MLVC forced LTR frame indices must be -1 or non-negative");
}

void ValidateMlvcFrameMetadataImpl(int frame_index, mlvc::codec::MlvcFrameType frame_type,
                                   int q_index, int expected_frame_index, bool require_i_frame) {
  Check(frame_index >= 0,
        "MLVC bitstream frame index must be non-negative, got " + std::to_string(frame_index));
  Check(frame_index == expected_frame_index,
        "MLVC bitstream frame index must be consecutive: expected " +
            std::to_string(expected_frame_index) + ", got " + std::to_string(frame_index));
  Check(static_cast<uint8_t>(frame_type) <=
            static_cast<uint8_t>(mlvc::codec::MlvcFrameType::kLtrRecovery),
        "invalid MLVC frame type");
  Check(q_index >= kMinQp && q_index <= kMaxQp,
        "MLVC bitstream q_index must be in [" + std::to_string(kMinQp) + ", " +
            std::to_string(kMaxQp) + "], got " + std::to_string(q_index));
  if (require_i_frame) {
    Check(frame_type == mlvc::codec::MlvcFrameType::kIFrame,
          "first MLVC bitstream frame must be an I-frame");
  }
}

}  // namespace

uint64_t MaxMlvcFramePayloadBytes(int width, int height) {
  return MaxMlvcFramePayloadBytesImpl(width, height);
}

void ValidateMlvcBitstreamHeader(const MlvcBitstreamHeader& header) {
  ValidateMlvcBitstreamHeaderImpl(header);
}

void ValidateMlvcFrameMetadata(int frame_index, mlvc::codec::MlvcFrameType frame_type, int q_index,
                               int expected_frame_index, bool require_i_frame) {
  ValidateMlvcFrameMetadataImpl(frame_index, frame_type, q_index, expected_frame_index,
                                require_i_frame);
}

OfficialMlvcBitstreamWriter::OfficialMlvcBitstreamWriter(const std::filesystem::path& path)
    : path_(path) {
  if (!path_.parent_path().empty()) {
    std::filesystem::create_directories(path_.parent_path());
  }
  output_.open(path_, std::ios::binary | std::ios::trunc);
  Check(output_.good(), "failed to open official MLVC bitstream output: " + path_.string());
}

OfficialMlvcBitstreamWriter::~OfficialMlvcBitstreamWriter() { Close(); }

void OfficialMlvcBitstreamWriter::WriteFrame(int q_index, const std::vector<uint8_t>& payload) {
  Check(!closed_, "write on closed official MLVC bitstream");
  Check(payload.size() <= static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()),
        "official MLVC frame payload is too large");
  WritePod(output_, static_cast<int32_t>(q_index), "q_index");
  WritePod(output_, static_cast<uint32_t>(payload.size()), "payload_size");
  output_.write(reinterpret_cast<const char*>(payload.data()),
                static_cast<std::streamsize>(payload.size()));
  Check(output_.good(), "failed to write official MLVC payload");
}

void OfficialMlvcBitstreamWriter::Close() {
  if (closed_) {
    return;
  }
  if (output_.is_open()) {
    output_.flush();
    Check(output_.good(), "failed to flush official MLVC bitstream: " + path_.string());
    output_.close();
  }
  closed_ = true;
}

OfficialMlvcBitstreamReader::OfficialMlvcBitstreamReader(const std::filesystem::path& path)
    : path_(path) {
  input_.open(path_, std::ios::binary);
  Check(input_.good(), "failed to open official MLVC bitstream input: " + path_.string());
}

bool OfficialMlvcBitstreamReader::ReadFrame(int* frame_index, int* q_index,
                                            std::vector<uint8_t>* payload) {
  Check(frame_index != nullptr && q_index != nullptr && payload != nullptr,
        "official MLVC frame outputs are required");
  if (input_.peek() == std::char_traits<char>::eof()) {
    return false;
  }
  *frame_index = next_frame_index_++;
  *q_index = static_cast<int>(ReadPod<int32_t>(input_, "q_index"));

  const uint32_t bytes = ReadPod<uint32_t>(input_, "payload_size");
  Check(bytes <= kMaxPayloadSize4K,
        "official MLVC frame payload size exceeds maximum: " + std::to_string(bytes) + " > " +
            std::to_string(kMaxPayloadSize4K));

  const std::streampos current_pos = input_.tellg();
  input_.seekg(0, std::ios::end);
  const std::streampos file_size = input_.tellg();
  input_.seekg(current_pos);
  Check(current_pos + static_cast<std::streamoff>(bytes) <= file_size,
        "official MLVC frame payload size exceeds remaining file size: " + std::to_string(bytes) +
            " > " + std::to_string(file_size - current_pos));

  payload->resize(bytes);
  input_.read(reinterpret_cast<char*>(payload->data()), static_cast<std::streamsize>(bytes));
  Check(input_.good(), "failed to read official MLVC payload");
  return true;
}

MlvcBitstreamWriter::MlvcBitstreamWriter(const std::filesystem::path& path,
                                         const MlvcBitstreamHeader& header)
    : path_(path) {
  ValidateMlvcBitstreamHeader(header);

  if (!path_.parent_path().empty()) {
    std::filesystem::create_directories(path_.parent_path());
  }
  output_.open(path_, std::ios::binary | std::ios::trunc);
  Check(output_.good(), "failed to open mlvc bitstream output: " + path_.string());
  output_.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
  Check(output_.good(), "failed to write mlvc bitstream magic");
  WritePod(output_, static_cast<uint32_t>(header.version == 0 ? kVersion : header.version),
           "version");
  WritePod(output_, static_cast<int32_t>(header.width), "width");
  WritePod(output_, static_cast<int32_t>(header.height), "height");
  WritePod(output_, header.fps, "fps");
  WritePod(output_, static_cast<int32_t>(header.q_index), "q_index");
  WritePod(output_, static_cast<int32_t>(header.gop), "gop");
  WritePod(output_, static_cast<int32_t>(header.reset_interval), "reset_interval");
  WritePod(output_, static_cast<int32_t>(header.ltr_start_idx), "ltr_start_idx");
  WritePod(output_, static_cast<int32_t>(header.ltr_period), "ltr_period");
  WritePod(output_, static_cast<int32_t>(header.ltr_qp_shift), "ltr_qp_shift");
  WritePod(output_, header.target_bitrate_bps, "target_bitrate_bps");
  WritePod(output_, static_cast<uint32_t>(header.flags), "flags");
  if ((header.version == 0 ? kVersion : header.version) >= 4) {
    WritePod(output_, static_cast<int32_t>(header.forced_ltr_recovery_frame),
             "forced_ltr_recovery_frame");
    WritePod(output_, static_cast<int32_t>(header.forced_ltr_reference_frame),
             "forced_ltr_reference_frame");
  }
}

MlvcBitstreamWriter::~MlvcBitstreamWriter() { Close(); }

void MlvcBitstreamWriter::WriteFrame(int frame_index, mlvc::codec::MlvcFrameType frame_type,
                                     int q_index, const std::vector<uint8_t>& payload) {
  Check(!closed_, "write on closed mlvc bitstream");
  WritePod(output_, static_cast<int32_t>(frame_index), "frame_index");
  WritePod(output_, static_cast<uint8_t>(frame_type), "frame_type");
  WritePod(output_, static_cast<int32_t>(q_index), "q_index");
  const uint64_t bytes = static_cast<uint64_t>(payload.size());
  WritePod(output_, bytes, "payload_size");
  output_.write(reinterpret_cast<const char*>(payload.data()),
                static_cast<std::streamsize>(payload.size()));
  Check(output_.good(), "failed to write mlvc bitstream payload");
}

void MlvcBitstreamWriter::Close() {
  if (closed_) {
    return;
  }
  if (output_.is_open()) {
    output_.flush();
    Check(output_.good(), "failed to flush mlvc bitstream output: " + path_.string());
    output_.close();
  }
  closed_ = true;
}

MlvcBitstreamReader::MlvcBitstreamReader(const std::filesystem::path& path) : path_(path) {
  input_.open(path_, std::ios::binary);
  Check(input_.good(), "failed to open mlvc bitstream input: " + path_.string());

  std::array<char, 8> magic{};
  input_.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  Check(input_.good(), "failed to read mlvc bitstream magic: " + path_.string());
  Check(magic == kMagic, "invalid mlvc bitstream magic: " + path_.string());

  const uint32_t version = ReadPod<uint32_t>(input_, "version");
  Check(version >= 2 && version <= kVersion, "unsupported mlvc bitstream version");
  version_ = version;
  header_.version = version_;
  header_.width = static_cast<int>(ReadPod<int32_t>(input_, "width"));
  header_.height = static_cast<int>(ReadPod<int32_t>(input_, "height"));
  header_.fps = ReadPod<double>(input_, "fps");
  header_.q_index = static_cast<int>(ReadPod<int32_t>(input_, "q_index"));

  if (version_ >= 3) {
    header_.gop = static_cast<int>(ReadPod<int32_t>(input_, "gop"));
    header_.reset_interval = static_cast<int>(ReadPod<int32_t>(input_, "reset_interval"));
    header_.ltr_start_idx = static_cast<int>(ReadPod<int32_t>(input_, "ltr_start_idx"));
    header_.ltr_period = static_cast<int>(ReadPod<int32_t>(input_, "ltr_period"));
    header_.ltr_qp_shift = static_cast<int>(ReadPod<int32_t>(input_, "ltr_qp_shift"));
    header_.target_bitrate_bps = ReadPod<double>(input_, "target_bitrate_bps");
    header_.flags = ReadPod<uint32_t>(input_, "flags");
    if (version_ >= 4) {
      header_.forced_ltr_recovery_frame =
          static_cast<int>(ReadPod<int32_t>(input_, "forced_ltr_recovery_frame"));
      header_.forced_ltr_reference_frame =
          static_cast<int>(ReadPod<int32_t>(input_, "forced_ltr_reference_frame"));
    }
  }

  ValidateMlvcBitstreamHeader(header_);
  max_payload_size_ = MaxMlvcFramePayloadBytes(header_.width, header_.height);
}

MlvcBitstreamReader::~MlvcBitstreamReader() = default;

bool MlvcBitstreamReader::ReadFrame(int* frame_index, mlvc::codec::MlvcFrameType* frame_type,
                                    int* q_index, std::vector<uint8_t>* payload) {
  Check(frame_index != nullptr, "mlvc bitstream frame index output is required");
  Check(frame_type != nullptr && q_index != nullptr && payload != nullptr,
        "mlvc bitstream frame outputs are required");

  if (input_.peek() == std::char_traits<char>::eof()) {
    return false;
  }

  const int read_frame_index = static_cast<int>(ReadPod<int32_t>(input_, "frame_index"));
  const uint8_t type_byte = ReadPod<uint8_t>(input_, "frame_type");
  mlvc::codec::MlvcFrameType read_frame_type;
  if (version_ >= 3) {
    Check(type_byte <= static_cast<uint8_t>(mlvc::codec::MlvcFrameType::kLtrRecovery),
          "invalid mlvc frame type");
    read_frame_type = static_cast<mlvc::codec::MlvcFrameType>(type_byte);
  } else {
    read_frame_type =
        type_byte != 0 ? mlvc::codec::MlvcFrameType::kIFrame : mlvc::codec::MlvcFrameType::kPFrame;
  }

  const int read_q_index = static_cast<int>(ReadPod<int32_t>(input_, "q_index"));
  ValidateMlvcFrameMetadata(read_frame_index, read_frame_type, read_q_index, expected_frame_index_,
                            expected_frame_index_ == 0);

  const uint64_t bytes = ReadPod<uint64_t>(input_, "payload_size");
  Check(bytes <= max_payload_size_,
        "MLVC frame payload size exceeds maximum for resolution " + std::to_string(header_.width) +
            "x" + std::to_string(header_.height) + ": " + std::to_string(bytes) + " > " +
            std::to_string(max_payload_size_));

  const std::streampos current_pos = input_.tellg();
  input_.seekg(0, std::ios::end);
  const std::streampos file_size = input_.tellg();
  input_.seekg(current_pos);
  Check(current_pos + static_cast<std::streamoff>(bytes) <= file_size,
        "MLVC frame payload size exceeds remaining file size: " + std::to_string(bytes) + " > " +
            std::to_string(file_size - current_pos));

  payload->resize(static_cast<std::size_t>(bytes));
  input_.read(reinterpret_cast<char*>(payload->data()), static_cast<std::streamsize>(bytes));
  Check(input_.good(), "failed to read mlvc bitstream payload");

  *frame_index = read_frame_index;
  *frame_type = read_frame_type;
  *q_index = read_q_index;
  ++expected_frame_index_;

  return true;
}

}  // namespace mlvc::io
