#include <ulbvc/io/ulbvc_bitstream.h>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include "ulbvc/core/status.h"

namespace ulbvc::io {
namespace {

constexpr std::array<char, 8> kMagic = {'U', 'L', 'B', 'V', 'C', 'B', 'S', '1'};
constexpr std::array<char, 8> kEsMagic = {'U', 'L', 'B', 'V', 'C', 'E', 'S', '1'};
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
  Check(out.good(), std::string("failed to write ulbvc bitstream ") + what);
}

template <typename T>
T ReadPod(std::istream& in, const char* what) {
  T value{};
  in.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
  Check(in.good(), std::string("failed to read ulbvc bitstream ") + what);
  return value;
}

uint64_t MaxUlbvcFramePayloadBytesImpl(int width, int height) {
  const int64_t pixels = static_cast<int64_t>(width) * height;
  if (pixels <= 1280 * 720) {
    return kMaxPayloadSize720p;
  } else if (pixels <= 1920 * 1080) {
    return kMaxPayloadSize1080p;
  } else {
    return kMaxPayloadSize4K;
  }
}

void ValidateUlbvcBitstreamHeaderImpl(const UlbvcBitstreamHeader& header) {
  Check(header.version == 0 || (header.version >= 2 && header.version <= kVersion),
        "ULBVC bitstream version must be 0, 2, 3, or 4, got " + std::to_string(header.version));
  Check(header.width >= kMinResolution && header.width <= kMaxResolution,
        "ULBVC bitstream width must be in [" + std::to_string(kMinResolution) + ", " +
            std::to_string(kMaxResolution) + "], got " + std::to_string(header.width));
  Check(header.height >= kMinResolution && header.height <= kMaxResolution,
        "ULBVC bitstream height must be in [" + std::to_string(kMinResolution) + ", " +
            std::to_string(kMaxResolution) + "], got " + std::to_string(header.height));
  const int coded_width = header.coded_width == 0 ? header.width : header.coded_width;
  const int coded_height = header.coded_height == 0 ? header.height : header.coded_height;
  Check(coded_width >= header.width && coded_width <= kMaxResolution,
        "ULBVC coded width must be at least visible width and within the supported limit, got " +
            std::to_string(header.coded_width));
  Check(coded_height >= header.height && coded_height <= kMaxResolution,
        "ULBVC coded height must be at least visible height and within the supported limit, got " +
            std::to_string(header.coded_height));
  Check(std::isfinite(header.fps) && header.fps >= 0.1 && header.fps <= 1000.0,
        "ULBVC bitstream fps must be finite and in [0.1, 1000], got " +
            std::to_string(header.fps));
  Check(header.q_index >= kMinQp && header.q_index <= kMaxQp,
        "ULBVC bitstream q_index must be in [" + std::to_string(kMinQp) + ", " +
            std::to_string(kMaxQp) + "], got " + std::to_string(header.q_index));
  Check(header.gop >= 0,
        "ULBVC bitstream gop must be non-negative, got " + std::to_string(header.gop));
  Check(header.reset_interval >= 0, "ULBVC bitstream reset_interval must be non-negative, got " +
                                        std::to_string(header.reset_interval));
  Check(header.ltr_start_idx >= 0, "ULBVC bitstream ltr_start_idx must be non-negative, got " +
                                       std::to_string(header.ltr_start_idx));
  Check(header.ltr_period >= 0,
        "ULBVC bitstream ltr_period must be non-negative, got " + std::to_string(header.ltr_period));
  Check(header.ltr_qp_shift >= kMinQp && header.ltr_qp_shift <= kMaxQp,
        "ULBVC bitstream ltr_qp_shift must be in [" + std::to_string(kMinQp) + ", " +
            std::to_string(kMaxQp) + "], got " + std::to_string(header.ltr_qp_shift));
  Check(std::isfinite(header.target_bitrate_bps) && header.target_bitrate_bps >= 0.0,
        "ULBVC bitstream target_bitrate_bps must be finite and non-negative, got " +
            std::to_string(header.target_bitrate_bps));
  Check(header.flags == 0, "ULBVC bitstream contains unsupported header flags");
  if (header.translation_warp) {
    Check(header.version == 0 || header.version == 4,
          "translation warp requires ULBVC-ES or RTP with SCU capability");
    Check(header.ltr_period == 0 && header.forced_ltr_recovery_frame == -1 &&
              header.forced_ltr_reference_frame == -1,
          "translation warp v1 does not support LTR recovery policy");
  }
  if (header.version > 0 && header.version < 4) {
    Check(header.forced_ltr_recovery_frame == -1 && header.forced_ltr_reference_frame == -1,
          "ULBVC bitstream versions before 4 cannot carry forced LTR metadata");
  }
  ulbvc::codec::ValidateForcedLtrConfiguration(
      header.gop, header.ltr_start_idx, header.ltr_period, header.forced_ltr_recovery_frame,
      header.forced_ltr_reference_frame);
}

void ValidateUlbvcFrameMetadataImpl(int frame_index, ulbvc::codec::UlbvcFrameType frame_type,
                                   int q_index, int expected_frame_index, bool require_i_frame) {
  Check(frame_index >= 0,
        "ULBVC bitstream frame index must be non-negative, got " + std::to_string(frame_index));
  Check(frame_index == expected_frame_index,
        "ULBVC bitstream frame index must be consecutive: expected " +
            std::to_string(expected_frame_index) + ", got " + std::to_string(frame_index));
  Check(static_cast<uint8_t>(frame_type) <=
            static_cast<uint8_t>(ulbvc::codec::UlbvcFrameType::kLtrRecovery),
        "invalid ULBVC frame type");
  Check(q_index >= kMinQp && q_index <= kMaxQp,
        "ULBVC bitstream q_index must be in [" + std::to_string(kMinQp) + ", " +
            std::to_string(kMaxQp) + "], got " + std::to_string(q_index));
  if (require_i_frame) {
    Check(frame_type == ulbvc::codec::UlbvcFrameType::kIFrame,
          "first ULBVC bitstream frame must be an I-frame");
  }
}

void PutBe16(std::ostream& output, uint16_t value) {
  const char bytes[2] = {static_cast<char>(value >> 8), static_cast<char>(value)};
  output.write(bytes, 2);
}

void PutBe32(std::vector<uint8_t>* output, uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8) output->push_back(static_cast<uint8_t>(value >> shift));
}

int ReadBe32ApplicationValue(const std::vector<uint8_t>& value, std::size_t offset,
                             const char* field, bool allow_unset = false) {
  Check(offset <= value.size() && value.size() - offset >= 4,
        std::string("truncated ULBVC SCU ") + field);
  const uint32_t raw = (static_cast<uint32_t>(value[offset]) << 24) |
                       (static_cast<uint32_t>(value[offset + 1]) << 16) |
                       (static_cast<uint32_t>(value[offset + 2]) << 8) |
                       static_cast<uint32_t>(value[offset + 3]);
  Check((allow_unset && raw == 0xffffffffu) ||
            raw <= static_cast<uint32_t>(std::numeric_limits<int>::max()),
        std::string("ULBVC SCU ") + field + " exceeds the application range");
  return allow_unset && raw == 0xffffffffu ? -1 : static_cast<int>(raw);
}

ulbvc::transport::UlbvcScu MakeScu(const UlbvcBitstreamHeader& header, uint32_t config_id) {
  ulbvc::transport::UlbvcScu scu;
  scu.translation_warp = header.translation_warp;
  scu.config_id = config_id;
  scu.codec_bundle_sha256 = header.codec_bundle_sha256;
  scu.coded_width = static_cast<uint32_t>(header.width);
  scu.coded_height = static_cast<uint32_t>(header.height);
  if (header.coded_width > 0) scu.coded_width = static_cast<uint32_t>(header.coded_width);
  if (header.coded_height > 0) scu.coded_height = static_cast<uint32_t>(header.coded_height);
  scu.visible_width = static_cast<uint32_t>(header.width);
  scu.visible_height = static_cast<uint32_t>(header.height);
  scu.max_frame_bytes = static_cast<uint32_t>(std::min<uint64_t>(
      MaxUlbvcFramePayloadBytes(header.width, header.height), std::numeric_limits<uint32_t>::max()));
  const uint64_t fps_scaled = static_cast<uint64_t>(std::llround(header.fps * 1000.0));
  const uint32_t fps_num = static_cast<uint32_t>(std::max<uint64_t>(1, fps_scaled));
  const uint32_t fps_den = 1000;
  const uint32_t divisor = std::gcd(fps_num, fps_den);
  scu.nominal_fps_num = fps_num / divisor;
  scu.nominal_fps_den = fps_den / divisor;
  std::vector<uint8_t> policy;
  PutBe32(&policy, static_cast<uint32_t>(header.gop));
  PutBe32(&policy, static_cast<uint32_t>(header.reset_interval));
  PutBe32(&policy, static_cast<uint32_t>(header.ltr_start_idx));
  PutBe32(&policy, static_cast<uint32_t>(header.ltr_period));
  scu.tlvs.push_back({ulbvc::transport::kUlbvcTlvNominalPolicy, std::move(policy)});
  std::vector<uint8_t> forced;
  PutBe32(&forced, static_cast<uint32_t>(header.forced_ltr_recovery_frame));
  PutBe32(&forced, static_cast<uint32_t>(header.forced_ltr_reference_frame));
  scu.tlvs.push_back({ulbvc::transport::kUlbvcTlvVendorData, std::move(forced)});
  return scu;
}

std::vector<uint8_t> ReadMediaUnit(std::ifstream* input,
                                  uint64_t max_efu_payload_bytes =
                                      ulbvc::transport::kUlbvcMediaUnitMaxBytes) {
  constexpr std::size_t kCommonBytes = ulbvc::transport::kUlbvcMediaUnitCommonHeaderBytes;
  for (;;) {
    std::array<uint8_t, kCommonBytes> common{};
    input->read(reinterpret_cast<char*>(common.data()), static_cast<std::streamsize>(common.size()));
    if (input->gcount() == 0 && input->eof()) return {};
    Check(input->gcount() == static_cast<std::streamsize>(common.size()),
          "truncated ULBVC-ES media unit header");
    Check(common[0] == ulbvc::transport::kUlbvcMediaUnitVersion,
          "unsupported ULBVC-ES media unit version");
    const auto read_u16 = [&common](std::size_t offset) {
      return static_cast<uint16_t>((static_cast<uint16_t>(common[offset]) << 8) |
                                   common[offset + 1]);
    };
    const auto read_u32 = [&common](std::size_t offset) {
      return (static_cast<uint32_t>(common[offset]) << 24) |
             (static_cast<uint32_t>(common[offset + 1]) << 16) |
             (static_cast<uint32_t>(common[offset + 2]) << 8) | common[offset + 3];
    };
    const uint8_t raw_type = common[1];
    const uint16_t header_length = read_u16(2);
    const uint32_t unit_length = read_u32(4);
    Check(header_length >= kCommonBytes && (header_length & 3u) == 0 &&
              header_length <= ulbvc::transport::kUlbvcHeaderMaxBytes &&
              header_length <= unit_length && unit_length <= ulbvc::transport::kUlbvcMediaUnitMaxBytes,
          "invalid ULBVC-ES common header length");
    const auto type = static_cast<ulbvc::transport::UlbvcMediaUnitType>(raw_type);
    if (type != ulbvc::transport::UlbvcMediaUnitType::kScu &&
        type != ulbvc::transport::UlbvcMediaUnitType::kEfu &&
        type != ulbvc::transport::UlbvcMediaUnitType::kEos) {
      // Unknown optional/reserved units are length-delimited and skippable.
      // Consume them in the stream without allocating attacker-selected size.
      const auto remaining = static_cast<std::streamsize>(unit_length - kCommonBytes);
      input->ignore(remaining);
      Check(input->gcount() == remaining, "truncated unknown ULBVC-ES media unit");
      continue;
    }

    if (type == ulbvc::transport::UlbvcMediaUnitType::kScu) {
      Check(unit_length <= ulbvc::transport::kUlbvcScuMaxBytes &&
                header_length >= ulbvc::transport::kUlbvcScuFixedBytes &&
                unit_length == header_length,
            "invalid ULBVC-ES SCU length");
      std::vector<uint8_t> unit(common.begin(), common.end());
      unit.resize(unit_length);
      const auto remaining = static_cast<std::streamsize>(unit_length - kCommonBytes);
      input->read(reinterpret_cast<char*>(unit.data() + kCommonBytes), remaining);
      Check(input->gcount() == remaining, "truncated ULBVC-ES SCU");
      return unit;
    }
    if (type == ulbvc::transport::UlbvcMediaUnitType::kEos) {
      Check(header_length == kCommonBytes && unit_length == kCommonBytes,
            "invalid ULBVC-ES EOS length");
      return std::vector<uint8_t>(common.begin(), common.end());
    }

    Check(header_length >= ulbvc::transport::kUlbvcEfuFixedBytes,
          "invalid ULBVC-ES EFU header length");
    std::array<uint8_t, ulbvc::transport::kUlbvcEfuFixedBytes - kCommonBytes> fixed{};
    input->read(reinterpret_cast<char*>(fixed.data()), static_cast<std::streamsize>(fixed.size()));
    Check(input->gcount() == static_cast<std::streamsize>(fixed.size()),
          "truncated ULBVC-ES EFU fixed header");
    const uint32_t payload_length =
        (static_cast<uint32_t>(fixed[20]) << 24) | (static_cast<uint32_t>(fixed[21]) << 16) |
        (static_cast<uint32_t>(fixed[22]) << 8) | fixed[23];
    Check(static_cast<uint64_t>(header_length) + payload_length == unit_length &&
              payload_length <= max_efu_payload_bytes,
          "invalid or oversized ULBVC-ES EFU payload length");
    const std::streamsize remaining = static_cast<std::streamsize>(unit_length - fixed.size() - kCommonBytes);
    const std::streampos payload_position = input->tellg();
    Check(payload_position >= 0, "failed to determine ULBVC-ES payload position");
    input->seekg(0, std::ios::end);
    const std::streampos file_end = input->tellg();
    input->seekg(payload_position);
    Check(input->good() && file_end >= payload_position &&
              file_end - payload_position >= remaining,
          "ULBVC-ES EFU payload exceeds remaining file bytes");

    std::vector<uint8_t> unit(common.begin(), common.end());
    unit.reserve(unit_length);
    unit.insert(unit.end(), fixed.begin(), fixed.end());
    const std::size_t old_size = unit.size();
    unit.resize(unit_length);
    input->read(reinterpret_cast<char*>(unit.data() + old_size), remaining);
    Check(input->gcount() == remaining, "truncated ULBVC-ES EFU");
    return unit;
  }
}

}  // namespace

uint64_t MaxUlbvcFramePayloadBytes(int width, int height) {
  return MaxUlbvcFramePayloadBytesImpl(width, height);
}

void ValidateUlbvcBitstreamHeader(const UlbvcBitstreamHeader& header) {
  ValidateUlbvcBitstreamHeaderImpl(header);
}

void ValidateUlbvcDecoderOutputShape(const UlbvcBitstreamHeader& header,
                                    const std::vector<int64_t>& output_shape) {
  ValidateUlbvcBitstreamHeaderImpl(header);
  Check(output_shape.size() == 4, "ULBVCDecoder x_hat output must be a rank-4 NCHW tensor");
  Check(output_shape[0] == 1 && output_shape[1] == 3,
        "ULBVCDecoder x_hat output must have shape [1, 3, H, W]");
  Check(output_shape[2] > 0 && output_shape[3] > 0,
        "ULBVCDecoder x_hat output dimensions must be positive");
  Check(static_cast<int64_t>(header.height) <= output_shape[2] &&
            static_cast<int64_t>(header.width) <= output_shape[3],
        "ULBVC bitstream resolution " + std::to_string(header.width) + "x" +
            std::to_string(header.height) + " exceeds ULBVCDecoder x_hat shape " +
            std::to_string(output_shape[3]) + "x" + std::to_string(output_shape[2]));
  if (header.coded_width > 0 || header.coded_height > 0) {
    Check(header.coded_width > 0 && header.coded_height > 0 &&
              static_cast<int64_t>(header.coded_width) == output_shape[3] &&
              static_cast<int64_t>(header.coded_height) == output_shape[2],
          "ULBVC coded dimensions do not match the decoder model output shape");
  }
}

void ValidateUlbvcQIndexForSidecar(int q_index, int supported_q_index_count) {
  Check(supported_q_index_count > 0, "sidecar supported Q index count must be positive");
  Check(q_index >= 0 && q_index < supported_q_index_count,
        "ULBVC Q index must be in the sidecar-supported range [0, " +
            std::to_string(supported_q_index_count - 1) + "], got " + std::to_string(q_index));
}

ForcedLtrFrames ResolveForcedLtrFrames(const UlbvcBitstreamHeader& header,
                                       int requested_recovery_frame,
                                       int requested_reference_frame) {
  Check(requested_recovery_frame >= -1 && requested_reference_frame >= -1,
        "decoder forced LTR frame indices must be -1 or non-negative");
  Check((requested_recovery_frame >= 0) == (requested_reference_frame >= 0),
        "decoder forced LTR recovery and reference frames must be configured together");

  ForcedLtrFrames result{requested_recovery_frame, requested_reference_frame};
  if (header.version >= 4) {
    if (requested_recovery_frame >= 0) {
      Check(requested_recovery_frame == header.forced_ltr_recovery_frame,
            "decoder forced_ltr_recovery_frame does not match bitstream header");
      Check(requested_reference_frame == header.forced_ltr_reference_frame,
            "decoder forced_ltr_reference_frame does not match bitstream header");
    }
    result = ForcedLtrFrames{header.forced_ltr_recovery_frame,
                             header.forced_ltr_reference_frame};
  }
  ulbvc::codec::ValidateForcedLtrConfiguration(header.gop, header.ltr_start_idx,
                                              header.ltr_period, result.recovery_frame,
                                              result.reference_frame);
  return result;
}

void ValidateUlbvcFrameMetadata(int frame_index, ulbvc::codec::UlbvcFrameType frame_type, int q_index,
                               int expected_frame_index, bool require_i_frame) {
  ValidateUlbvcFrameMetadataImpl(frame_index, frame_type, q_index, expected_frame_index,
                                require_i_frame);
}

OfficialUlbvcBitstreamWriter::OfficialUlbvcBitstreamWriter(const std::filesystem::path& path)
    : path_(path) {
  if (!path_.parent_path().empty()) {
    std::filesystem::create_directories(path_.parent_path());
  }
  output_.open(path_, std::ios::binary | std::ios::trunc);
  Check(output_.good(), "failed to open official ULBVC bitstream output: " + path_.string());
}

OfficialUlbvcBitstreamWriter::~OfficialUlbvcBitstreamWriter() { Close(); }

void OfficialUlbvcBitstreamWriter::WriteFrame(int q_index, const std::vector<uint8_t>& payload) {
  Check(!closed_, "write on closed official ULBVC bitstream");
  Check(q_index >= kMinQp && q_index <= kMaxQp,
        "official ULBVC q_index must be in [" + std::to_string(kMinQp) + ", " +
            std::to_string(kMaxQp) + "]");
  Check(payload.size() <= kMaxPayloadSize4K,
        "official ULBVC frame payload exceeds the maximum supported size");
  Check(payload.size() <= static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()),
        "official ULBVC frame payload is too large");
  WritePod(output_, static_cast<int32_t>(q_index), "q_index");
  WritePod(output_, static_cast<uint32_t>(payload.size()), "payload_size");
  output_.write(reinterpret_cast<const char*>(payload.data()),
                static_cast<std::streamsize>(payload.size()));
  Check(output_.good(), "failed to write official ULBVC payload");
}

void OfficialUlbvcBitstreamWriter::Close() {
  if (closed_) {
    return;
  }
  if (output_.is_open()) {
    output_.flush();
    Check(output_.good(), "failed to flush official ULBVC bitstream: " + path_.string());
    output_.close();
  }
  closed_ = true;
}

OfficialUlbvcBitstreamReader::OfficialUlbvcBitstreamReader(const std::filesystem::path& path)
    : path_(path) {
  input_.open(path_, std::ios::binary);
  Check(input_.good(), "failed to open official ULBVC bitstream input: " + path_.string());
}

bool OfficialUlbvcBitstreamReader::ReadFrame(int* frame_index, int* q_index,
                                            std::vector<uint8_t>* payload) {
  Check(frame_index != nullptr && q_index != nullptr && payload != nullptr,
        "official ULBVC frame outputs are required");
  if (input_.peek() == std::char_traits<char>::eof()) {
    return false;
  }
  const int read_frame_index = next_frame_index_;
  *q_index = static_cast<int>(ReadPod<int32_t>(input_, "q_index"));
  Check(*q_index >= kMinQp && *q_index <= kMaxQp,
        "official ULBVC q_index must be in [" + std::to_string(kMinQp) + ", " +
            std::to_string(kMaxQp) + "]");

  const uint32_t bytes = ReadPod<uint32_t>(input_, "payload_size");
  Check(bytes <= kMaxPayloadSize4K,
        "official ULBVC frame payload size exceeds maximum: " + std::to_string(bytes) + " > " +
            std::to_string(kMaxPayloadSize4K));

  const std::streampos current_pos = input_.tellg();
  input_.seekg(0, std::ios::end);
  const std::streampos file_size = input_.tellg();
  input_.seekg(current_pos);
  Check(current_pos >= 0 && file_size >= current_pos,
        "failed to determine official ULBVC file size");
  Check(current_pos + static_cast<std::streamoff>(bytes) <= file_size,
        "official ULBVC frame payload size exceeds remaining file size: " + std::to_string(bytes) +
            " > " + std::to_string(file_size - current_pos));

  payload->resize(bytes);
  input_.read(reinterpret_cast<char*>(payload->data()), static_cast<std::streamsize>(bytes));
  Check(input_.good(), "failed to read official ULBVC payload");
  *frame_index = read_frame_index;
  ++next_frame_index_;
  return true;
}

UlbvcBitstreamWriter::UlbvcBitstreamWriter(const std::filesystem::path& path,
                                         const UlbvcBitstreamHeader& header)
    : path_(path) {
  ValidateUlbvcBitstreamHeader(header);
  max_payload_size_ = MaxUlbvcFramePayloadBytes(header.width, header.height);
  fps_ = header.fps;

  if (!path_.parent_path().empty()) {
    std::filesystem::create_directories(path_.parent_path());
  }
  output_.open(path_, std::ios::binary | std::ios::trunc);
  Check(output_.good(), "failed to open ulbvc bitstream output: " + path_.string());
  output_.write(kEsMagic.data(), static_cast<std::streamsize>(kEsMagic.size()));
  PutBe16(output_, 1);
  PutBe16(output_, 0);
  PutBe16(output_, 16);
  PutBe16(output_, 0);
  const auto scu = MakeScu(header, config_id_);
  const auto scu_bytes = ulbvc::transport::SerializeScu(scu);
  active_config_unit_ = scu_bytes;
  output_.write(reinterpret_cast<const char*>(scu_bytes.data()),
                static_cast<std::streamsize>(scu_bytes.size()));
  Check(output_.good(), "failed to write ULBVC-ES preamble or SCU");
}

UlbvcBitstreamWriter::~UlbvcBitstreamWriter() { Close(); }

void UlbvcBitstreamWriter::WriteFrame(int frame_index, ulbvc::codec::UlbvcFrameType frame_type,
                                     int q_index, const std::vector<uint8_t>& payload) {
  UlbvcFrameMetadata metadata;
  metadata.explicit_metadata = false;
  WriteFrame(frame_index, frame_type, q_index, metadata, payload);
}

void UlbvcBitstreamWriter::WriteFrame(int frame_index, ulbvc::codec::UlbvcFrameType frame_type,
                                     int q_index, const UlbvcFrameMetadata& metadata,
                                     const std::vector<uint8_t>& payload) {
  Check(!closed_, "write on closed ulbvc bitstream");
  if (configuration_switch_pending_) {
    Check(frame_type == ulbvc::codec::UlbvcFrameType::kIFrame,
          "ULBVC-ES configuration switch must begin with an I-frame");
    if (metadata.explicit_metadata) {
      Check((metadata.unit_flags & (ulbvc::transport::kEfuRandomAccess |
                                    ulbvc::transport::kEfuResetReference)) ==
                (ulbvc::transport::kEfuRandomAccess | ulbvc::transport::kEfuResetReference),
            "ULBVC-ES configuration switch I-frame is not random access");
    }
  }
  ValidateUlbvcFrameMetadata(frame_index, frame_type, q_index, next_frame_index_,
                            next_frame_index_ == 0);
  Check(metadata.explicit_metadata || frame_type == ulbvc::codec::UlbvcFrameType::kIFrame,
        "ULBVC-ES inter frames require explicit reference metadata");
  Check(payload.size() <= max_payload_size_,
        "ULBVC frame payload exceeds maximum for configured resolution");
  const bool warp = ulbvc::transport::ParseScu(active_config_unit_).translation_warp;
  Check(metadata.translation_warp == warp,
        "ULBVC frame warp mode disagrees with stream capability");
  Check((metadata.unit_flags & ulbvc::transport::kEfuTranslationWarp) == 0 || warp,
        "unexpected ULBVC frame warp flag");
  ulbvc::transport::UlbvcEfu efu;
  efu.config_id = config_id_;
  efu.frame_id = static_cast<uint32_t>(frame_index);
  efu.frame_type = static_cast<uint8_t>(frame_type);
  efu.entropy_q_index = static_cast<uint8_t>(q_index);
  efu.model_q_index = static_cast<uint8_t>(q_index);
  efu.pts = metadata.explicit_metadata
                ? metadata.pts
                : static_cast<int64_t>(std::llround(
                      static_cast<long double>(frame_index) * 90000.0L /
                      static_cast<long double>(fps_)));
  if (metadata.explicit_metadata) {
    efu.unit_flags = metadata.unit_flags | ulbvc::transport::kEfuCrcPresent;
    efu.short_ref_frame_id = metadata.short_ref_frame_id;
    efu.long_ref_frame_id = metadata.long_ref_frame_id;
    const int model_q_index = metadata.model_q_index < 0 ? q_index : metadata.model_q_index;
    Check(model_q_index >= 0 && model_q_index < 64,
          "ULBVC explicit model Q index must be in [0, 63]");
    efu.model_q_index = static_cast<uint8_t>(model_q_index);
  } else if (frame_type == ulbvc::codec::UlbvcFrameType::kIFrame) {
    efu.unit_flags |= ulbvc::transport::kEfuRandomAccess | ulbvc::transport::kEfuResetReference;
  }
  if (warp) efu.unit_flags |= ulbvc::transport::kEfuTranslationWarp;
  efu.kx = metadata.kx;
  efu.ky = metadata.ky;
  efu.tlvs = metadata.tlvs;
  efu.entropy_payload = payload;
  const auto unit = ulbvc::transport::SerializeEfu(efu);
  output_.write(reinterpret_cast<const char*>(unit.data()), static_cast<std::streamsize>(unit.size()));
  Check(output_.good(), "failed to write ulbvc bitstream payload");
  configuration_switch_pending_ = false;
  ++next_frame_index_;
}

void UlbvcBitstreamWriter::SwitchConfiguration(uint32_t config_id,
                                              const UlbvcBitstreamHeader& header) {
  Check(!closed_, "configuration switch on closed ulbvc bitstream");
  Check(ulbvc::transport::CompareUlbvcSerial32(config_id, config_id_) ==
                ulbvc::transport::UlbvcSerial32Order::kNewer &&
            config_id != 0,
        "ULBVC-ES configuration IDs must increase monotonically");
  UlbvcBitstreamHeader normalized = header;
  if (normalized.version == 0) normalized.version = 4;
  ValidateUlbvcBitstreamHeader(normalized);
  const auto scu = MakeScu(normalized, config_id);
  const auto bytes = ulbvc::transport::SerializeScu(scu);
  Check(ulbvc::transport::UlbvcScuDecoderCompatible(
            ulbvc::transport::ParseScu(active_config_unit_),
            ulbvc::transport::ParseScu(bytes)),
        "ULBVC-ES configuration switch changes decoder compatibility fields");
  output_.write(reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
  Check(output_.good(), "failed to write ULBVC-ES configuration switch");
  active_config_unit_ = bytes;
  config_id_ = config_id;
  fps_ = normalized.fps;
  max_payload_size_ = MaxUlbvcFramePayloadBytes(normalized.width, normalized.height);
  configuration_switch_pending_ = true;
}

void UlbvcBitstreamWriter::Close() {
  if (closed_) {
    return;
  }
  if (output_.is_open()) {
    output_.flush();
    Check(output_.good(), "failed to flush ulbvc bitstream output: " + path_.string());
    output_.close();
  }
  closed_ = true;
}

UlbvcBitstreamReader::UlbvcBitstreamReader(const std::filesystem::path& path) : path_(path) {
  input_.open(path_, std::ios::binary);
  Check(input_.good(), "failed to open ulbvc bitstream input: " + path_.string());

  std::array<char, 8> magic{};
  input_.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  Check(input_.good(), "failed to read ulbvc bitstream magic: " + path_.string());
  if (magic == kEsMagic) {
    std::array<uint8_t, 8> preamble{};
    input_.read(reinterpret_cast<char*>(preamble.data()), static_cast<std::streamsize>(preamble.size()));
    Check(input_.good(), "truncated ULBVC-ES preamble: " + path_.string());
    const uint16_t major = static_cast<uint16_t>(preamble[0] << 8 | preamble[1]);
    const uint16_t minor = static_cast<uint16_t>(preamble[2] << 8 | preamble[3]);
    const uint16_t preamble_length = static_cast<uint16_t>(preamble[4] << 8 | preamble[5]);
    const uint16_t flags = static_cast<uint16_t>(preamble[6] << 8 | preamble[7]);
    Check(major == 1 && minor == 0 && preamble_length == 16 && flags == 0,
          "unsupported ULBVC-ES preamble");
    const auto scu_unit = ReadMediaUnit(&input_);
    Check(!scu_unit.empty(), "ULBVC-ES file is missing its SCU");
    const auto scu = ulbvc::transport::ParseScu(scu_unit);
    active_config_unit_ = scu_unit;
    media_unit_format_ = true;
    version_ = 4;
    config_id_ = scu.config_id;
    header_.version = version_;
    header_.translation_warp = scu.translation_warp;
    header_.width = static_cast<int>(scu.visible_width);
    header_.height = static_cast<int>(scu.visible_height);
    header_.coded_width = static_cast<int>(scu.coded_width);
    header_.coded_height = static_cast<int>(scu.coded_height);
    header_.codec_bundle_sha256 = scu.codec_bundle_sha256;
    header_.fps = static_cast<double>(scu.nominal_fps_num) /
                  static_cast<double>(scu.nominal_fps_den);
    header_.q_index = std::min<int>(21, scu.q_index_count - 1);
    for (const auto& tlv : scu.tlvs) {
      if (tlv.type == ulbvc::transport::kUlbvcTlvNominalPolicy && tlv.value.size() == 16) {
        header_.gop = ReadBe32ApplicationValue(tlv.value, 0, "gop");
        header_.reset_interval = ReadBe32ApplicationValue(tlv.value, 4, "reset_interval");
        header_.ltr_start_idx = ReadBe32ApplicationValue(tlv.value, 8, "ltr_start_idx");
        header_.ltr_period = ReadBe32ApplicationValue(tlv.value, 12, "ltr_period");
      }
      if (tlv.type == ulbvc::transport::kUlbvcTlvVendorData && tlv.value.size() == 8) {
        header_.forced_ltr_recovery_frame =
            ReadBe32ApplicationValue(tlv.value, 0, "forced_ltr_recovery_frame", true);
        header_.forced_ltr_reference_frame =
            ReadBe32ApplicationValue(tlv.value, 4, "forced_ltr_reference_frame", true);
      }
    }
    max_payload_size_ = std::min<uint64_t>(scu.max_frame_bytes,
                                           MaxUlbvcFramePayloadBytes(header_.width, header_.height));
    ValidateUlbvcBitstreamHeader(header_);
    return;
  }
  Check(magic == kMagic, "invalid ulbvc bitstream magic: " + path_.string());

  const uint32_t version = ReadPod<uint32_t>(input_, "version");
  Check(version >= 2 && version <= kVersion, "unsupported ulbvc bitstream version");
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

  ValidateUlbvcBitstreamHeader(header_);
  max_payload_size_ = MaxUlbvcFramePayloadBytes(header_.width, header_.height);
}

UlbvcBitstreamReader::~UlbvcBitstreamReader() = default;

bool UlbvcBitstreamReader::ReadFrame(int* frame_index, ulbvc::codec::UlbvcFrameType* frame_type,
                                    int* q_index, std::vector<uint8_t>* payload) {
  Check(frame_index != nullptr, "ulbvc bitstream frame index output is required");
  Check(frame_type != nullptr && q_index != nullptr && payload != nullptr,
        "ulbvc bitstream frame outputs are required");

  if (input_.peek() == std::char_traits<char>::eof()) {
    return false;
  }

  if (media_unit_format_) {
    std::vector<uint8_t> unit;
    ulbvc::transport::UlbvcCommonHeader common;
    ulbvc::transport::UlbvcEfu efu;
    for (;;) {
      unit = ReadMediaUnit(&input_, max_payload_size_ +
                          (header_.translation_warp ? 2u : 0u));
      if (unit.empty()) return false;
      common = ulbvc::transport::ParseMediaUnitHeader(unit);
      if (common.unit_type == ulbvc::transport::UlbvcMediaUnitType::kEos) {
        ulbvc::transport::ValidateMediaUnit(unit);
        Check(common.config_id == config_id_, "ULBVC-ES EOS references an unknown SCU");
        if (common.unit_id != ulbvc::transport::kUlbvcNoReference) {
          Check(common.unit_id == static_cast<uint32_t>(expected_frame_index_),
                "ULBVC-ES EOS frame boundary does not match the number of frames read");
        }
        Check(input_.peek() == std::char_traits<char>::eof(),
              "ULBVC-ES contains data after EOS");
        return false;
      }
      if (common.unit_type == ulbvc::transport::UlbvcMediaUnitType::kScu) {
        const auto scu = ulbvc::transport::ParseScu(unit);
        const auto config_order =
            ulbvc::transport::CompareUlbvcSerial32(scu.config_id, config_id_);
        Check(config_order != ulbvc::transport::UlbvcSerial32Order::kAmbiguous,
              "ULBVC-ES configuration ID is serial-number ambiguous");
        Check(config_order != ulbvc::transport::UlbvcSerial32Order::kOlder,
              "ULBVC-ES configuration ID moved backwards");
        if (config_order == ulbvc::transport::UlbvcSerial32Order::kSame) {
          Check(unit == active_config_unit_,
                "ULBVC-ES configuration ID was reused with different SCU bytes");
          continue;
        }
        if (config_order == ulbvc::transport::UlbvcSerial32Order::kNewer) {
          Check(ulbvc::transport::UlbvcScuDecoderCompatible(
                    ulbvc::transport::ParseScu(active_config_unit_), scu),
                "ULBVC-ES configuration switch changes decoder compatibility fields");
          config_id_ = scu.config_id;
          active_config_unit_ = unit;
          max_payload_size_ = std::min<uint64_t>(scu.max_frame_bytes,
                                                 MaxUlbvcFramePayloadBytes(header_.width, header_.height));
          pending_config_id_ = config_id_;
          last_ltr_frame_id_ = ulbvc::transport::kUlbvcNoReference;
        }
        continue;
      }
      if (common.unit_type != ulbvc::transport::UlbvcMediaUnitType::kEfu) continue;
      efu = ulbvc::transport::ParseEfu(unit);
      break;
    }
    Check(efu.config_id == config_id_, "ULBVC-ES EFU references an unknown SCU");
    ulbvc::transport::ValidateEfuTranslationWarp(efu, header_.translation_warp);
    if (pending_config_id_ != 0) {
      Check(efu.config_id == pending_config_id_ && efu.frame_type == 0 &&
                (efu.unit_flags & (ulbvc::transport::kEfuRandomAccess |
                                   ulbvc::transport::kEfuResetReference)) ==
                    (ulbvc::transport::kEfuRandomAccess | ulbvc::transport::kEfuResetReference),
            "ULBVC-ES configuration switch was not followed by a random-access EFU");
      pending_config_id_ = 0;
    }
    Check(efu.frame_id <= static_cast<uint32_t>(std::numeric_limits<int>::max()),
          "ULBVC-ES frame ID exceeds the application range");
    const auto read_frame_type = static_cast<ulbvc::codec::UlbvcFrameType>(efu.frame_type);
    const int read_frame_index = static_cast<int>(efu.frame_id);
    const int read_q_index = static_cast<int>(efu.entropy_q_index);
    ValidateUlbvcFrameMetadata(read_frame_index, read_frame_type, read_q_index,
                              expected_frame_index_, expected_frame_index_ == 0);
    Check(efu.entropy_payload.size() <= max_payload_size_,
          "ULBVC-ES frame payload exceeds maximum for resolution");
    *frame_index = read_frame_index;
    *frame_type = read_frame_type;
    *q_index = read_q_index;
    *payload = efu.entropy_payload;
    last_frame_metadata_.explicit_metadata = true;
    last_frame_metadata_.translation_warp = header_.translation_warp;
    last_frame_metadata_.kx = efu.kx;
    last_frame_metadata_.ky = efu.ky;
    last_frame_metadata_.model_q_index = efu.model_q_index;
    last_frame_metadata_.unit_flags = efu.unit_flags;
    last_frame_metadata_.short_ref_frame_id = efu.short_ref_frame_id;
    last_frame_metadata_.long_ref_frame_id = efu.long_ref_frame_id;
    last_frame_metadata_.pts = efu.pts;
    last_frame_metadata_.tlvs = efu.tlvs;
    Check(expected_frame_index_ < std::numeric_limits<int>::max(),
          "ULBVC-ES frame index reached the application limit");
    ++expected_frame_index_;
    return true;
  }

  const int read_frame_index = static_cast<int>(ReadPod<int32_t>(input_, "frame_index"));
  const uint8_t type_byte = ReadPod<uint8_t>(input_, "frame_type");
  ulbvc::codec::UlbvcFrameType read_frame_type;
  if (version_ >= 3) {
    Check(type_byte <= static_cast<uint8_t>(ulbvc::codec::UlbvcFrameType::kLtrRecovery),
          "invalid ulbvc frame type");
    read_frame_type = static_cast<ulbvc::codec::UlbvcFrameType>(type_byte);
  } else {
    read_frame_type =
        type_byte != 0 ? ulbvc::codec::UlbvcFrameType::kIFrame : ulbvc::codec::UlbvcFrameType::kPFrame;
  }

  const int read_q_index = static_cast<int>(ReadPod<int32_t>(input_, "q_index"));
  ValidateUlbvcFrameMetadata(read_frame_index, read_frame_type, read_q_index, expected_frame_index_,
                            expected_frame_index_ == 0);

  const uint64_t bytes = ReadPod<uint64_t>(input_, "payload_size");
  Check(bytes <= max_payload_size_,
        "ULBVC frame payload size exceeds maximum for resolution " + std::to_string(header_.width) +
            "x" + std::to_string(header_.height) + ": " + std::to_string(bytes) + " > " +
            std::to_string(max_payload_size_));

  const std::streampos current_pos = input_.tellg();
  input_.seekg(0, std::ios::end);
  const std::streampos file_size = input_.tellg();
  input_.seekg(current_pos);
  Check(current_pos >= 0 && file_size >= current_pos, "failed to determine ULBVC file size");
  Check(static_cast<uint64_t>(file_size - current_pos) >= bytes,
        "ULBVC frame payload size exceeds remaining file size: " + std::to_string(bytes) + " > " +
            std::to_string(file_size - current_pos));

  payload->resize(static_cast<std::size_t>(bytes));
  input_.read(reinterpret_cast<char*>(payload->data()), static_cast<std::streamsize>(bytes));
  Check(input_.good(), "failed to read ulbvc bitstream payload");

  *frame_index = read_frame_index;
  *frame_type = read_frame_type;
  *q_index = read_q_index;
  Check(expected_frame_index_ < std::numeric_limits<int>::max(),
        "ULBVC frame index reached the application limit");
  ++expected_frame_index_;

  last_frame_metadata_ = UlbvcFrameMetadata{};

  return true;
}

}  // namespace ulbvc::io
