#ifndef MLVC_TRANSPORT_MLVC_MEDIA_UNIT_H_
#define MLVC_TRANSPORT_MLVC_MEDIA_UNIT_H_

#include <array>
#include <cstdint>
#include <vector>

namespace mlvc::transport {

constexpr uint8_t kMlvcMediaUnitVersion = 1;
constexpr std::size_t kMlvcMediaUnitCommonHeaderBytes = 20;
constexpr std::size_t kMlvcScuFixedBytes = 80;
constexpr std::size_t kMlvcEfuFixedBytes = 52;
constexpr std::size_t kMlvcMediaUnitMaxBytes = 128u * 1024u * 1024u;
constexpr std::size_t kMlvcScuMaxBytes = 64u * 1024u;
constexpr std::size_t kMlvcHeaderMaxBytes = 64u * 1024u;
constexpr uint32_t kMlvcNoReference = 0xffffffffu;

enum class MlvcSerial32Order { kSame, kNewer, kOlder, kAmbiguous };

inline MlvcSerial32Order CompareMlvcSerial32(uint32_t candidate, uint32_t current) {
  const uint32_t distance = candidate - current;
  if (distance == 0) return MlvcSerial32Order::kSame;
  if (distance == 0x80000000u) return MlvcSerial32Order::kAmbiguous;
  return distance < 0x80000000u ? MlvcSerial32Order::kNewer : MlvcSerial32Order::kOlder;
}

enum class MlvcMediaUnitType : uint8_t {
  kScu = 0x01,
  kEfu = 0x02,
  kEos = 0x03,
  // Unassigned values are skippable once their common header and length pass
  // validation; this version does not assign them decoding semantics.
  kUnknown = 0xff,
};

enum MlvcScuFlags : uint32_t {
  kScuRequiresRandomAccess = 1u << 0,
  kScuStaticTensorShape = 1u << 1,
  kScuFullRange = 1u << 2,
};

enum MlvcEfuFlags : uint32_t {
  kEfuRandomAccess = 1u << 0,
  kEfuResetReference = 1u << 1,
  kEfuStoreAsLtr = 1u << 2,
  kEfuDiscardable = 1u << 3,
  kEfuDiscontinuity = 1u << 4,
  kEfuCrcPresent = 1u << 5,
};

constexpr uint16_t kMlvcTlvCodecBundleSha256 = 0x8001;
constexpr uint16_t kMlvcTlvProfileName = 0x0002;
constexpr uint16_t kMlvcTlvColorimetry = 0x0003;
constexpr uint16_t kMlvcTlvNominalPolicy = 0x0004;
constexpr uint16_t kMlvcTlvVendorData = 0x0005;

struct MlvcTlv {
  uint16_t type = 0;
  std::vector<uint8_t> value;
};

struct MlvcCommonHeader {
  MlvcMediaUnitType unit_type = MlvcMediaUnitType::kEos;
  uint16_t header_length = 0;
  uint32_t unit_length = 0;
  uint32_t unit_id = 0;
  uint32_t config_id = 0;
  uint32_t unit_flags = 0;
};

struct MlvcScu {
  uint32_t config_id = 1;
  uint16_t profile_id = 1;
  uint16_t level_id = 0;
  uint32_t coded_width = 0;
  uint32_t coded_height = 0;
  uint32_t visible_width = 0;
  uint32_t visible_height = 0;
  uint32_t timebase_num = 1;
  uint32_t timebase_den = 90000;
  uint32_t nominal_fps_num = 30;
  uint32_t nominal_fps_den = 1;
  uint8_t color_format = 1;
  uint8_t bit_depth = 16;
  uint8_t entropy_mode = 1;
  uint8_t q_index_count = 64;
  uint32_t max_frame_bytes = 64u * 1024u * 1024u;
  std::array<uint8_t, 16> config_hash{};
  std::array<uint8_t, 32> codec_bundle_sha256{};
  uint32_t unit_flags = kScuRequiresRandomAccess | kScuStaticTensorShape;
  std::vector<MlvcTlv> tlvs;
};

struct MlvcEfu {
  uint32_t config_id = 1;
  uint32_t frame_id = 0;
  uint32_t unit_flags = kEfuCrcPresent;
  uint8_t frame_type = 0;
  uint8_t entropy_q_index = 0;
  uint8_t model_q_index = 0;
  uint8_t temporal_id = 0;
  int64_t pts = 0;
  uint32_t short_ref_frame_id = kMlvcNoReference;
  uint32_t long_ref_frame_id = kMlvcNoReference;
  std::vector<MlvcTlv> tlvs;
  std::vector<uint8_t> entropy_payload;
};

struct MlvcMediaUnit {
  MlvcCommonHeader header;
  std::vector<uint8_t> bytes;
};

// Serializes and validates the network-order MLVC-ES media unit format.
std::vector<uint8_t> SerializeScu(const MlvcScu& scu);
MlvcScu ParseScu(const std::vector<uint8_t>& unit);
bool MlvcScuDecoderCompatible(const MlvcScu& current, const MlvcScu& next);
std::vector<uint8_t> SerializeEfu(const MlvcEfu& efu);
MlvcEfu ParseEfu(const std::vector<uint8_t>& unit);
std::vector<uint8_t> SerializeEos(uint32_t config_id);
// RTP EOS carries the exclusive frame boundary so a receiver can wait for
// completed units that were reordered behind the end marker.
std::vector<uint8_t> SerializeEos(uint32_t config_id, uint32_t next_frame_id);
MlvcCommonHeader ParseMediaUnitHeader(const std::vector<uint8_t>& unit);
void ValidateMediaUnit(const std::vector<uint8_t>& unit);

// CRC-32C (Castagnoli), covering only the EFU entropy payload.
uint32_t MlvcCrc32c(const std::vector<uint8_t>& bytes);

}  // namespace mlvc::transport

#endif  // MLVC_TRANSPORT_MLVC_MEDIA_UNIT_H_
