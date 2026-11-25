#ifndef ULBVC_TRANSPORT_ULBVC_MEDIA_UNIT_H_
#define ULBVC_TRANSPORT_ULBVC_MEDIA_UNIT_H_

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ulbvc::transport {

class UlbvcEfuCrcError : public std::runtime_error {
 public:
  UlbvcEfuCrcError() : std::runtime_error("ULBVC EFU CRC-32C mismatch") {}
};

constexpr uint8_t kUlbvcMediaUnitVersion = 1;
constexpr std::size_t kUlbvcMediaUnitCommonHeaderBytes = 20;
constexpr std::size_t kUlbvcScuFixedBytes = 80;
constexpr std::size_t kUlbvcEfuFixedBytes = 52;
constexpr std::size_t kUlbvcMediaUnitMaxBytes = 128u * 1024u * 1024u;
constexpr std::size_t kUlbvcScuMaxBytes = 64u * 1024u;
constexpr std::size_t kUlbvcHeaderMaxBytes = 64u * 1024u;
constexpr uint32_t kUlbvcNoReference = 0xffffffffu;

enum class UlbvcSerial32Order { kSame, kNewer, kOlder, kAmbiguous };

inline UlbvcSerial32Order CompareUlbvcSerial32(uint32_t candidate, uint32_t current) {
  const uint32_t distance = candidate - current;
  if (distance == 0) return UlbvcSerial32Order::kSame;
  if (distance == 0x80000000u) return UlbvcSerial32Order::kAmbiguous;
  return distance < 0x80000000u ? UlbvcSerial32Order::kNewer : UlbvcSerial32Order::kOlder;
}

enum class UlbvcMediaUnitType : uint8_t {
  kScu = 0x01,
  kEfu = 0x02,
  kEos = 0x03,
  // Unassigned values are skippable once their common header and length pass
  // validation; this version does not assign them decoding semantics.
  kUnknown = 0xff,
};

enum UlbvcScuFlags : uint32_t {
  kScuRequiresRandomAccess = 1u << 0,
  kScuStaticTensorShape = 1u << 1,
  kScuFullRange = 1u << 2,
};

enum UlbvcEfuFlags : uint32_t {
  kEfuRandomAccess = 1u << 0,
  kEfuResetReference = 1u << 1,
  kEfuStoreAsLtr = 1u << 2,
  kEfuDiscardable = 1u << 3,
  kEfuDiscontinuity = 1u << 4,
  kEfuCrcPresent = 1u << 5,
  kEfuTranslationWarp = 1u << 6,
};

constexpr uint16_t kUlbvcTlvCodecBundleSha256 = 0x8001;
constexpr uint16_t kUlbvcTlvProfileName = 0x0002;
constexpr uint16_t kUlbvcTlvColorimetry = 0x0003;
constexpr uint16_t kUlbvcTlvNominalPolicy = 0x0004;
constexpr uint16_t kUlbvcTlvVendorData = 0x0005;
// Critical capability; value {1} selects integer translation warp v1.
constexpr uint16_t kUlbvcTlvTranslationWarp = 0x8006;
constexpr std::size_t kUlbvcTranslationWarpScuOverheadBytes = 8;
constexpr std::size_t kUlbvcTranslationWarpPFrameOverheadBytes = 2;

struct UlbvcTlv {
  uint16_t type = 0;
  std::vector<uint8_t> value;
};

struct UlbvcCommonHeader {
  UlbvcMediaUnitType unit_type = UlbvcMediaUnitType::kEos;
  uint16_t header_length = 0;
  uint32_t unit_length = 0;
  uint32_t unit_id = 0;
  uint32_t config_id = 0;
  uint32_t unit_flags = 0;
};

struct UlbvcScu {
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
  std::vector<UlbvcTlv> tlvs;
  bool translation_warp = false;
};

struct UlbvcEfu {
  uint32_t config_id = 1;
  uint32_t frame_id = 0;
  uint32_t unit_flags = kEfuCrcPresent;
  uint8_t frame_type = 0;
  uint8_t entropy_q_index = 0;
  uint8_t model_q_index = 0;
  uint8_t temporal_id = 0;
  int64_t pts = 0;
  uint32_t short_ref_frame_id = kUlbvcNoReference;
  uint32_t long_ref_frame_id = kUlbvcNoReference;
  std::vector<UlbvcTlv> tlvs;
  std::vector<uint8_t> entropy_payload;
  int8_t kx = 0;
  int8_t ky = 0;
};

struct UlbvcMediaUnit {
  UlbvcCommonHeader header;
  std::vector<uint8_t> bytes;
};

// Serializes and validates the network-order ULBVC-ES media unit format.
std::vector<uint8_t> SerializeScu(const UlbvcScu& scu);
UlbvcScu ParseScu(const std::vector<uint8_t>& unit);
bool UlbvcScuDecoderCompatible(const UlbvcScu& current, const UlbvcScu& next);
std::vector<uint8_t> SerializeEfu(const UlbvcEfu& efu);
UlbvcEfu ParseEfu(const std::vector<uint8_t>& unit);
// Rejects a missing or unexpected warp flag against the active SCU capability.
void ValidateEfuTranslationWarp(const UlbvcEfu& efu, bool translation_warp);
std::vector<uint8_t> SerializeEos(uint32_t config_id);
// RTP EOS carries the exclusive frame boundary so a receiver can wait for
// completed units that were reordered behind the end marker.
std::vector<uint8_t> SerializeEos(uint32_t config_id, uint32_t next_frame_id);
UlbvcCommonHeader ParseMediaUnitHeader(const std::vector<uint8_t>& unit);
void ValidateMediaUnit(const std::vector<uint8_t>& unit);

// CRC-32C (Castagnoli), covering the EFU wire payload, including P geometry.
uint32_t UlbvcCrc32c(const std::vector<uint8_t>& bytes);

}  // namespace ulbvc::transport

#endif  // ULBVC_TRANSPORT_ULBVC_MEDIA_UNIT_H_
