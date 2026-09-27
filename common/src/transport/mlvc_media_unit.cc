#include "mlvc/transport/mlvc_media_unit.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>

namespace mlvc::transport {
namespace {

[[maybe_unused]] std::array<uint8_t, 32> Sha256(const std::vector<uint8_t>& input) {
  constexpr std::array<uint32_t, 64> k = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
      0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
      0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
      0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
      0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
      0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
      0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
      0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
      0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
      0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
  auto rotr = [](uint32_t value, unsigned count) {
    return (value >> count) | (value << (32u - count));
  };
  std::array<uint32_t, 8> state = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  std::vector<uint8_t> padded = input;
  padded.push_back(0x80);
  while ((padded.size() % 64) != 56) padded.push_back(0);
  const uint64_t bit_length = static_cast<uint64_t>(input.size()) * 8;
  for (int shift = 56; shift >= 0; shift -= 8) padded.push_back(static_cast<uint8_t>(bit_length >> shift));
  for (std::size_t block = 0; block < padded.size(); block += 64) {
    std::array<uint32_t, 64> words{};
    for (std::size_t i = 0; i < 16; ++i) {
      const std::size_t at = block + i * 4;
      words[i] = (static_cast<uint32_t>(padded[at]) << 24) |
                 (static_cast<uint32_t>(padded[at + 1]) << 16) |
                 (static_cast<uint32_t>(padded[at + 2]) << 8) | padded[at + 3];
    }
    for (std::size_t i = 16; i < words.size(); ++i) {
      const uint32_t s0 = rotr(words[i - 15], 7) ^ rotr(words[i - 15], 18) ^ (words[i - 15] >> 3);
      const uint32_t s1 = rotr(words[i - 2], 17) ^ rotr(words[i - 2], 19) ^ (words[i - 2] >> 10);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (std::size_t i = 0; i < words.size(); ++i) {
      const uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) +
                          ((e & f) ^ (~e & g)) + k[i] + words[i];
      const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) +
                          ((a & b) ^ (a & c) ^ (b & c));
      h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
  }
  std::array<uint8_t, 32> digest{};
  for (std::size_t i = 0; i < state.size(); ++i) {
    for (int shift = 24; shift >= 0; shift -= 8) digest[i * 4 + (24 - shift) / 8] =
        static_cast<uint8_t>(state[i] >> shift);
  }
  return digest;
}

void Fail(const char* message) { throw std::runtime_error(message); }

void Put8(std::vector<uint8_t>* out, uint8_t value) { out->push_back(value); }
void Put16(std::vector<uint8_t>* out, uint16_t value) {
  out->push_back(static_cast<uint8_t>(value >> 8));
  out->push_back(static_cast<uint8_t>(value));
}
void Put32(std::vector<uint8_t>* out, uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8) out->push_back(static_cast<uint8_t>(value >> shift));
}
void Put64(std::vector<uint8_t>* out, int64_t value) {
  const uint64_t bits = static_cast<uint64_t>(value);
  for (int shift = 56; shift >= 0; shift -= 8) out->push_back(static_cast<uint8_t>(bits >> shift));
}
uint8_t Get8(const std::vector<uint8_t>& bytes, std::size_t* offset) {
  if (*offset >= bytes.size()) Fail("truncated MLVC media unit");
  return bytes[(*offset)++];
}
uint16_t Get16(const std::vector<uint8_t>& bytes, std::size_t* offset) {
  if (bytes.size() - *offset < 2) Fail("truncated MLVC media unit");
  const uint16_t value = static_cast<uint16_t>(bytes[*offset] << 8 | bytes[*offset + 1]);
  *offset += 2;
  return value;
}
uint32_t Get32(const std::vector<uint8_t>& bytes, std::size_t* offset) {
  if (bytes.size() - *offset < 4) Fail("truncated MLVC media unit");
  uint32_t value = 0;
  for (int i = 0; i < 4; ++i) value = (value << 8) | bytes[*offset + i];
  *offset += 4;
  return value;
}
int64_t Get64(const std::vector<uint8_t>& bytes, std::size_t* offset) {
  const uint64_t value = [&] {
    if (bytes.size() - *offset < 8) Fail("truncated MLVC media unit");
    uint64_t result = 0;
    for (int i = 0; i < 8; ++i) result = (result << 8) | bytes[*offset + i];
    *offset += 8;
    return result;
  }();
  return static_cast<int64_t>(value);
}

void PutTlv(std::vector<uint8_t>* out, const MlvcTlv& tlv) {
  if (tlv.value.size() > std::numeric_limits<uint16_t>::max()) Fail("MLVC TLV is too large");
  const bool known = tlv.type == kMlvcTlvCodecBundleSha256 || tlv.type == kMlvcTlvProfileName ||
                     tlv.type == kMlvcTlvColorimetry || tlv.type == kMlvcTlvNominalPolicy ||
                     tlv.type == kMlvcTlvVendorData;
  if ((tlv.type & 0x8000u) != 0 && !known) Fail("unknown critical MLVC TLV");
  if ((tlv.type == kMlvcTlvCodecBundleSha256 && tlv.value.size() != 32) ||
      (tlv.type == kMlvcTlvColorimetry && tlv.value.size() != 4) ||
      (tlv.type == kMlvcTlvNominalPolicy && tlv.value.size() != 16)) {
    Fail("invalid fixed-size MLVC TLV");
  }
  Put16(out, tlv.type);
  Put16(out, static_cast<uint16_t>(tlv.value.size()));
  out->insert(out->end(), tlv.value.begin(), tlv.value.end());
  while ((out->size() & 3u) != 0) out->push_back(0);
}

std::vector<MlvcTlv> ParseTlvs(const std::vector<uint8_t>& unit, std::size_t offset,
                               std::size_t end) {
  if (end > unit.size() || offset > end || (end & 3u) != 0) Fail("invalid MLVC TLV area");
  std::vector<MlvcTlv> result;
  std::set<uint16_t> unique_semantic_tlvs;
  while (offset < end) {
    if (end - offset < 4) Fail("truncated MLVC TLV");
    const uint16_t type = static_cast<uint16_t>(unit[offset] << 8 | unit[offset + 1]);
    const uint16_t length = static_cast<uint16_t>(unit[offset + 2] << 8 | unit[offset + 3]);
    offset += 4;
    const std::size_t padded = (static_cast<std::size_t>(length) + 3u) & ~std::size_t(3u);
    if (padded > end - offset) Fail("MLVC TLV exceeds header");
    for (std::size_t i = length; i < padded; ++i) {
      if (unit[offset + i] != 0) Fail("non-zero MLVC TLV padding");
    }
    const bool critical = (type & 0x8000u) != 0;
    const bool known = type == kMlvcTlvCodecBundleSha256 || type == kMlvcTlvProfileName ||
                       type == kMlvcTlvColorimetry || type == kMlvcTlvNominalPolicy ||
                       type == kMlvcTlvVendorData;
    if (critical && !known) Fail("unknown critical MLVC TLV");
    if ((type == kMlvcTlvCodecBundleSha256 && length != 32) ||
        (type == kMlvcTlvColorimetry && length != 4) ||
        (type == kMlvcTlvNominalPolicy && length != 16)) {
      Fail("invalid fixed-size MLVC TLV");
    }
    if (type == kMlvcTlvCodecBundleSha256 || type == kMlvcTlvColorimetry ||
        type == kMlvcTlvNominalPolicy) {
      if (!unique_semantic_tlvs.insert(type).second) {
        Fail("duplicate semantic MLVC TLV");
      }
    }
    result.push_back(MlvcTlv{type, std::vector<uint8_t>(unit.begin() + offset,
                                                         unit.begin() + offset + length)});
    offset += padded;
  }
  return result;
}

const MlvcTlv* FindTlv(const std::vector<MlvcTlv>& tlvs, uint16_t type) {
  for (const auto& tlv : tlvs) {
    if (tlv.type == type) return &tlv;
  }
  return nullptr;
}

bool IsZero(const std::array<uint8_t, 32>& value) {
  return std::all_of(value.begin(), value.end(), [](uint8_t byte) { return byte == 0; });
}

bool IsZero(const std::array<uint8_t, 16>& value) {
  return std::all_of(value.begin(), value.end(), [](uint8_t byte) { return byte == 0; });
}

void ValidateCommon(const MlvcCommonHeader& header, std::size_t actual_size) {
  if (static_cast<uint8_t>(header.unit_type) == 0) {
    Fail("unknown MLVC media unit type");
  }
  if (header.header_length < kMlvcMediaUnitCommonHeaderBytes ||
      (header.header_length & 3u) != 0 ||
      header.header_length > header.unit_length || header.header_length > kMlvcHeaderMaxBytes ||
      header.unit_length < kMlvcMediaUnitCommonHeaderBytes ||
      header.unit_length > kMlvcMediaUnitMaxBytes || header.unit_length != actual_size) {
    Fail("invalid MLVC media unit length");
  }
}

void ValidateReferenceIds(const MlvcEfu& efu) {
  const bool random_access = (efu.unit_flags & kEfuRandomAccess) != 0;
  const bool reset = (efu.unit_flags & kEfuResetReference) != 0;
  if (efu.frame_type == 0) {
    if (!random_access || !reset || efu.short_ref_frame_id != kMlvcNoReference ||
        efu.long_ref_frame_id != kMlvcNoReference) {
      Fail("invalid MLVC Intra EFU references");
    }
    return;
  }
  if (random_access) Fail("only Intra EFU may be random access");
  if (efu.frame_type == 1 && efu.long_ref_frame_id != kMlvcNoReference) {
    Fail("inter EFU must not carry an LTR reference");
  }
  if (reset && efu.short_ref_frame_id != kMlvcNoReference) {
    Fail("reset EFU carries a short reference");
  }
  if (efu.frame_type == 1 && !reset && efu.short_ref_frame_id == kMlvcNoReference) {
    Fail("inter EFU is missing its short reference");
  }
  if (efu.frame_type == 2 &&
      (efu.long_ref_frame_id == kMlvcNoReference || efu.short_ref_frame_id != kMlvcNoReference)) {
    Fail("invalid MLVC LTR reference");
  }
  if (efu.short_ref_frame_id != kMlvcNoReference &&
      CompareMlvcSerial32(efu.frame_id, efu.short_ref_frame_id) !=
          MlvcSerial32Order::kNewer) {
    Fail("MLVC short reference must point to an earlier frame");
  }
  if (efu.long_ref_frame_id != kMlvcNoReference &&
      CompareMlvcSerial32(efu.frame_id, efu.long_ref_frame_id) !=
          MlvcSerial32Order::kNewer) {
    Fail("MLVC long reference must point to an earlier frame");
  }
}

std::vector<uint8_t> MakeCommon(MlvcMediaUnitType type, uint16_t header_length,
                                uint32_t unit_length, uint32_t unit_id, uint32_t config_id,
                                uint32_t flags) {
  std::vector<uint8_t> bytes;
  bytes.reserve(unit_length);
  Put8(&bytes, kMlvcMediaUnitVersion);
  Put8(&bytes, static_cast<uint8_t>(type));
  Put16(&bytes, header_length);
  Put32(&bytes, unit_length);
  Put32(&bytes, unit_id);
  Put32(&bytes, config_id);
  Put32(&bytes, flags);
  return bytes;
}

}  // namespace

uint32_t MlvcCrc32c(const std::vector<uint8_t>& bytes) {
  uint32_t crc = 0xffffffffu;
  for (uint8_t byte : bytes) {
    crc ^= byte;
    for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ (0x82f63b78u & -(crc & 1u));
  }
  return ~crc;
}

std::vector<uint8_t> SerializeScu(const MlvcScu& scu) {
  if (scu.config_id == 0 || scu.coded_width == 0 || scu.coded_height == 0 ||
      scu.visible_width == 0 || scu.visible_height == 0 || scu.timebase_num == 0 ||
      scu.timebase_den == 0 || scu.nominal_fps_num == 0 || scu.nominal_fps_den == 0 ||
      scu.visible_width > scu.coded_width || scu.visible_height > scu.coded_height ||
      scu.coded_width > 16384 || scu.coded_height > 16384 ||
      scu.q_index_count != 64 || (scu.color_format != 1 && scu.color_format != 2) ||
      (scu.bit_depth != 8 && scu.bit_depth != 10 && scu.bit_depth != 16) ||
      scu.entropy_mode != 1 || scu.max_frame_bytes == 0 ||
      (scu.unit_flags & (kScuRequiresRandomAccess | kScuStaticTensorShape)) !=
          (kScuRequiresRandomAccess | kScuStaticTensorShape) ||
      scu.max_frame_bytes > kMlvcMediaUnitMaxBytes - kMlvcEfuFixedBytes ||
      (scu.unit_flags & ~7u) != 0) {
    Fail("invalid MLVC SCU fields");
  }
  std::vector<MlvcTlv> tlvs = scu.tlvs;
  std::array<uint8_t, 32> bundle_hash = scu.codec_bundle_sha256;
  if (IsZero(bundle_hash)) Fail("MLVC SCU requires a model bundle SHA-256");
  std::array<uint8_t, 16> config_hash = scu.config_hash;
  const MlvcTlv* bundle = FindTlv(tlvs, kMlvcTlvCodecBundleSha256);
  if (bundle == nullptr) {
    tlvs.push_back(MlvcTlv{kMlvcTlvCodecBundleSha256,
                           std::vector<uint8_t>(bundle_hash.begin(), bundle_hash.end())});
  } else if (bundle->value.size() != 32) {
    Fail("MLVC SCU bundle hash must be 32 bytes");
  } else if (!std::equal(bundle_hash.begin(), bundle_hash.end(), bundle->value.begin())) {
    Fail("MLVC SCU bundle hash TLV disagrees with the SCU fixed field");
  }
  if (std::count_if(tlvs.begin(), tlvs.end(), [](const MlvcTlv& tlv) {
        return tlv.type == kMlvcTlvCodecBundleSha256;
      }) != 1) {
    Fail("MLVC SCU must contain exactly one model bundle hash TLV");
  }
  for (const uint16_t semantic_type : {kMlvcTlvColorimetry, kMlvcTlvNominalPolicy}) {
    if (std::count_if(tlvs.begin(), tlvs.end(), [semantic_type](const MlvcTlv& tlv) {
          return tlv.type == semantic_type;
        }) > 1) {
      Fail("MLVC SCU contains a duplicate semantic TLV");
    }
  }
  std::array<uint8_t, 16> expected_config_hash{};
  std::copy_n(bundle_hash.begin(), expected_config_hash.size(), expected_config_hash.begin());
  if (IsZero(config_hash)) {
    std::copy(expected_config_hash.begin(), expected_config_hash.end(), config_hash.begin());
  } else if (config_hash != expected_config_hash) {
    Fail("MLVC SCU config hash does not match its fields");
  }
  std::vector<uint8_t> tlv_bytes;
  for (const auto& tlv : tlvs) PutTlv(&tlv_bytes, tlv);
  if (kMlvcScuFixedBytes + tlv_bytes.size() > kMlvcScuMaxBytes ||
      kMlvcScuFixedBytes + tlv_bytes.size() > std::numeric_limits<uint16_t>::max()) {
    Fail("MLVC SCU is too large");
  }
  const uint16_t header_length = static_cast<uint16_t>(kMlvcScuFixedBytes + tlv_bytes.size());
  auto bytes = MakeCommon(MlvcMediaUnitType::kScu, header_length, header_length, scu.config_id,
                          scu.config_id, scu.unit_flags);
  Put16(&bytes, scu.profile_id); Put16(&bytes, scu.level_id);
  Put32(&bytes, scu.coded_width); Put32(&bytes, scu.coded_height);
  Put32(&bytes, scu.visible_width); Put32(&bytes, scu.visible_height);
  Put32(&bytes, scu.timebase_num); Put32(&bytes, scu.timebase_den);
  Put32(&bytes, scu.nominal_fps_num); Put32(&bytes, scu.nominal_fps_den);
  Put8(&bytes, scu.color_format); Put8(&bytes, scu.bit_depth); Put8(&bytes, scu.entropy_mode);
  Put8(&bytes, scu.q_index_count); Put32(&bytes, scu.max_frame_bytes);
  bytes.insert(bytes.end(), config_hash.begin(), config_hash.end());
  bytes.insert(bytes.end(), tlv_bytes.begin(), tlv_bytes.end());
  return bytes;
}

MlvcCommonHeader ParseMediaUnitHeader(const std::vector<uint8_t>& unit) {
  if (unit.size() < kMlvcMediaUnitCommonHeaderBytes || unit[0] != kMlvcMediaUnitVersion) {
    Fail("invalid MLVC media unit version or length");
  }
  std::size_t offset = 1;
  MlvcCommonHeader header;
  header.unit_type = static_cast<MlvcMediaUnitType>(Get8(unit, &offset));
  header.header_length = Get16(unit, &offset);
  header.unit_length = Get32(unit, &offset);
  header.unit_id = Get32(unit, &offset);
  header.config_id = Get32(unit, &offset);
  header.unit_flags = Get32(unit, &offset);
  ValidateCommon(header, unit.size());
  return header;
}

MlvcScu ParseScu(const std::vector<uint8_t>& unit) {
  const MlvcCommonHeader common = ParseMediaUnitHeader(unit);
  if (common.unit_type != MlvcMediaUnitType::kScu || common.header_length < kMlvcScuFixedBytes ||
      common.unit_length != common.header_length || common.unit_length > kMlvcScuMaxBytes ||
      common.unit_id != common.config_id ||
      (common.unit_flags & (kScuRequiresRandomAccess | kScuStaticTensorShape)) !=
          (kScuRequiresRandomAccess | kScuStaticTensorShape)) {
    Fail("invalid MLVC SCU header");
  }
  std::size_t offset = kMlvcMediaUnitCommonHeaderBytes;
  MlvcScu scu;
  scu.config_id = common.config_id; scu.unit_flags = common.unit_flags;
  scu.profile_id = Get16(unit, &offset); scu.level_id = Get16(unit, &offset);
  scu.coded_width = Get32(unit, &offset); scu.coded_height = Get32(unit, &offset);
  scu.visible_width = Get32(unit, &offset); scu.visible_height = Get32(unit, &offset);
  scu.timebase_num = Get32(unit, &offset); scu.timebase_den = Get32(unit, &offset);
  scu.nominal_fps_num = Get32(unit, &offset); scu.nominal_fps_den = Get32(unit, &offset);
  scu.color_format = Get8(unit, &offset); scu.bit_depth = Get8(unit, &offset);
  scu.entropy_mode = Get8(unit, &offset); scu.q_index_count = Get8(unit, &offset);
  scu.max_frame_bytes = Get32(unit, &offset);
  std::copy_n(unit.begin() + offset, scu.config_hash.size(), scu.config_hash.begin());
  offset += scu.config_hash.size();
  scu.tlvs = ParseTlvs(unit, offset, common.header_length);
  const MlvcTlv* bundle = FindTlv(scu.tlvs, kMlvcTlvCodecBundleSha256);
  if (bundle == nullptr || bundle->value.size() != 32) Fail("MLVC SCU bundle hash is missing");
  std::copy(bundle->value.begin(), bundle->value.end(), scu.codec_bundle_sha256.begin());
  if (IsZero(scu.codec_bundle_sha256)) Fail("MLVC SCU model bundle hash is empty");
  if (scu.coded_width == 0 || scu.coded_height == 0 || scu.visible_width == 0 ||
      scu.visible_height == 0 || scu.timebase_num == 0 || scu.timebase_den == 0 ||
      scu.visible_width > scu.coded_width || scu.visible_height > scu.coded_height ||
      scu.nominal_fps_num == 0 || scu.nominal_fps_den == 0 || scu.q_index_count != 64 ||
      (scu.color_format != 1 && scu.color_format != 2) ||
      (scu.bit_depth != 8 && scu.bit_depth != 10 && scu.bit_depth != 16) ||
      scu.coded_width > 16384 || scu.coded_height > 16384 ||
       scu.entropy_mode != 1 || scu.max_frame_bytes == 0 ||
       (scu.unit_flags & (kScuRequiresRandomAccess | kScuStaticTensorShape)) !=
           (kScuRequiresRandomAccess | kScuStaticTensorShape) ||
       scu.max_frame_bytes > kMlvcMediaUnitMaxBytes - kMlvcEfuFixedBytes) {
    Fail("invalid MLVC SCU values");
  }
  if (std::count_if(scu.tlvs.begin(), scu.tlvs.end(), [](const MlvcTlv& tlv) {
        return tlv.type == kMlvcTlvCodecBundleSha256;
      }) != 1) {
    Fail("MLVC SCU must contain exactly one model bundle hash TLV");
  }
  std::array<uint8_t, 16> expected_config_hash{};
  std::copy_n(scu.codec_bundle_sha256.begin(), expected_config_hash.size(),
              expected_config_hash.begin());
  if (scu.config_hash != expected_config_hash) {
    Fail("MLVC SCU config hash does not match its fields");
  }
  return scu;
}

bool MlvcScuDecoderCompatible(const MlvcScu& current, const MlvcScu& next) {
  if (current.profile_id != next.profile_id || current.level_id != next.level_id ||
      current.coded_width != next.coded_width || current.coded_height != next.coded_height ||
      current.visible_width != next.visible_width || current.visible_height != next.visible_height ||
      current.timebase_num != next.timebase_num || current.timebase_den != next.timebase_den ||
      static_cast<uint64_t>(current.nominal_fps_num) * next.nominal_fps_den !=
          static_cast<uint64_t>(next.nominal_fps_num) * current.nominal_fps_den ||
      current.color_format != next.color_format || current.bit_depth != next.bit_depth ||
      current.entropy_mode != next.entropy_mode || current.q_index_count != next.q_index_count ||
      (current.unit_flags & 7u) != (next.unit_flags & 7u) ||
      current.codec_bundle_sha256 != next.codec_bundle_sha256 ||
      current.config_hash != next.config_hash) {
    return false;
  }
  const auto colorimetry = [](const MlvcScu& scu) -> const MlvcTlv* {
    const auto it = std::find_if(scu.tlvs.begin(), scu.tlvs.end(), [](const MlvcTlv& tlv) {
      return tlv.type == kMlvcTlvColorimetry;
    });
    return it == scu.tlvs.end() ? nullptr : &*it;
  };
  const MlvcTlv* current_colorimetry = colorimetry(current);
  const MlvcTlv* next_colorimetry = colorimetry(next);
  return (current_colorimetry == nullptr && next_colorimetry == nullptr) ||
         (current_colorimetry != nullptr && next_colorimetry != nullptr &&
          current_colorimetry->value == next_colorimetry->value);
}

std::vector<uint8_t> SerializeEfu(const MlvcEfu& efu) {
  if (efu.config_id == 0 || efu.frame_id == kMlvcNoReference || efu.frame_type > 2 || efu.pts < 0 ||
      efu.entropy_q_index >= 64 ||
      efu.model_q_index >= 64 || efu.temporal_id != 0 || (efu.unit_flags & ~0x3fu) != 0 ||
      (efu.unit_flags & kEfuDiscardable) != 0) {
    Fail("invalid MLVC EFU fields");
  }
  ValidateReferenceIds(efu);
  if (efu.entropy_payload.size() > std::numeric_limits<uint32_t>::max()) Fail("EFU payload too large");
  std::vector<uint8_t> tlv_bytes;
  for (const auto& tlv : efu.tlvs) PutTlv(&tlv_bytes, tlv);
  const std::size_t header_size = kMlvcEfuFixedBytes + tlv_bytes.size();
  const std::size_t unit_size = header_size + efu.entropy_payload.size();
  if (header_size > kMlvcHeaderMaxBytes || unit_size > kMlvcMediaUnitMaxBytes ||
      header_size > std::numeric_limits<uint16_t>::max() || unit_size > std::numeric_limits<uint32_t>::max()) {
    Fail("MLVC EFU is too large");
  }
  auto bytes = MakeCommon(MlvcMediaUnitType::kEfu, static_cast<uint16_t>(header_size),
                          static_cast<uint32_t>(unit_size), efu.frame_id, efu.config_id,
                          efu.unit_flags | kEfuCrcPresent);
  Put8(&bytes, efu.frame_type); Put8(&bytes, efu.entropy_q_index); Put8(&bytes, efu.model_q_index);
  Put8(&bytes, efu.temporal_id); Put64(&bytes, efu.pts); Put32(&bytes, efu.short_ref_frame_id);
  Put32(&bytes, efu.long_ref_frame_id); Put32(&bytes, static_cast<uint32_t>(efu.entropy_payload.size()));
  Put32(&bytes, MlvcCrc32c(efu.entropy_payload)); Put32(&bytes, 0);
  bytes.insert(bytes.end(), tlv_bytes.begin(), tlv_bytes.end());
  bytes.insert(bytes.end(), efu.entropy_payload.begin(), efu.entropy_payload.end());
  return bytes;
}

MlvcEfu ParseEfu(const std::vector<uint8_t>& unit) {
  const MlvcCommonHeader common = ParseMediaUnitHeader(unit);
  if (common.unit_type != MlvcMediaUnitType::kEfu || common.header_length < kMlvcEfuFixedBytes ||
      common.config_id == 0 || common.unit_id == kMlvcNoReference ||
      (common.unit_flags & ~0x3fu) != 0 ||
      (common.unit_flags & kEfuCrcPresent) == 0 ||
      (common.unit_flags & kEfuDiscardable) != 0) {
    Fail("invalid MLVC EFU header");
  }
  std::size_t offset = kMlvcMediaUnitCommonHeaderBytes;
  MlvcEfu efu;
  efu.config_id = common.config_id; efu.frame_id = common.unit_id; efu.unit_flags = common.unit_flags;
  efu.frame_type = Get8(unit, &offset); efu.entropy_q_index = Get8(unit, &offset);
  efu.model_q_index = Get8(unit, &offset); efu.temporal_id = Get8(unit, &offset);
  efu.pts = Get64(unit, &offset); efu.short_ref_frame_id = Get32(unit, &offset);
  efu.long_ref_frame_id = Get32(unit, &offset);
  const uint32_t payload_size = Get32(unit, &offset);
  const uint32_t payload_crc = Get32(unit, &offset);
  if (Get32(unit, &offset) != 0 || efu.pts < 0 || efu.frame_type > 2 || efu.entropy_q_index >= 64 ||
      efu.model_q_index >= 64 || efu.temporal_id != 0 ||
      payload_size != common.unit_length - common.header_length ||
      payload_size > unit.size() - common.header_length) {
    Fail("invalid MLVC EFU metadata");
  }
  efu.tlvs = ParseTlvs(unit, offset, common.header_length);
  efu.entropy_payload.assign(unit.begin() + common.header_length, unit.end());
  if (MlvcCrc32c(efu.entropy_payload) != payload_crc) Fail("MLVC EFU CRC-32C mismatch");
  ValidateReferenceIds(efu);
  return efu;
}

std::vector<uint8_t> SerializeEos(uint32_t config_id) {
  if (config_id == 0) Fail("EOS requires a config ID");
  return MakeCommon(MlvcMediaUnitType::kEos, 20, 20, 0xffffffffu, config_id, 0);
}

std::vector<uint8_t> SerializeEos(uint32_t config_id, uint32_t next_frame_id) {
  if (config_id == 0 || next_frame_id > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    Fail("invalid RTP EOS configuration or frame boundary");
  }
  return MakeCommon(MlvcMediaUnitType::kEos, 20, 20, next_frame_id, config_id, 0);
}

void ValidateMediaUnit(const std::vector<uint8_t>& unit) {
  const auto header = ParseMediaUnitHeader(unit);
  if (header.unit_type == MlvcMediaUnitType::kScu) {
    (void)ParseScu(unit);
  } else if (header.unit_type == MlvcMediaUnitType::kEfu) {
    (void)ParseEfu(unit);
  } else if (header.unit_type == MlvcMediaUnitType::kEos) {
    if (header.header_length != 20 || header.unit_length != 20 || header.config_id == 0 ||
        header.unit_flags != 0 ||
        (header.unit_id != 0xffffffffu &&
         header.unit_id > static_cast<uint32_t>(std::numeric_limits<int>::max()))) {
      Fail("invalid MLVC EOS");
    }
  }
}

}  // namespace mlvc::transport
