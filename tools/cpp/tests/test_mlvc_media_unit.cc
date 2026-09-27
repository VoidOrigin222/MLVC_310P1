#include "mlvc/transport/mlvc_media_unit.h"

#include <cassert>
#include <cstdint>
#include <vector>

using namespace mlvc::transport;

int main() {
  MlvcScu scu;
  scu.config_id = 7;
  scu.coded_width = 1920;
  scu.coded_height = 1088;
  scu.visible_width = 1920;
  scu.visible_height = 1080;
  scu.nominal_fps_num = 30000;
  scu.nominal_fps_den = 1001;
  for (std::size_t i = 0; i < scu.codec_bundle_sha256.size(); ++i) {
    scu.codec_bundle_sha256[i] = static_cast<uint8_t>(i + 1);
  }
  const auto scu_bytes = SerializeScu(scu);
  assert(scu_bytes.size() >= kMlvcScuFixedBytes);
  assert(scu_bytes[0] == 1 && scu_bytes[1] == 1);
  const auto parsed_scu = ParseScu(scu_bytes);
  assert(parsed_scu.config_id == 7 && parsed_scu.coded_width == 1920);
  assert(parsed_scu.nominal_fps_num == 30000 && parsed_scu.nominal_fps_den == 1001);
  assert(parsed_scu.codec_bundle_sha256 == scu.codec_bundle_sha256);
  for (std::size_t i = 0; i < parsed_scu.config_hash.size(); ++i) {
    assert(parsed_scu.config_hash[i] == scu.codec_bundle_sha256[i]);
  }
  assert(CompareMlvcSerial32(0, 0xffffffffu) == MlvcSerial32Order::kNewer);
  assert(CompareMlvcSerial32(0xffffffffu, 0) == MlvcSerial32Order::kOlder);
  assert(CompareMlvcSerial32(0x80000000u, 0) == MlvcSerial32Order::kAmbiguous);
  MlvcScu policy_switch = scu;
  policy_switch.config_id = 8;
  policy_switch.tlvs.push_back(
      MlvcTlv{kMlvcTlvNominalPolicy, std::vector<uint8_t>(16, 1)});
  assert(MlvcScuDecoderCompatible(parsed_scu, ParseScu(SerializeScu(policy_switch))));
  MlvcScu color_switch = policy_switch;
  color_switch.tlvs.clear();
  color_switch.tlvs.push_back(MlvcTlv{kMlvcTlvColorimetry, {1, 2, 3, 4}});
  assert(!MlvcScuDecoderCompatible(parsed_scu, ParseScu(SerializeScu(color_switch))));
  auto bad_scu_hash = scu_bytes;
  bad_scu_hash[64] ^= 0x01;  // config_hash starts at byte 64 in the fixed SCU header.
  bool rejected_scu_hash = false;
  try {
    (void)ParseScu(bad_scu_hash);
  } catch (...) {
    rejected_scu_hash = true;
  }
  assert(rejected_scu_hash);
  auto reserved_scu_flags = scu_bytes;
  reserved_scu_flags[16] |= 0x80;  // Reserved SCU flags are ignored by readers.
  assert(ParseScu(reserved_scu_flags).config_id == scu.config_id);
  auto duplicate_policy = scu;
  duplicate_policy.tlvs.push_back(
      MlvcTlv{kMlvcTlvNominalPolicy, std::vector<uint8_t>(16, 0)});
  duplicate_policy.tlvs.push_back(
      MlvcTlv{kMlvcTlvNominalPolicy, std::vector<uint8_t>(16, 1)});
  bool rejected_duplicate_policy = false;
  try {
    (void)SerializeScu(duplicate_policy);
  } catch (...) {
    rejected_duplicate_policy = true;
  }
  assert(rejected_duplicate_policy);
  MlvcScu missing_bundle = scu;
  missing_bundle.codec_bundle_sha256.fill(0);
  bool rejected_missing_bundle = false;
  try {
    (void)SerializeScu(missing_bundle);
  } catch (...) {
    rejected_missing_bundle = true;
  }
  assert(rejected_missing_bundle);

  MlvcEfu efu;
  efu.config_id = 7;
  efu.frame_id = 3;
  efu.frame_type = 0;
  efu.entropy_q_index = 12;
  efu.model_q_index = 13;
  efu.unit_flags |= kEfuRandomAccess | kEfuResetReference;
  efu.entropy_payload = {0, 1, 2, 3, 4};
  const auto efu_bytes = SerializeEfu(efu);
  const auto parsed_efu = ParseEfu(efu_bytes);
  assert(parsed_efu.frame_id == 3 && parsed_efu.model_q_index == 13);
  assert(parsed_efu.entropy_payload == efu.entropy_payload);
  MlvcEfu wrapped_reference;
  wrapped_reference.config_id = 7;
  wrapped_reference.frame_id = 0;
  wrapped_reference.frame_type = 1;
  wrapped_reference.short_ref_frame_id = 0xfffffffeu;
  assert(ParseEfu(SerializeEfu(wrapped_reference)).short_ref_frame_id == 0xfffffffeu);
  auto inter_with_ltr = wrapped_reference;
  inter_with_ltr.long_ref_frame_id = 0xfffffffeu;
  bool rejected_inter_ltr = false;
  try {
    (void)SerializeEfu(inter_with_ltr);
  } catch (...) {
    rejected_inter_ltr = true;
  }
  assert(rejected_inter_ltr);
  auto discardable = efu_bytes;
  // unit_flags is the final word of the common header.  EFU fragments are
  // never discardable in protocol v1; accepting this bit would allow a
  // sender queue to silently drop a reference frame.
  discardable[19] = static_cast<uint8_t>(discardable[19] | kEfuDiscardable);
  bool rejected_discardable = false;
  try {
    (void)ParseEfu(discardable);
  } catch (...) {
    rejected_discardable = true;
  }
  assert(rejected_discardable);
  auto corrupt = efu_bytes;
  corrupt.back() ^= 0x80;
  bool rejected = false;
  try {
    (void)ParseEfu(corrupt);
  } catch (...) {
    rejected = true;
  }
  assert(rejected);
  const auto eos = SerializeEos(7);
  ValidateMediaUnit(eos);
  const auto bounded_eos = SerializeEos(7, 1234);
  assert(ParseMediaUnitHeader(bounded_eos).unit_id == 1234);
  ValidateMediaUnit(bounded_eos);

  // Unassigned media-unit types are length-delimited and skippable.  Their
  // payload is not interpreted by a version-1 decoder.
  std::vector<uint8_t> optional_extension(20, 0);
  optional_extension[0] = kMlvcMediaUnitVersion;
  optional_extension[1] = 0x05;
  optional_extension[3] = 20;
  optional_extension[7] = 20;
  ValidateMediaUnit(optional_extension);

  MlvcEfu invalid_pts = efu;
  invalid_pts.pts = -1;
  bool rejected_negative_pts = false;
  try {
    (void)SerializeEfu(invalid_pts);
  } catch (...) {
    rejected_negative_pts = true;
  }
  assert(rejected_negative_pts);
  return 0;
}
