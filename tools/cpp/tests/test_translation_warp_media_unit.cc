#include "mlvc/transport/mlvc_media_unit.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void Require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
template <class Function> void Reject(Function function, const char* message) {
  bool rejected = false;
  try { function(); } catch (const std::exception&) { rejected = true; }
  Require(rejected, message);
}
void Put32(std::vector<uint8_t>* bytes, std::size_t offset, uint32_t value) {
  for (int i = 0; i < 4; ++i) (*bytes)[offset + i] = static_cast<uint8_t>(value >> (24 - i * 8));
}
}  // namespace

int main() {
  using namespace mlvc::transport;
  try {
    MlvcScu legacy;
    legacy.coded_width = legacy.visible_width = 64;
    legacy.coded_height = legacy.visible_height = 64;
    legacy.codec_bundle_sha256.fill(0x55);
    const auto legacy_scu = SerializeScu(legacy);
    auto warp = legacy;
    warp.translation_warp = true;
    const auto warp_scu = SerializeScu(warp);
    Require(warp_scu.size() == legacy_scu.size() + 8, "SCU capability overhead must be 8 bytes");
    Require(!ParseScu(legacy_scu).translation_warp && ParseScu(warp_scu).translation_warp,
            "SCU capability round trip");
    Require(!MlvcScuDecoderCompatible(legacy, warp), "warp mode switch must be rejected");
    Require(SerializeScu(ParseScu(legacy_scu)) == legacy_scu, "legacy SCU bytes changed");
    auto invalid = warp;
    invalid.tlvs.push_back({kMlvcTlvTranslationWarp, {2}});
    Reject([&] { SerializeScu(invalid); }, "unsupported warp version accepted");
    invalid.tlvs = {{kMlvcTlvTranslationWarp, {1}}, {kMlvcTlvTranslationWarp, {1}}};
    Reject([&] { SerializeScu(invalid); }, "duplicate capability accepted");
    invalid = legacy;
    invalid.tlvs.push_back({kMlvcTlvTranslationWarp, {1}});
    Reject([&] { SerializeScu(invalid); }, "contradictory capability accepted");

    MlvcEfu iframe;
    iframe.unit_flags |= kEfuRandomAccess | kEfuResetReference;
    iframe.entropy_payload = {8, 9, 10};
    const auto old_i = SerializeEfu(iframe);
    iframe.unit_flags |= kEfuTranslationWarp;
    const auto warp_i = SerializeEfu(iframe);
    Require(warp_i.size() == old_i.size(), "warp I-frame must not add geometry bytes");
    ValidateEfuTranslationWarp(ParseEfu(warp_i), true);
    iframe.kx = 1;
    Reject([&] { SerializeEfu(iframe); }, "nonzero I geometry accepted");

    MlvcEfu pframe;
    pframe.frame_type = 1;
    pframe.frame_id = 1;
    pframe.short_ref_frame_id = 0;
    pframe.entropy_payload = {8, 9, 10};
    const auto old_p = SerializeEfu(pframe);
    Require(SerializeEfu(ParseEfu(old_p)) == old_p, "legacy EFU bytes changed");
    pframe.unit_flags |= kEfuTranslationWarp;
    for (int kx : {-128, -1, 0, 1, 127}) {
      for (int ky : {-128, -1, 0, 1, 127}) {
        pframe.kx = static_cast<int8_t>(kx);
        pframe.ky = static_cast<int8_t>(ky);
        auto bytes = SerializeEfu(pframe);
        Require(bytes.size() == old_p.size() + 2, "P geometry overhead must be exactly 2 bytes");
        auto parsed = ParseEfu(bytes);
        Require(parsed.kx == kx && parsed.ky == ky && parsed.entropy_payload == pframe.entropy_payload,
                "signed geometry round trip");
        ValidateEfuTranslationWarp(parsed, true);
        Reject([&] { ValidateEfuTranslationWarp(parsed, false); }, "unexpected warp flag accepted");
        bytes[kMlvcEfuFixedBytes] ^= 1;
        Reject([&] { ParseEfu(bytes); }, "geometry corruption not covered by CRC");
      }
    }
    Reject([&] { ValidateEfuTranslationWarp(ParseEfu(old_p), true); }, "missing warp flag accepted");
    auto reset_p = pframe;
    reset_p.unit_flags |= kEfuResetReference;
    reset_p.short_ref_frame_id = kMlvcNoReference;
    Reject([&] { SerializeEfu(reset_p); }, "warp P frame accepted removed previous-frame dependency");
    auto reset_wire = SerializeEfu(pframe);
    reset_wire[19] |= kEfuResetReference;
    Put32(&reset_wire, 32, kMlvcNoReference);
    Reject([&] { ParseEfu(reset_wire); }, "valid CRC warp P frame accepted reset reference flag");
    auto truncated = SerializeEfu(pframe);
    truncated.resize(kMlvcEfuFixedBytes + 1);
    Put32(&truncated, 4, static_cast<uint32_t>(truncated.size()));
    Put32(&truncated, 40, 1);
    Put32(&truncated, 44, MlvcCrc32c({truncated.back()}));
    Reject([&] { ParseEfu(truncated); }, "truncated geometry accepted even with valid CRC");
    auto ltr = pframe;
    ltr.frame_type = 2;
    ltr.short_ref_frame_id = kMlvcNoReference;
    ltr.long_ref_frame_id = 0;
    Reject([&] { SerializeEfu(ltr); }, "undefined warp LTR semantics accepted");
    std::cout << "translation warp media unit tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
