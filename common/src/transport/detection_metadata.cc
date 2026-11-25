#include "ulbvc/transport/detection_metadata.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace ulbvc::transport {
namespace {
void Require(bool value) {
  if (!value) throw std::runtime_error("invalid ulbvc detection metadata");
}
void Put32(std::vector<uint8_t>* out, uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8) out->push_back(value >> shift);
}
void Put16(std::vector<uint8_t>* out, uint16_t value) {
  out->push_back(value >> 8);
  out->push_back(value);
}
uint32_t Get32(const std::vector<uint8_t>& in, std::size_t* pos) {
  Require(*pos + 4 <= in.size());
  uint32_t value = 0;
  for (int i = 0; i < 4; ++i) value = (value << 8) | in[(*pos)++];
  return value;
}
uint16_t Get16(const std::vector<uint8_t>& in, std::size_t* pos) {
  Require(*pos + 2 <= in.size());
  const uint16_t value = (static_cast<uint16_t>(in[*pos]) << 8) | in[*pos + 1];
  *pos += 2;
  return value;
}
void PutFloat(std::vector<uint8_t>* out, float value) {
  static_assert(sizeof(float) == sizeof(uint32_t));
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  Put32(out, bits);
}
float GetFloat(const std::vector<uint8_t>& in, std::size_t* pos) {
  const uint32_t bits = Get32(in, pos);
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}
void Validate(const DetectionFrame& frame) {
  Require(frame.width > 0 && frame.height > 0 && frame.boxes.size() <= kMaxDetections);
  for (const auto& box : frame.boxes) {
    Require(std::isfinite(box.confidence) && box.confidence >= 0 && box.confidence <= 1);
    Require(std::isfinite(box.cx) && std::isfinite(box.cy) && std::isfinite(box.width) &&
            std::isfinite(box.height) && box.width > 0 && box.height > 0);
    Require(box.cx >= 0 && box.cx <= frame.width && box.cy >= 0 && box.cy <= frame.height &&
            box.width <= frame.width && box.height <= frame.height);
    Require(!box.label.empty() && box.label.size() <= 63 &&
            box.label.find('\0') == std::string::npos);
  }
}
}  // namespace

UlbvcTlv SerializeDetections(const DetectionFrame& frame) {
  Validate(frame);
  UlbvcTlv result{kDetectionTlv, {1, 0}};
  auto& out = result.value;
  Put16(&out, frame.boxes.size());
  Put16(&out, frame.width);
  Put16(&out, frame.height);
  Put32(&out, frame.frame_id);
  for (const auto& box : frame.boxes) {
    Put16(&out, box.class_id);
    out.push_back(box.label.size());
    out.push_back(0);
    Put32(&out, static_cast<uint32_t>(box.track_id));
    for (float value : {box.confidence, box.cx, box.cy, box.width, box.height})
      PutFloat(&out, value);
    out.insert(out.end(), box.label.begin(), box.label.end());
  }
  return result;
}

DetectionFrame ParseDetections(const UlbvcTlv& tlv) {
  const auto& in = tlv.value;
  Require(tlv.type == kDetectionTlv && in.size() >= 12 && in[0] == 1 && in[1] == 0);
  std::size_t pos = 2;
  const uint16_t count = Get16(in, &pos);
  Require(count <= kMaxDetections);
  DetectionFrame result;
  result.width = Get16(in, &pos);
  result.height = Get16(in, &pos);
  result.frame_id = Get32(in, &pos);
  for (uint16_t i = 0; i < count; ++i) {
    DetectionBox box;
    box.class_id = Get16(in, &pos);
    Require(pos + 2 <= in.size());
    const uint8_t length = in[pos++];
    Require(in[pos++] == 0);
    const uint32_t track_bits = Get32(in, &pos);
    std::memcpy(&box.track_id, &track_bits, sizeof(track_bits));
    box.confidence = GetFloat(in, &pos);
    box.cx = GetFloat(in, &pos);
    box.cy = GetFloat(in, &pos);
    box.width = GetFloat(in, &pos);
    box.height = GetFloat(in, &pos);
    Require(length <= 63 && pos + length <= in.size());
    box.label.assign(in.begin() + pos, in.begin() + pos + length);
    pos += length;
    result.boxes.push_back(std::move(box));
  }
  Require(pos == in.size());
  Validate(result);
  return result;
}
}  // namespace ulbvc::transport
