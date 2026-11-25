#ifndef ULBVC_TRANSPORT_DETECTION_METADATA_H_
#define ULBVC_TRANSPORT_DETECTION_METADATA_H_

#include <cstdint>
#include <string>
#include <vector>

#include "ulbvc/transport/ulbvc_media_unit.h"

namespace ulbvc::transport {

constexpr uint16_t kDetectionTlv = 0x0010;
constexpr std::size_t kMaxDetections = 100;

struct DetectionBox {
  uint16_t class_id = 0;
  int32_t track_id = -1;
  float confidence = 0;
  float cx = 0, cy = 0, width = 0, height = 0;
  std::string label;
};

struct DetectionFrame {
  uint32_t frame_id = 0;
  uint16_t width = 0, height = 0;
  std::vector<DetectionBox> boxes;
};

UlbvcTlv SerializeDetections(const DetectionFrame& frame);
DetectionFrame ParseDetections(const UlbvcTlv& tlv);

}  // namespace ulbvc::transport
#endif  // ULBVC_TRANSPORT_DETECTION_METADATA_H_
