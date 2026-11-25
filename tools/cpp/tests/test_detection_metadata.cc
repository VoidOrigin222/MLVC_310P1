#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "ulbvc/detection/geometry.h"
#include "ulbvc/transport/detection_metadata.h"

using namespace ulbvc::transport;
template <class Function>
void Reject(Function function) {
  bool rejected = false;
  try {
    function();
  } catch (const std::exception&) {
    rejected = true;
  }
  assert(rejected);
}
int main() {
  DetectionFrame frame{42, 1920, 1080, {{0, -1, 0.9f, 960, 540, 100, 200, "person"}}};
  const auto tlv = SerializeDetections(frame);
  auto restored = ParseDetections(tlv);
  assert(restored.frame_id == 42 && restored.width == 1920 && restored.height == 1080);
  assert(restored.boxes.size() == 1 && restored.boxes[0].label == "person");
  assert(restored.boxes[0].track_id == -1 && restored.boxes[0].cx == 960);
  UlbvcEfu efu;
  efu.unit_flags |= kEfuRandomAccess | kEfuResetReference;
  efu.frame_id = 42;
  efu.tlvs.push_back(tlv);
  efu.entropy_payload = {1, 2, 3};
  const auto parsed = ParseEfu(SerializeEfu(efu));
  assert(ParseDetections(parsed.tlvs.at(0)).frame_id == parsed.frame_id);
  assert(parsed.entropy_payload == efu.entropy_payload);
  for (std::size_t size = 0; size < tlv.value.size(); ++size) {
    auto truncated = tlv;
    truncated.value.resize(size);
    Reject([&] { ParseDetections(truncated); });
  }
  auto invalid = frame;
  invalid.boxes[0].confidence = std::numeric_limits<float>::quiet_NaN();
  Reject([&] { SerializeDetections(invalid); });
  invalid = frame;
  invalid.boxes.resize(kMaxDetections + 1);
  Reject([&] { SerializeDetections(invalid); });
  frame.boxes.clear();
  assert(ParseDetections(SerializeDetections(frame)).boxes.empty());

  const auto full = ulbvc::detection::PlanLetterbox(1920, 1080, 1920, 1088);
  const auto small = ulbvc::detection::PlanLetterbox(1920, 1080, 1280, 736);
  assert(full.left == 0 && full.top == 4 && full.gain == 1);
  assert(small.left == 0 && small.top == 8 && small.resized_height == 720);
  DetectionBox box{0, -1, 1, 640, 368, 1280, 720, "person"};
  assert(ulbvc::detection::RestoreBox(&box, small));
  assert(box.cx == 960 && box.cy == 540 && box.width == 1920 && box.height == 1080);
  box = {0, -1, 1, 640, 4, 10, 4, "person"};
  assert(!ulbvc::detection::RestoreBox(&box, small));
  box = {0, -1, 1, 960, 544, 1920, 1080, "person"};
  assert(ulbvc::detection::RestoreBox(&box, full) && box.cy == 540 && box.height == 1080);
  std::cout << "detection metadata, frame association and letterbox geometry passed\n";
}
