#include <cassert>
#include <stdexcept>
#include <chrono>
#include <thread>

#include "mlvc/transport/rtp_mlvc.h"
using namespace mlvc::transport;
int main() {
  RtpMlvcReassembler r;
  std::vector<uint8_t> out;
  std::vector<uint8_t> first_payload(10, 0x11);
  std::vector<uint8_t> second_payload(10, 0x22);
  RtpMlvcFragment a{RtpUnitType::kEfu, kRtpMlvcStart, 1, 2, 0, 20, first_payload},
      b{RtpUnitType::kEfu, kRtpMlvcEnd, 1, 2, 10, 20, second_payload};
  // A bounded pre-start cache preserves packet reordering without allocating
  // the whole unit until the offset-zero fragment arrives.
  assert(!r.Push(b, &out));
  assert(r.Push(a, &out));
  assert(out.size() == 20 && out.front() == 0x11 && out.back() == 0x22);
  // A retransmission with a new RTP sequence is still the same complete
  // media unit and must not be delivered twice.
  assert(!r.Push(a, &out));
  assert(!r.Push(b, &out));
  auto conflicting_complete = a;
  conflicting_complete.payload[0] ^= 0xff;
  bool conflicting_complete_rejected = false;
  try {
    (void)r.Push(conflicting_complete, &out);
  } catch (const std::runtime_error&) {
    conflicting_complete_rejected = true;
  }
  assert(conflicting_complete_rejected);
  // A conflicting duplicate must be rejected while the unit is pending.
  RtpMlvcReassembler conflict;
  assert(!conflict.Push(a, &out));
  auto bad = a;
  bad.payload[0] = 9;
  bool threw = false;
  try {
    conflict.Push(bad, &out);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);
  RtpMlvcReassembler overlap;
  const RtpMlvcFragment overlap_head{RtpUnitType::kEfu, kRtpMlvcStart, 1, 3, 0, 20,
                                     std::vector<uint8_t>(10, 1)};
  const RtpMlvcFragment overlap_tail{RtpUnitType::kEfu, kRtpMlvcEnd, 1, 3, 5, 20,
                                     std::vector<uint8_t>(15, 2)};
  assert(!overlap.Push(overlap_head, &out));
  threw = false;
  try {
    overlap.Push(overlap_tail, &out);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);
  // Reusing the unit ID in a new RTP timestamp starts a fresh logical unit.
  auto next_timestamp_head = a;
  next_timestamp_head.timestamp = 1;
  auto next_timestamp_tail = b;
  next_timestamp_tail.timestamp = 1;
  assert(!r.Push(next_timestamp_head, &out));
  assert(r.Push(next_timestamp_tail, &out));
  RtpMlvcReassembler bounded(1024, 1, std::chrono::milliseconds(10));
  assert(!bounded.Push(a, &out));
  std::this_thread::sleep_for(std::chrono::milliseconds(15));
  assert(!bounded.Push(b, &out));
  RtpMlvcReassembler cleanup_boundary(1024, 1, std::chrono::milliseconds(1000));
  std::this_thread::sleep_for(std::chrono::milliseconds(800));
  assert(!cleanup_boundary.Push(a, &out));
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  assert(cleanup_boundary.Push(b, &out));
  assert(out.size() == 20 && out.front() == 0x11 && out.back() == 0x22);

  // Identical config/unit IDs from different RTP sessions or timestamps must
  // not be allowed to splice into a cross-frame unit.
  RtpMlvcReassembler isolated;
  RtpMlvcFragment first_head{RtpUnitType::kEfu, kRtpMlvcStart, 3, 4, 0, 20,
                             std::vector<uint8_t>(10, 0xaa)};
  first_head.ssrc = 100;
  first_head.timestamp = 1000;
  RtpMlvcFragment other_tail{RtpUnitType::kEfu, kRtpMlvcEnd, 3, 4, 10, 20,
                             std::vector<uint8_t>(10, 0xbb)};
  other_tail.ssrc = 200;
  other_tail.timestamp = 1000;
  assert(!isolated.Push(first_head, &out));
  assert(!isolated.Push(other_tail, &out));
  RtpMlvcFragment first_tail{RtpUnitType::kEfu, kRtpMlvcEnd, 3, 4, 10, 20,
                             std::vector<uint8_t>(10, 0xcc)};
  first_tail.ssrc = 100;
  first_tail.timestamp = 1000;
  assert(isolated.Push(first_tail, &out));
  assert(out.size() == 20 && out.front() == 0xaa && out.back() == 0xcc);
  RtpMlvcFragment other_head{RtpUnitType::kEfu, kRtpMlvcStart, 3, 4, 0, 20,
                             std::vector<uint8_t>(10, 0xdd)};
  other_head.ssrc = 200;
  other_head.timestamp = 1000;
  assert(isolated.Push(other_head, &out));
  assert(out.size() == 20 && out.front() == 0xdd && out.back() == 0xbb);

  RtpMlvcReassembler byte_bounded(1024, 64, std::chrono::milliseconds(1000));
  RtpMlvcFragment large_head{RtpUnitType::kEfu, kRtpMlvcStart, 5, 6, 0, 700, {0x11}};
  RtpMlvcFragment competing_tail{RtpUnitType::kEfu, kRtpMlvcEnd, 5, 7, 399, 400, {0x22}};
  assert(!byte_bounded.Push(large_head, &out));
  assert(!byte_bounded.Push(competing_tail, &out));
  RtpMlvcFragment large_tail{RtpUnitType::kEfu, kRtpMlvcEnd, 5, 6, 1, 700,
                              std::vector<uint8_t>(699, 0x33)};
  assert(!byte_bounded.Push(large_tail, &out));

  // Media bytes alone do not bound the allocation cost of the interval map:
  // one-byte fragments can create a large number of tree nodes. The
  // reassembler applies an independent global fragment-range budget.
  constexpr uint32_t kRangeLimit = 131072;
  RtpMlvcReassembler range_bounded(kRangeLimit + 2, 1, std::chrono::milliseconds(1000));
  bool range_limit_rejected = false;
  for (uint32_t offset = 0; offset <= kRangeLimit; ++offset) {
    const uint8_t flags = offset == 0 ? kRtpMlvcStart : 0;
    RtpMlvcFragment tiny{RtpUnitType::kEfu, flags, 6, 8, offset, kRangeLimit + 2, {0x01}};
    try {
      (void)range_bounded.Push(tiny, &out);
    } catch (const std::runtime_error&) {
      range_limit_rejected = true;
      break;
    }
  }
  assert(range_limit_rejected);

  return 0;
}
