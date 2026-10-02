#include <mlvc/motion/h264_reference_policy.h>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

class Writer {
 public:
  explicit Writer(uint8_t header) : bytes_{0, 0, 1, header} {}
  void Put(unsigned value, unsigned count = 1) {
    for (unsigned i = count; i > 0; --i) {
      if (offset_ == 0) bytes_.push_back(0);
      bytes_.back() |= ((value >> (i - 1)) & 1) << (7 - offset_);
      offset_ = (offset_ + 1) % 8;
    }
  }
  void Ue(unsigned value) {
    const unsigned code = value + 1;
    unsigned length = 0;
    for (unsigned n = code; n; n >>= 1) ++length;
    Put(0, length - 1);
    Put(code, length);
  }
  std::vector<uint8_t> Finish() { Put(1); return bytes_; }
 private:
  std::vector<uint8_t> bytes_;
  unsigned offset_ = 0;
};

std::vector<uint8_t> Keyframe(unsigned default_refs = 1) {
  // Baseline progressive SPS, 4-bit frame_num, POC type 2 and a two-frame DPB.
  Writer sps(0x67);
  sps.Put(66, 8); sps.Put(0, 8); sps.Put(30, 8); sps.Ue(0);
  sps.Ue(0); sps.Ue(2); sps.Ue(2); sps.Put(0);
  sps.Ue(15); sps.Ue(7); sps.Put(1); sps.Put(1); sps.Put(0); sps.Put(0);
  auto packet = sps.Finish();
  Writer pps(0x68);
  pps.Ue(0); pps.Ue(0); pps.Put(0); pps.Put(0); pps.Ue(0);
  pps.Ue(default_refs - 1); pps.Ue(0); pps.Put(0); pps.Put(0, 2);
  pps.Ue(0); pps.Ue(0); pps.Ue(0); pps.Put(1); pps.Put(0); pps.Put(0);
  const auto pps_bytes = pps.Finish();
  packet.insert(packet.end(), pps_bytes.begin(), pps_bytes.end());
  Writer slice(0x65);
  slice.Ue(0); slice.Ue(2); slice.Ue(0); slice.Put(0, 4); slice.Ue(0);
  slice.Put(0); slice.Put(0);
  const auto slice_bytes = slice.Finish();
  packet.insert(packet.end(), slice_bytes.begin(), slice_bytes.end());
  return packet;
}

std::vector<uint8_t> PFrame(unsigned frame_num = 1, unsigned override_refs = 0,
                            bool modify = false, bool adaptive = false, bool is_reference = true,
                            unsigned slice_type = 0) {
  Writer slice(is_reference ? 0x41 : 0x01);
  slice.Ue(0); slice.Ue(slice_type); slice.Ue(0); slice.Put(frame_num, 4);
  slice.Put(override_refs > 0);
  if (override_refs > 0) slice.Ue(override_refs - 1);
  slice.Put(modify);
  slice.Put(adaptive);
  return slice.Finish();
}

void ExpectRejected(const std::vector<uint8_t>& packet, unsigned default_refs = 1) {
  mlvc::motion::H264ReferencePolicy policy;
  const auto keyframe = Keyframe(default_refs);
  policy.Inspect(keyframe.data(), keyframe.size(), true);
  bool rejected = false;
  try { policy.Inspect(packet.data(), packet.size(), false); }
  catch (const std::exception&) { rejected = true; }
  if (!rejected) throw std::runtime_error("unsafe H.264 reference syntax was accepted");
}

}  // namespace

int main() {
  try {
    mlvc::motion::H264ReferencePolicy policy;
    auto keyframe = Keyframe();
    policy.Inspect(keyframe.data(), keyframe.size(), true);
    for (unsigned frame = 1; frame < 20; ++frame) {
      auto packet = PFrame(frame % 16);
      policy.Inspect(packet.data(), packet.size(), false);
    }
    // DPB capacity 2 with one active L0 entry and no reordering is valid.
    policy.Inspect(keyframe.data(), keyframe.size(), true);
    auto packet = PFrame();
    policy.Inspect(packet.data(), packet.size(), false);
    ExpectRejected(PFrame(1, 2));
    ExpectRejected(PFrame(), 2);
    ExpectRejected(PFrame(1, 0, true));
    ExpectRejected(PFrame(1, 0, false, true));
    ExpectRejected(PFrame(1, 0, false, false, false));
    ExpectRejected(PFrame(2));
    ExpectRejected(PFrame(1, 0, false, false, true, 1));
    mlvc::motion::H264ReferencePolicy override_policy;
    keyframe = Keyframe(2);
    override_policy.Inspect(keyframe.data(), keyframe.size(), true);
    packet = PFrame(1, 1);
    override_policy.Inspect(packet.data(), packet.size(), false);
    ExpectRejected({0, 0, 1, 0x41}); // truncated slice
    ExpectRejected({0, 0, 1, 0x67}); // truncated SPS
    ExpectRejected({0, 0, 1, 0x68}); // truncated PPS
    ExpectRejected({0, 0, 1, 0x09, 0x10}); // access unit without a picture
    ExpectRejected({0, 0, 1, 0x41, 0, 0, 3, 4}); // invalid emulation prevention
    ExpectRejected({0, 0, 1, 0xC1, 0x80}); // forbidden-zero bit
    ExpectRejected({0, 0, 1, 0x42, 0x80}); // partitioned slices
    ExpectRejected({0, 0, 1, 0x54, 0x80}); // MVC slices
    ExpectRejected({0, 0, 0, 8, 0x41}); // no Annex B start code
    std::cout << "h264_reference_policy status=ok\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "h264_reference_policy status=failed: " << error.what() << '\n';
    return 1;
  }
}
