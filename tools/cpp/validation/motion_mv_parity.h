#ifndef MLVC_VALIDATION_MOTION_MV_PARITY_H_
#define MLVC_VALIDATION_MOTION_MV_PARITY_H_

#include <mlvc/motion/translation_estimator.h>
#include <mlvc/motion/h264_reference_policy.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace mlvc::validation {

// Two independent decoder histories consume exactly the same access units.
// Compare fields rather than struct storage (which may contain padding).
class MotionMvParity {
 public:
  explicit MotionMvParity(int gop) : baseline_(gop, false), fast_(gop, true), gop_(gop) {}

  motion::Translation Inspect(const uint8_t* packet, std::size_t bytes,
                              uint64_t index, bool random_access) {
    if (index != frames_) throw std::runtime_error("MV parity observer missed/reordered a frame");
    std::vector<motion::MotionVector> baseline, fast;
    const auto a = baseline_.Decode(packet, bytes, index, random_access, &baseline);
    const auto b = fast_.Decode(packet, bytes, index, random_access, &fast);
    if (baseline.size() != fast.size() || a.kx != b.kx || a.ky != b.ky ||
        a.tx != b.tx || a.ty != b.ty || a.vector_count != b.vector_count) {
      throw std::runtime_error("loop-filter decode changed MV count or translation");
    }
    for (std::size_t i = 0; i < baseline.size(); ++i) {
      const auto& x = baseline[i];
      const auto& y = fast[i];
      if (x.source != y.source || x.w != y.w || x.h != y.h ||
          x.motion_x != y.motion_x || x.motion_y != y.motion_y ||
          x.motion_scale != y.motion_scale) {
        throw std::runtime_error("loop-filter decode changed a raw MV field/order");
      }
      Hash(static_cast<uint32_t>(x.source), 4);
      Hash(x.w, 1); Hash(x.h, 1);
      Hash(static_cast<uint32_t>(x.motion_x), 4);
      Hash(static_cast<uint32_t>(x.motion_y), 4);
      Hash(x.motion_scale, 2);
    }
    vectors_ += baseline.size();
    if (index == 0) first_packet_.assign(packet, packet + bytes);
    if (index == 1 && !random_access && index % static_cast<uint64_t>(gop_) != 0)
      second_packet_.assign(packet, packet + bytes);
    ++frames_;
    return a;
  }

  void CheckTruncatedPRejected() const {
    if (second_packet_.empty()) throw std::runtime_error("MV error test needs a P frame");
    std::size_t last_slice = 0;
    for (std::size_t i = 0; i + 3 < second_packet_.size(); ++i) {
      if (second_packet_[i] == 0 && second_packet_[i + 1] == 0 && second_packet_[i + 2] == 1 &&
          (second_packet_[i + 3] & 31) == 1) last_slice = i + 3;
    }
    if (last_slice == 0 || second_packet_.size() - last_slice <= 16)
      throw std::runtime_error("P fixture is too short for a macroblock truncation test");
    const auto payload = second_packet_.size() - last_slice;
    // FFmpeg tolerates some truncations; this is a fixture-selection check,
    // not a claim that arbitrary corruption is detectable. Preserve valid
    // reference-policy headers and choose a short candidate FFmpeg reports.
    for (std::size_t retained : {payload / 2, payload / 4, std::size_t(16),
                                 std::size_t(12), std::size_t(8), std::size_t(6),
                                 std::size_t(4), std::size_t(3), std::size_t(2), std::size_t(1)}) {
      if (retained >= payload) continue;
      auto truncated = second_packet_;
      truncated.resize(last_slice + retained);
      motion::H264ReferencePolicy headers;
      try {
        headers.Inspect(first_packet_.data(), first_packet_.size(), true);
        headers.Inspect(truncated.data(), truncated.size(), false);
      } catch (const std::exception&) { continue; }
      motion::H264MotionExtractor strict(gop_, true);
      (void)strict.Decode(first_packet_.data(), first_packet_.size(), 0);
      try { (void)strict.Decode(truncated.data(), truncated.size(), 1); }
      catch (const std::exception&) { return; }
    }
    throw std::runtime_error("no header-valid truncated P candidate produced a decoder error");
  }

  uint64_t frames() const { return frames_; }
  uint64_t vectors() const { return vectors_; }
  uint64_t field_hash() const { return hash_; }

 private:
  void Hash(uint64_t value, unsigned bytes) {
    for (unsigned i = 0; i < bytes; ++i) {
      hash_ ^= static_cast<uint8_t>(value >> (8 * i));
      hash_ *= 1099511628211ULL;
    }
  }
  motion::H264MotionExtractor baseline_, fast_;
  int gop_;
  uint64_t frames_ = 0, vectors_ = 0, hash_ = 14695981039346656037ULL;
  std::vector<uint8_t> first_packet_, second_packet_;
};

}  // namespace mlvc::validation

#endif  // MLVC_VALIDATION_MOTION_MV_PARITY_H_
