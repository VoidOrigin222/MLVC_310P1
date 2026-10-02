#include <mlvc/motion/h264_reference_policy.h>

#include <algorithm>
#include <climits>
#include <stdexcept>
#include <vector>

namespace mlvc::motion {
namespace {

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

class Bits {
 public:
  Bits(const uint8_t* source, std::size_t bytes) {
    unsigned zeros = 0;
    for (std::size_t i = 0; i < bytes; ++i) {
      if (zeros >= 2 && source[i] == 3) {
        Require(i + 1 < bytes && source[i + 1] <= 3, "invalid H.264 emulation-prevention byte");
        zeros = 0;
        continue;
      }
      data_.push_back(source[i]);
      zeros = source[i] == 0 ? zeros + 1 : 0;
    }
  }
  unsigned Read(unsigned count = 1) {
    Require(count <= 32 && bit_ + count <= data_.size() * 8, "truncated H.264 reference syntax");
    unsigned value = 0;
    for (unsigned i = 0; i < count; ++i, ++bit_) {
      value = (value << 1) | ((data_[bit_ / 8] >> (7 - bit_ % 8)) & 1);
    }
    return value;
  }
  unsigned Ue() {
    unsigned zeros = 0;
    while (Read() == 0) {
      ++zeros;
      Require(zeros < 31, "H.264 Exp-Golomb value is too large");
    }
    return ((1U << zeros) - 1) + Read(zeros);
  }
  int Se() {
    const unsigned value = Ue();
    return value & 1 ? static_cast<int>((value + 1) / 2) : -static_cast<int>(value / 2);
  }
  void ScalingList(int size) {
    int last = 8;
    int next = 8;
    for (int i = 0; i < size; ++i) {
      if (next != 0) next = (last + Se() + 256) % 256;
      if (next != 0) last = next;
    }
  }
 private:
  std::vector<uint8_t> data_;
  std::size_t bit_ = 0;
};

std::size_t StartCode(const uint8_t* data, std::size_t bytes, std::size_t from) {
  for (std::size_t i = from; i + 2 < bytes; ++i) {
    if (data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1) return i;
    if (i + 3 < bytes && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 0 &&
        data[i + 3] == 1) return i;
  }
  return bytes;
}

}  // namespace

void H264ReferencePolicy::Inspect(const uint8_t* data, std::size_t bytes, bool keyframe) {
  Require(data && bytes > 0 && bytes <= static_cast<std::size_t>(INT_MAX),
          "empty or oversized H.264 proxy access unit");
  std::size_t start = StartCode(data, bytes, 0);
  Require(start < bytes, "motion proxy requires Annex B H.264 packets");
  unsigned slices = 0;
  unsigned picture_frame_num = 0;
  unsigned frame_num_modulus = 0;
  while (start < bytes) {
    const std::size_t nal = start + (data[start + 2] == 1 ? 3 : 4);
    Require(nal < bytes, "empty H.264 NAL unit");
    const std::size_t next = StartCode(data, bytes, nal + 1);
    const unsigned type = data[nal] & 31;
    Require((data[nal] & 128) == 0, "invalid H.264 forbidden-zero bit");
    Require(type < 2 || type > 4, "motion proxy does not support partitioned H.264 slices");
    Require(type != 20 && type != 21, "motion proxy does not support MVC H.264 slices");
    const unsigned reference_idc = (data[nal] >> 5) & 3;
    Bits bits(data + nal + 1, next - nal - 1);
    if (type == 7) {
      Sps sps;
      const unsigned profile = bits.Read(8);
      bits.Read(8); // constraints
      bits.Read(8); // level
      const unsigned id = bits.Ue();
      Require(id <= 31, "invalid H.264 SPS identifier");
      if (profile == 100 || profile == 110 || profile == 122 || profile == 244 || profile == 44 ||
          profile == 83 || profile == 86 || profile == 118 || profile == 128 || profile == 138 ||
          profile == 139 || profile == 134 || profile == 135) {
        sps.chroma_format = bits.Ue();
        Require(sps.chroma_format <= 3, "unsupported H.264 chroma syntax");
        if (sps.chroma_format == 3) sps.separate_color_plane = bits.Read();
        bits.Ue(); // bit_depth_luma_minus8
        bits.Ue(); // bit_depth_chroma_minus8
        bits.Read(); // qpprime_y_zero_transform_bypass_flag
        if (bits.Read()) {
          const int count = sps.chroma_format == 3 ? 12 : 8;
          for (int i = 0; i < count; ++i) if (bits.Read()) bits.ScalingList(i < 6 ? 16 : 64);
        }
      }
      sps.frame_num_bits = bits.Ue() + 4;
      Require(sps.frame_num_bits <= 16, "invalid H.264 frame number width");
      sps.poc_type = bits.Ue();
      Require(sps.poc_type <= 2, "invalid H.264 POC type");
      if (sps.poc_type == 0) {
        sps.poc_bits = bits.Ue() + 4;
        Require(sps.poc_bits <= 16, "invalid H.264 POC width");
      } else if (sps.poc_type == 1) {
        sps.delta_poc_always_zero = bits.Read();
        bits.Se(); bits.Se();
        const unsigned cycle = bits.Ue();
        Require(cycle <= 255, "invalid H.264 POC cycle");
        for (unsigned i = 0; i < cycle; ++i) bits.Se();
      }
      bits.Ue(); // max_num_ref_frames is DPB capacity, not active count
      bits.Read(); // gaps_in_frame_num_value_allowed_flag
      bits.Ue(); bits.Ue(); // dimensions in macroblocks
      sps.frame_mbs_only = bits.Read();
      Require(sps.frame_mbs_only, "motion proxy requires progressive H.264 frames");
      sps_[id] = sps;
    } else if (type == 8) {
      const unsigned id = bits.Ue();
      Require(id <= 255, "invalid H.264 PPS identifier");
      Pps pps;
      pps.sps_id = bits.Ue();
      Require(pps.sps_id <= 31, "invalid H.264 PPS sequence identifier");
      bits.Read(); // entropy_coding_mode_flag
      pps.bottom_poc_present = bits.Read();
      Require(bits.Ue() == 0, "motion proxy does not support H.264 slice groups");
      pps.default_l0 = bits.Ue() + 1;
      bits.Ue(); // num_ref_idx_l1_default_active_minus1
      pps.weighted_pred = bits.Read();
      bits.Read(2); // weighted_bipred_idc
      bits.Se(); bits.Se(); bits.Se();
      bits.Read(); bits.Read();
      pps.redundant_pic_count = bits.Read();
      pps_[id] = pps;
    } else if (type == 1 || type == 5) {
      ++slices;
      Require(type == (keyframe ? 5U : 1U), "motion proxy keyframes must be IDR pictures");
      Require(reference_idc > 0,
              "motion proxy frames must retain the immediately preceding reference");
      bits.Ue(); // first_mb_in_slice
      const unsigned slice_type = bits.Ue() % 5;
      Require(slice_type == (keyframe ? 2U : 0U), "motion proxy requires I/P-only slice syntax");
      const unsigned pps_id = bits.Ue();
      Require(pps_.count(pps_id), "H.264 slice references an unknown PPS");
      const Pps& pps = pps_.at(pps_id);
      Require(sps_.count(pps.sps_id), "H.264 PPS references an unknown SPS");
      const Sps& sps = sps_.at(pps.sps_id);
      if (sps.separate_color_plane) bits.Read(2);
      const unsigned frame_num = bits.Read(sps.frame_num_bits);
      Require(keyframe ? frame_num == 0 : have_reference_ && frame_num == next_frame_num_,
              "H.264 proxy reference frame numbers are not consecutive");
      if (slices == 1) picture_frame_num = frame_num;
      Require(frame_num == picture_frame_num,
              "H.264 access unit contains different picture numbers");
      frame_num_modulus = 1U << sps.frame_num_bits;
      if (type == 5) bits.Ue(); // idr_pic_id
      if (sps.poc_type == 0) {
        bits.Read(sps.poc_bits);
        if (pps.bottom_poc_present) bits.Se();
      } else if (sps.poc_type == 1 && !sps.delta_poc_always_zero) {
        bits.Se();
        if (pps.bottom_poc_present) bits.Se();
      }
      if (pps.redundant_pic_count) bits.Ue();
      if (slice_type == 0) {
        unsigned active_l0 = pps.default_l0;
        if (bits.Read()) active_l0 = bits.Ue() + 1;
        Require(active_l0 == 1, "H.264 proxy P slice uses more than one active reference");
        Require(bits.Read() == 0, "H.264 proxy modifies the previous-frame reference list");
        if (pps.weighted_pred) {
          bits.Ue(); // luma_log2_weight_denom
          const bool chroma = !sps.separate_color_plane && sps.chroma_format != 0;
          if (chroma) bits.Ue();
          if (bits.Read()) { bits.Se(); bits.Se(); }
          if (chroma && bits.Read()) for (int i = 0; i < 2; ++i) { bits.Se(); bits.Se(); }
        }
      }
      if (type == 5) {
        bits.Read(); // no_output_of_prior_pics_flag
        Require(bits.Read() == 0, "H.264 proxy IDR must not be a long-term reference");
      } else {
        Require(bits.Read() == 0,
                "H.264 proxy uses adaptive reference marking; previous-frame policy unverified");
      }
    }
    start = next;
  }
  Require(slices > 0, "H.264 proxy access unit contains no picture slices");
  have_reference_ = true;
  next_frame_num_ = (picture_frame_num + 1) % frame_num_modulus;
}

}  // namespace mlvc::motion
