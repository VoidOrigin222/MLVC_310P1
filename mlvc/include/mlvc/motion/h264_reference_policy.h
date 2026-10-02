#ifndef MLVC_MOTION_H264_REFERENCE_POLICY_H_
#define MLVC_MOTION_H264_REFERENCE_POLICY_H_

#include <cstddef>
#include <cstdint>
#include <map>

namespace mlvc::motion {

// Checks active reference count and list/marking syntax, rather than equating
// SPS DPB capacity with the references actually used by each P slice.
class H264ReferencePolicy {
 public:
  void Inspect(const uint8_t* annex_b, std::size_t bytes, bool keyframe);

 private:
  struct Sps {
    int frame_num_bits = 0;
    int poc_type = 0;
    int poc_bits = 0;
    int chroma_format = 1;
    bool separate_color_plane = false;
    bool delta_poc_always_zero = false;
    bool frame_mbs_only = true;
  };
  struct Pps {
    unsigned sps_id = 0;
    unsigned default_l0 = 1;
    bool bottom_poc_present = false;
    bool redundant_pic_count = false;
    bool weighted_pred = false;
  };
  std::map<unsigned, Sps> sps_;
  std::map<unsigned, Pps> pps_;
  unsigned next_frame_num_ = 0;
  bool have_reference_ = false;
};

}  // namespace mlvc::motion

#endif  // MLVC_MOTION_H264_REFERENCE_POLICY_H_
