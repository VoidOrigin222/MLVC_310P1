#ifndef MLVC_MOTION_DVPP_TRANSLATION_ESTIMATOR_H_
#define MLVC_MOTION_DVPP_TRANSLATION_ESTIMATOR_H_

#include <mlvc/io/dvpp_h264_encoder.h>
#include <mlvc/motion/translation_estimator.h>

namespace mlvc::motion {

// Experimental persistent hardware proxy. VBR changes motion decisions relative
// to libx264 CRF18. Software decode is used solely because VDEC does not expose
// AVMotionVector data. Any unexpected I/B/reference policy is rejected.
class DvppTranslationEstimator {
 public:
  DvppTranslationEstimator(aclrtContext context, io::DvppH264EncoderConfig config,
                           bool skip_loop_filter = false);
  Translation Estimate(const codec::TensorData& input, uint64_t frame_index,
                       bool random_access = false);
  // NV12 must use the exact layout/strides configured at construction and
  // contain Nv12BufferSize(layout) bytes. Copied once into the owned DVPP pool.
  Translation EstimateNv12Host(const void* nv12_host, std::size_t bytes, uint64_t frame_index,
                               bool random_access = false);
  Translation EstimateDevice(const void* nv12_device, std::size_t bytes, uint64_t frame_index,
                             aclrtEvent ready_event = nullptr, bool random_access = false);

 private:
  uint64_t next_frame_ = 0;
  aclrtContext context_ = nullptr;
  io::Nv12Layout layout_;
  uint32_t gop_;
  Translation DecodePacket(const std::vector<uint8_t>& packet, uint64_t frame_index,
                           bool random_access);
  std::vector<uint8_t> nv12_host_;
  io::DvppH264Encoder encoder_;
  H264MotionExtractor extractor_;
};

}  // namespace mlvc::motion

#endif  // MLVC_MOTION_DVPP_TRANSLATION_ESTIMATOR_H_
