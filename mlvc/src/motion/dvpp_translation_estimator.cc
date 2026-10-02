#include <mlvc/motion/dvpp_translation_estimator.h>

#include <stdexcept>
#include <mlvc/runtime/acl_runtime.h>

namespace mlvc::motion {
namespace {

io::DvppH264EncoderConfig MotionProxyConfig(io::DvppH264EncoderConfig config) {
  config.persistent_channel = true;
  config.single_reference = true;
  // Avoid the legacy RTSP publisher channel; each proxy owns its channel.
  if (config.channel == 0) config.channel = 1;
  return config;
}

}  // namespace

DvppTranslationEstimator::DvppTranslationEstimator(aclrtContext context,
                                                   io::DvppH264EncoderConfig config)
    : context_(context), layout_(config.layout), gop_(config.gop),
      encoder_(context, MotionProxyConfig(config)), extractor_(config.gop) {}

Translation DvppTranslationEstimator::Estimate(const codec::TensorData& input,
                                               uint64_t frame_index, bool random_access) {
  if (frame_index != next_frame_) {
    throw std::runtime_error("DVPP motion proxy requires consecutive frames from zero");
  }
  const auto location = input.View().location();
  if ((location != MemoryLocation::kCpu && location != MemoryLocation::kPinnedCpu) ||
      input.dtype != DataType::kFloat16 ||
      input.ByteSize() != input.shape.NumElements() * sizeof(uint16_t)) {
    throw std::runtime_error("DVPP motion FP16 input must be a valid host tensor");
  }
  codec::TensorData mirror;
  const codec::TensorData* host = &input;
  if (input.has_external_buffer()) {
    mirror.shape = input.shape;
    mirror.dtype = input.dtype;
    const auto* data = static_cast<const uint8_t*>(input.external_data);
    mirror.bytes.assign(data, data + input.ByteSize());
    host = &mirror;
  }
  io::ConvertFp16Yuv444ToNv12(*host, layout_, &nv12_host_);
  CheckAcl(aclrtSetCurrentContext(context_), "set context for DVPP motion proxy upload");
  if (nv12_device_.bytes() != nv12_host_.size()) nv12_device_.Allocate(nv12_host_.size());
  CheckAcl(aclrtMemcpy(nv12_device_.data(), nv12_device_.bytes(), nv12_host_.data(),
                        nv12_host_.size(), ACL_MEMCPY_HOST_TO_DEVICE),
           "upload DVPP motion proxy NV12");
  return EstimateDevice(nv12_device_.data(), nv12_device_.bytes(), frame_index, nullptr,
                        random_access);
}

Translation DvppTranslationEstimator::EstimateDevice(const void* nv12_device, std::size_t bytes,
                                                     uint64_t frame_index, aclrtEvent ready_event,
                                                     bool random_access) {
  if (frame_index != next_frame_) {
    throw std::runtime_error("DVPP motion proxy requires consecutive frames from zero");
  }
  const bool keyframe = random_access || frame_index % gop_ == 0;
  auto packet = encoder_.EncodeDevice(nv12_device, bytes, ready_event, keyframe);
  auto result = extractor_.Decode(packet.data(), packet.size(), frame_index, random_access);
  ++next_frame_;
  return result;
}

}  // namespace mlvc::motion
