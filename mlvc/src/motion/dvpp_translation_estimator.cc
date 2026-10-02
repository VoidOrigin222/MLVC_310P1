#include <mlvc/motion/dvpp_translation_estimator.h>

#include <stdexcept>
#include <mlvc/runtime/acl_runtime.h>

namespace mlvc::motion {
namespace {

io::DvppH264EncoderConfig MotionProxyConfig(io::DvppH264EncoderConfig config) {
  config.persistent_channel = true;
  config.single_reference = true;
  // Host uploads use the encoder-owned DVPP pool, with no intermediate copy.
  config.zero_copy_input = false;
  // Avoid the legacy RTSP publisher channel; each proxy owns its channel.
  if (config.channel == 0) config.channel = 1;
  return config;
}

}  // namespace

DvppTranslationEstimator::DvppTranslationEstimator(aclrtContext context,
                                                   io::DvppH264EncoderConfig config,
                                                   bool skip_loop_filter)
    : context_(context), layout_(config.layout), gop_(config.gop),
      encoder_(context, MotionProxyConfig(config)), extractor_(config.gop, skip_loop_filter) {}

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
  return EstimateNv12Host(nv12_host_.data(), nv12_host_.size(), frame_index, random_access);
}

Translation DvppTranslationEstimator::EstimateNv12Host(const void* nv12_host, std::size_t bytes,
                                                       uint64_t frame_index,
                                                       bool random_access) {
  if (frame_index != next_frame_) {
    throw std::runtime_error("DVPP motion proxy requires consecutive frames from zero");
  }
  if (nv12_host == nullptr || bytes != io::Nv12BufferSize(layout_)) {
    throw std::runtime_error("DVPP motion host NV12 must match the configured layout");
  }
  const bool keyframe = random_access || frame_index % gop_ == 0;
  auto packet = encoder_.EncodeHostNv12(nv12_host, bytes, keyframe);
  return DecodePacket(packet, frame_index, random_access);
}

Translation DvppTranslationEstimator::EstimateDevice(const void* nv12_device, std::size_t bytes,
                                                     uint64_t frame_index, aclrtEvent ready_event,
                                                     bool random_access) {
  if (frame_index != next_frame_) {
    throw std::runtime_error("DVPP motion proxy requires consecutive frames from zero");
  }
  const bool keyframe = random_access || frame_index % gop_ == 0;
  auto packet = encoder_.EncodeDevice(nv12_device, bytes, ready_event, keyframe);
  return DecodePacket(packet, frame_index, random_access);
}

Translation DvppTranslationEstimator::DecodePacket(const std::vector<uint8_t>& packet,
                                                   uint64_t frame_index, bool random_access) {
  auto result = extractor_.Decode(packet.data(), packet.size(), frame_index, random_access);
  ++next_frame_;
  return result;
}

}  // namespace mlvc::motion
