#include <mlvc/io/dvpp_h264_encoder.h>

#include <acl/dvpp/hi_dvpp_venc.h>
#include <acl/dvpp/hi_mpi_sys.h>
#include <acl/dvpp/hi_dvpp_vb.h>

#include <chrono>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "mlvc/core/status.h"
#include "mlvc/runtime/acl_runtime.h"

namespace mlvc::io {
namespace {

constexpr std::size_t kVencStreamBufferBytes = 4U * 1024U * 1024U;

void CheckMpi(hi_s32 status, const char* operation) {
  if (status != HI_SUCCESS) {
    throw Error(std::string(operation) + " failed: ret=" + std::to_string(status));
  }
}

hi_u32 ToMpiBitrate(std::uint32_t bits_per_second) {
  const std::uint64_t kilobits_per_second =
      (static_cast<std::uint64_t>(bits_per_second) + 999U) / 1000U;
  Check(kilobits_per_second >= 2 && kilobits_per_second <= 614400,
        "DVPP H.264 bitrate must be between 2 kbps and 614400 kbps");
  return static_cast<hi_u32>(kilobits_per_second);
}

}  // namespace

class DvppH264Encoder::Impl {
 public:
  Impl(aclrtContext context, DvppH264EncoderConfig config)
      : context_(context), config_(std::move(config)) {
    Check(context_ != nullptr, "DVPP H.264 encoder requires an ACL context");
    Check(config_.channel < 256, "DVPP H.264 encoder channel must be in [0, 256)");
    channel_ = static_cast<hi_venc_chn>(config_.channel);
    Check(config_.layout.width > 0 && config_.layout.height > 0 &&
              config_.layout.width % 2 == 0 && config_.layout.height % 2 == 0,
          "DVPP H.264 dimensions must be positive and even");
    Check(config_.layout.width_stride >= config_.layout.width &&
              config_.layout.height_stride >= config_.layout.height &&
              config_.layout.width_stride % 2 == 0 && config_.layout.height_stride % 2 == 0,
          "DVPP H.264 strides are invalid");
    Check(config_.fps > 0 && config_.fps <= 240 && config_.gop > 0,
          "DVPP H.264 frame rate or GOP is invalid");
    input_bytes_ = Nv12BufferSize(config_.layout);

    try {
      CheckAcl(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext for MPI VENC");
      CheckMpi(hi_mpi_sys_init(), "hi_mpi_sys_init");
      mpi_initialized_ = true;
      CreateChannel();
      StartChannel();
      if (!config_.zero_copy_input) {
        for (void*& input_buffer : input_buffers_) {
          CheckMpi(hi_mpi_dvpp_malloc(0, &input_buffer, input_bytes_),
                   "hi_mpi_dvpp_malloc VENC input");
          Check(input_buffer != nullptr, "hi_mpi_dvpp_malloc returned a null VENC input buffer");
        }
        input_buffer_ = input_buffers_.front();
      }
      ConfigureFrame();
    } catch (...) {
      Cleanup();
      throw;
    }
  }

  ~Impl() { Cleanup(); }

  std::vector<std::uint8_t> EncodeDevice(const void* nv12_device, std::size_t bytes,
                                         aclrtEvent ready_event, bool force_keyframe) {
    Check(nv12_device != nullptr, "DVPP H.264 input must not be null");
    Check(bytes == input_bytes_, "DVPP H.264 input NV12 size mismatch");
    // CANN 9.1 on the 310P1 rejects the second submission when this channel
    // remains alive (HI_ERR_VENC_ILLEGAL_PARAM); keep the validated workaround
    // until a runtime with persistent-channel support is available.
    if (frame_index_ > 0 && !config_.persistent_channel) RestartChannel();
    CheckAcl(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext MPI VENC encode");
    if (ready_event != nullptr) {
      CheckAcl(aclrtSynchronizeEvent(ready_event), "aclrtSynchronizeEvent MPI VENC input");
    }
    input_buffer_ = config_.zero_copy_input
                        ? const_cast<void*>(nv12_device)
                        : input_buffers_[frame_index_ % input_buffers_.size()];
    // Keep the public frame metadata fresh for each MPI submission. Some vendor
    // runtimes update bookkeeping in the frame descriptor despite its const API.
    ConfigureFrame();
    if (!config_.zero_copy_input) {
      CheckAcl(aclrtMemcpy(input_buffer_, input_bytes_, nv12_device, input_bytes_,
                           ACL_MEMCPY_DEVICE_TO_DEVICE),
               "aclrtMemcpy NV12 to MPI VENC input");
    }

    // The opt-in persistent proxy uses increasing microsecond timestamps and
    // frame references. Preserve the validated legacy restart metadata.
    frame_.v_frame.pts = config_.persistent_channel ? frame_index_ * 1000000U / config_.fps : 0;
    frame_.v_frame.time_ref =
        config_.persistent_channel ? static_cast<hi_u32>(frame_index_ * 2) : 0;
    if (force_keyframe) {
      CheckMpi(hi_mpi_venc_request_idr(channel_, HI_TRUE), "hi_mpi_venc_request_idr");
    }
    const hi_s32 send_status = hi_mpi_venc_send_frame(channel_, &frame_, 1000);
    if (send_status != HI_SUCCESS) {
      throw Error("hi_mpi_venc_send_frame failed at frame " + std::to_string(frame_index_) +
                  " (" + std::to_string(config_.layout.width) + "x" +
                  std::to_string(config_.layout.height) + ", stride=" +
                  std::to_string(config_.layout.width_stride) + "x" +
                  std::to_string(config_.layout.height_stride) + ", ret=" +
                  std::to_string(send_status) + ")");
    }
    std::vector<std::uint8_t> output = GetStream();
    ++frame_index_;
    return output;
  }

 private:
  void ConfigureFrame() {
    frame_ = {};
    // This frame uses our own hi_mpi_dvpp_malloc allocation, not a VB pool.
    // Pool 0 happens to accept the first frame on 310P1, but later sends are
    // rejected as illegal parameters.
    frame_.pool_id = static_cast<hi_u32>(HI_VB_INVALID_POOLID);
    frame_.mod_id = HI_ID_USER;
    frame_.v_frame.width = static_cast<hi_u32>(config_.layout.width);
    frame_.v_frame.height = static_cast<hi_u32>(config_.layout.height);
    frame_.v_frame.field = HI_VIDEO_FIELD_FRAME;
    frame_.v_frame.pixel_format = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    frame_.v_frame.video_format = HI_VIDEO_FORMAT_LINEAR;
    frame_.v_frame.compress_mode = HI_COMPRESS_MODE_NONE;
    frame_.v_frame.dynamic_range = HI_DYNAMIC_RANGE_SDR8;
    frame_.v_frame.color_gamut = HI_COLOR_GAMUT_BT709;
    frame_.v_frame.width_stride[0] = static_cast<hi_u32>(config_.layout.width_stride);
    frame_.v_frame.width_stride[1] = static_cast<hi_u32>(config_.layout.width_stride);
    frame_.v_frame.height_stride[0] = static_cast<hi_u32>(config_.layout.height_stride);
    frame_.v_frame.height_stride[1] = static_cast<hi_u32>(config_.layout.height_stride / 2);
    frame_.v_frame.virt_addr[0] = input_buffer_;
    frame_.v_frame.virt_addr[1] = static_cast<hi_u8*>(input_buffer_) +
                                  config_.layout.width_stride * config_.layout.height_stride;
  }

  void CreateChannel() {
    hi_venc_chn_attr attr{};
    attr.venc_attr.type = HI_PT_H264;
    attr.venc_attr.profile = 0;
    attr.venc_attr.buf_size = static_cast<hi_u32>(kVencStreamBufferBytes);
    attr.venc_attr.max_pic_width = static_cast<hi_u32>(config_.layout.width);
    attr.venc_attr.max_pic_height = static_cast<hi_u32>(config_.layout.height);
    attr.venc_attr.pic_width = static_cast<hi_u32>(config_.layout.width);
    attr.venc_attr.pic_height = static_cast<hi_u32>(config_.layout.height);
    attr.venc_attr.is_by_frame = HI_TRUE;
    attr.venc_attr.h264_attr.rcn_ref_share_buf_en = HI_FALSE;
    attr.rc_attr.rc_mode = HI_VENC_RC_MODE_H264_VBR;
    attr.rc_attr.h264_vbr.gop = config_.gop;
    attr.rc_attr.h264_vbr.stats_time = 1;
    attr.rc_attr.h264_vbr.src_frame_rate = config_.fps;
    attr.rc_attr.h264_vbr.dst_frame_rate = config_.fps;
    attr.rc_attr.h264_vbr.max_bit_rate = ToMpiBitrate(config_.bitrate);
    attr.gop_attr.gop_mode = HI_VENC_GOP_MODE_NORMAL_P;
    attr.gop_attr.normal_p.ip_qp_delta = 3;

    CheckMpi(hi_mpi_venc_create_chn(channel_, &attr), "hi_mpi_venc_create_chn H.264");
    channel_created_ = true;
    CheckMpi(hi_mpi_venc_set_scene_mode(channel_, HI_VENC_SCENE_0),
             "hi_mpi_venc_set_scene_mode");
    hi_venc_rc_param rc_param{};
    CheckMpi(hi_mpi_venc_get_rc_param(channel_, &rc_param), "hi_mpi_venc_get_rc_param");
    CheckMpi(hi_mpi_venc_set_rc_param(channel_, &rc_param), "hi_mpi_venc_set_rc_param");
    if (config_.single_reference) {
      hi_venc_ref_param ref_param{};
      ref_param.base = 1;
      ref_param.enhance = 0;
      // Huawei 1x mode: base=1, enhance=0, pred_en=true. false anchors
      // every P frame to the GOP IDR and is unsuitable for adjacent motion.
      ref_param.pred_en = HI_TRUE;
      CheckMpi(hi_mpi_venc_set_ref_param(channel_, &ref_param),
               "hi_mpi_venc_set_ref_param single-reference proxy");
      hi_venc_ref_param actual{};
      CheckMpi(hi_mpi_venc_get_ref_param(channel_, &actual),
               "hi_mpi_venc_get_ref_param single-reference proxy");
      Check(actual.base == 1 && actual.enhance == 0 && actual.pred_en == HI_TRUE,
            "DVPP failed to retain the single-reference proxy configuration");
    }
  }

  void StartChannel() {
    hi_venc_start_param start_param{};
    // Request the maximum documented receive count instead of the -1
    // unlimited sentinel; this CANN runtime has rejected later frames with
    // HI_ERR_VENC_ILLEGAL_PARAM when started with -1.
    start_param.recv_pic_num = 2147483647;
    CheckMpi(hi_mpi_venc_start_chn(channel_, &start_param), "hi_mpi_venc_start_chn");
    channel_started_ = true;
  }

  void RestartChannel() {
    if (channel_started_) {
      CheckMpi(hi_mpi_venc_stop_chn(channel_), "hi_mpi_venc_stop_chn between frames");
      channel_started_ = false;
    }
    if (channel_created_) {
      CheckMpi(hi_mpi_venc_destroy_chn(channel_),
               "hi_mpi_venc_destroy_chn between frames");
      channel_created_ = false;
    }
    CreateChannel();
    StartChannel();
  }

  std::vector<std::uint8_t> GetStream() {
    hi_venc_chn_status status{};
    constexpr int kStatusPollCount = 1000;
    for (int poll = 0; poll < kStatusPollCount; ++poll) {
      CheckMpi(hi_mpi_venc_query_status(channel_, &status), "hi_mpi_venc_query_status");
      if (status.cur_packs > 0) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    Check(status.cur_packs > 0, "timed out waiting for DVPP H.264 output");

    std::vector<hi_venc_pack> packs(status.cur_packs);
    hi_venc_stream stream{};
    stream.pack = packs.data();
    stream.pack_cnt = static_cast<hi_u32>(packs.size());
    CheckMpi(hi_mpi_venc_get_stream(channel_, &stream, 1000), "hi_mpi_venc_get_stream");

    std::vector<std::uint8_t> output;
    try {
      Check(stream.pack_cnt > 0 && stream.pack_cnt <= packs.size(),
            "DVPP H.264 returned an invalid pack count");
      for (hi_u32 index = 0; index < stream.pack_cnt; ++index) {
        const hi_venc_pack& pack = stream.pack[index];
        Check(pack.addr != nullptr && pack.offset <= pack.len,
              "DVPP H.264 returned an invalid stream pack");
        const std::size_t data_bytes = static_cast<std::size_t>(pack.len - pack.offset);
        const auto* data = pack.addr + pack.offset;
        output.insert(output.end(), data, data + data_bytes);
      }
      CheckMpi(hi_mpi_venc_release_stream(channel_, &stream), "hi_mpi_venc_release_stream");
    } catch (...) {
      (void)hi_mpi_venc_release_stream(channel_, &stream);
      throw;
    }
    Check(!output.empty(), "DVPP H.264 returned an empty stream");
    return output;
  }

  void Cleanup() noexcept {
    if (context_ != nullptr) (void)aclrtSetCurrentContext(context_);
    if (channel_started_) {
      (void)hi_mpi_venc_stop_chn(channel_);
      channel_started_ = false;
    }
    if (channel_created_) {
      (void)hi_mpi_venc_destroy_chn(channel_);
      channel_created_ = false;
    }
    for (void*& input_buffer : input_buffers_) {
      if (input_buffer != nullptr) {
        (void)hi_mpi_dvpp_free(input_buffer);
        input_buffer = nullptr;
      }
    }
    input_buffer_ = nullptr;
    if (mpi_initialized_) {
      (void)hi_mpi_sys_exit();
      mpi_initialized_ = false;
    }
  }

  aclrtContext context_ = nullptr;
  DvppH264EncoderConfig config_;
  hi_venc_chn channel_ = 0;
  std::size_t input_bytes_ = 0;
  std::array<void*, 3> input_buffers_{};
  void* input_buffer_ = nullptr;
  hi_video_frame_info frame_{};
  std::uint64_t frame_index_ = 0;
  bool mpi_initialized_ = false;
  bool channel_created_ = false;
  bool channel_started_ = false;
};

DvppH264Encoder::DvppH264Encoder(aclrtContext context, DvppH264EncoderConfig config)
    : impl_(std::make_unique<Impl>(context, std::move(config))) {}

DvppH264Encoder::~DvppH264Encoder() = default;

std::vector<std::uint8_t> DvppH264Encoder::EncodeDevice(const void* nv12_device,
                                                        std::size_t bytes,
                                                        aclrtEvent ready_event,
                                                        bool force_keyframe) {
  return impl_->EncodeDevice(nv12_device, bytes, ready_event, force_keyframe);
}

}  // namespace mlvc::io
