#include <mlvc/application/stream/encode/encode_setup.h>
#include <mlvc/codec/execution_profile.h>

#include <filesystem>

#include "mlvc/core/status.h"

namespace mlvc::codec {

void ValidateEncodeInput(const EncodeStreamOptions& options) {
  Check(options.motion_backend == "libx264" || options.motion_backend == "dvpp",
        "motion_backend must be libx264 or dvpp");
  Check(options.translation_warp || options.motion_backend == "libx264",
        "a non-default motion_backend requires translation warp");
  Check(options.motion_shifts_file.empty() || options.motion_backend == "libx264",
        "motion_shifts_file replays saved motion and cannot be combined with motion_backend=dvpp");
  Check(options.translation_warp || options.motion_shifts_file.empty(),
        "motion_shifts_file requires translation warp");
  if (options.translation_warp) {
    Check(options.gop == 96 && options.reset_interval == 32,
          "translation warp currently requires GOP 96 and reset interval 32");
    Check(options.ltr_period == 0 && options.forced_ltr_recovery_frame < 0 &&
              options.forced_ltr_reference_frame < 0,
          "translation warp requires LTR disabled because proxy motion uses the previous frame");
    Check(options.output_transport_port == 0 || options.output_transport_mode == "rtp",
          "translation warp requires MLVC-ES files or RTP transport");
  }
  Check(mlvc::codec::IsCodecExecutionProfile(options.execution_profile),
        "unsupported codec execution profile: " + options.execution_profile);
  Check(!options.manifest_path.empty(), "encode stream requires a manifest path");
  Check(options.input_synthetic || !options.input_camera_device.empty() || !options.input_video_path.empty() ||
            !options.input_frame_dir.empty(),
        "MLVC encode requires input_synthetic, input_camera_device, input_frame_dir, or input_video");
  if (options.input_synthetic) {
    Check(options.input_camera_device.empty() && options.input_video_path.empty() &&
              options.input_frame_dir.empty() && options.frame_num > 0,
          "synthetic input requires a finite frame count and no other source");
    return;
  }
  if (!options.input_camera_device.empty()) {
    Check(options.camera_options.has_value(), "camera options are required for camera input");
    Check(options.input_camera_device == options.input_video_path,
          "camera device and selected input path must match");
  }
  if (!options.input_video_path.empty()) {
    Check(std::filesystem::exists(options.input_video_path),
          "MLVC input video does not exist: " + options.input_video_path.string());
    return;
  }
  Check(std::filesystem::is_directory(options.input_frame_dir),
        "MLVC input frame directory does not exist: " + options.input_frame_dir.string());
  Check(std::filesystem::exists(options.input_frame_dir / "source_info.txt"),
        "MLVC frame directory requires source_info.txt: " + options.input_frame_dir.string());
}

mlvc::io::MlvcBitstreamHeader BuildEncodeHeader(const EncodeStreamOptions& options,
                                                const SourceFrameGeometry& geometry, double fps) {
  mlvc::io::MlvcBitstreamHeader header;
  header.version = 4;
  header.width = geometry.width;
  header.height = geometry.height;
  header.fps = fps;
  header.q_index = options.qp;
  header.gop = options.gop;
  header.reset_interval = options.reset_interval;
  header.translation_warp = options.translation_warp;
  header.ltr_start_idx = options.ltr_start_idx;
  header.ltr_period = options.ltr_period;
  header.ltr_qp_shift = options.ltr_qp_shift;
  header.target_bitrate_bps = options.target_bitrate_bps;
  header.flags = 0;
  header.forced_ltr_recovery_frame = options.forced_ltr_recovery_frame;
  header.forced_ltr_reference_frame = options.forced_ltr_reference_frame;
  mlvc::io::ValidateMlvcBitstreamHeader(header);
  return header;
}

}  // namespace mlvc::codec
