#include <mlvc/application/stream/encode/encode_setup.h>

#include <filesystem>

#include "mlvc/core/status.h"

namespace mlvc::codec {

void ValidateEncodeInput(const EncodeStreamOptions& options) {
  Check(!options.manifest_path.empty(), "encode stream requires a manifest path");
  Check(!options.input_video_path.empty() || !options.input_frame_dir.empty(),
        "MLVC encode requires input_frame_dir or input_video; synthetic frames are disabled");
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
                                                const SourceFrameGeometry& geometry,
                                                double fps) {
  mlvc::io::MlvcBitstreamHeader header;
  header.version = 3;
  header.width = geometry.width;
  header.height = geometry.height;
  header.fps = fps;
  header.q_index = options.qp;
  header.gop = options.gop;
  header.reset_interval = options.reset_interval;
  header.ltr_start_idx = options.ltr_start_idx;
  header.ltr_period = options.ltr_period;
  header.ltr_qp_shift = options.ltr_qp_shift;
  header.target_bitrate_bps = options.target_bitrate_bps;
  header.flags = 0;
  return header;
}

}  // namespace mlvc::codec
