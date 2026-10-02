#ifndef MLVC_IO_MLVC_BITSTREAM_H_
#define MLVC_IO_MLVC_BITSTREAM_H_

#include <cstdint>
#include <array>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "mlvc/codec/mlvc_entropy.h"
#include "mlvc/transport/mlvc_media_unit.h"

namespace mlvc::io {

struct MlvcBitstreamHeader {
  uint32_t version = 0;
  int width = 0;
  int height = 0;
  double fps = 30.0;
  int q_index = 21;
  int gop = 0;
  int reset_interval = 0;
  int ltr_start_idx = 0;
  int ltr_period = 0;
  int ltr_qp_shift = 0;
  double target_bitrate_bps = 0.0;
  uint32_t flags = 0;
  int forced_ltr_recovery_frame = -1;
  int forced_ltr_reference_frame = -1;
  // The visible dimensions describe the source/output crop.  The coded
  // dimensions are the static model tensor shape (for example 1920x1088 for
  // a 1920x1080 stream).  A zero value means "same as visible" for legacy
  // headers that predate MLVC-ES v1.
  int coded_width = 0;
  int coded_height = 0;
  std::array<uint8_t, 32> codec_bundle_sha256{};
  bool translation_warp = false;
};

struct ForcedLtrFrames {
  int recovery_frame = -1;
  int reference_frame = -1;
};

struct MlvcFrameMetadata {
  bool explicit_metadata = false;
  int model_q_index = -1;
  uint32_t unit_flags = 0;
  uint32_t short_ref_frame_id = mlvc::transport::kMlvcNoReference;
  uint32_t long_ref_frame_id = mlvc::transport::kMlvcNoReference;
  int64_t pts = 0;
  bool translation_warp = false;
  int8_t kx = 0;
  int8_t ky = 0;
};

void ValidateMlvcBitstreamHeader(const MlvcBitstreamHeader& header);
void ValidateMlvcDecoderOutputShape(const MlvcBitstreamHeader& header,
                                    const std::vector<int64_t>& output_shape);
void ValidateMlvcQIndexForSidecar(int q_index, int supported_q_index_count);
ForcedLtrFrames ResolveForcedLtrFrames(const MlvcBitstreamHeader& header,
                                       int requested_recovery_frame,
                                       int requested_reference_frame);
uint64_t MaxMlvcFramePayloadBytes(int width, int height);
void ValidateMlvcFrameMetadata(int frame_index, mlvc::codec::MlvcFrameType frame_type, int q_index,
                               int expected_frame_index, bool require_i_frame);

constexpr std::size_t kMlvcBitstreamFrameOverheadBytes = 17;
constexpr std::size_t kOfficialMlvcFrameOverheadBytes = 8;

class OfficialMlvcBitstreamWriter {
 public:
  explicit OfficialMlvcBitstreamWriter(const std::filesystem::path& path);
  OfficialMlvcBitstreamWriter(const OfficialMlvcBitstreamWriter&) = delete;
  OfficialMlvcBitstreamWriter& operator=(const OfficialMlvcBitstreamWriter&) = delete;
  ~OfficialMlvcBitstreamWriter();

  void WriteFrame(int q_index, const std::vector<uint8_t>& payload);
  void Close();

 private:
  std::filesystem::path path_;
  std::ofstream output_;
  bool closed_ = false;
};

class OfficialMlvcBitstreamReader {
 public:
  explicit OfficialMlvcBitstreamReader(const std::filesystem::path& path);
  OfficialMlvcBitstreamReader(const OfficialMlvcBitstreamReader&) = delete;
  OfficialMlvcBitstreamReader& operator=(const OfficialMlvcBitstreamReader&) = delete;
  ~OfficialMlvcBitstreamReader() = default;

  bool ReadFrame(int* frame_index, int* q_index, std::vector<uint8_t>* payload);

 private:
  std::filesystem::path path_;
  std::ifstream input_;
  int next_frame_index_ = 0;
};

class MlvcBitstreamWriter {
 public:
  MlvcBitstreamWriter(const std::filesystem::path& path, const MlvcBitstreamHeader& header);
  MlvcBitstreamWriter(const MlvcBitstreamWriter&) = delete;
  MlvcBitstreamWriter& operator=(const MlvcBitstreamWriter&) = delete;
  ~MlvcBitstreamWriter();

  void WriteFrame(int frame_index, mlvc::codec::MlvcFrameType frame_type, int q_index,
                  const std::vector<uint8_t>& payload);
  void WriteFrame(int frame_index, mlvc::codec::MlvcFrameType frame_type, int q_index,
                  const MlvcFrameMetadata& metadata, const std::vector<uint8_t>& payload);
  void SwitchConfiguration(uint32_t config_id, const MlvcBitstreamHeader& header);
  void Close();

 private:
  std::filesystem::path path_;
  std::ofstream output_;
  bool closed_ = false;
  uint64_t max_payload_size_ = 0;
  double fps_ = 30.0;
  int next_frame_index_ = 0;
  uint32_t config_id_ = 1;
  bool configuration_switch_pending_ = false;
  std::vector<uint8_t> active_config_unit_;
};

class MlvcBitstreamReader {
 public:
  explicit MlvcBitstreamReader(const std::filesystem::path& path);
  MlvcBitstreamReader(const MlvcBitstreamReader&) = delete;
  MlvcBitstreamReader& operator=(const MlvcBitstreamReader&) = delete;
  ~MlvcBitstreamReader();

  const MlvcBitstreamHeader& header() const { return header_; }
  const MlvcFrameMetadata& last_frame_metadata() const { return last_frame_metadata_; }
  bool ReadFrame(int* frame_index, mlvc::codec::MlvcFrameType* frame_type, int* q_index,
                 std::vector<uint8_t>* payload);

 private:
  std::filesystem::path path_;
  std::ifstream input_;
  MlvcBitstreamHeader header_;
  uint32_t version_ = 0;
  int expected_frame_index_ = 0;
  uint64_t max_payload_size_ = 0;
  bool media_unit_format_ = false;
  uint32_t config_id_ = 0;
  std::vector<uint8_t> active_config_unit_;
  uint32_t pending_config_id_ = 0;
  uint32_t last_ltr_frame_id_ = mlvc::transport::kMlvcNoReference;
  MlvcFrameMetadata last_frame_metadata_;
};

}  // namespace mlvc::io

#endif  // MLVC_IO_MLVC_BITSTREAM_H_
