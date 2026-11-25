#ifndef ULBVC_IO_ULBVC_BITSTREAM_H_
#define ULBVC_IO_ULBVC_BITSTREAM_H_

#include <cstdint>
#include <array>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "ulbvc/codec/ulbvc_entropy.h"
#include "ulbvc/transport/ulbvc_media_unit.h"

namespace ulbvc::io {

struct UlbvcBitstreamHeader {
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
  // headers that predate ULBVC-ES v1.
  int coded_width = 0;
  int coded_height = 0;
  std::array<uint8_t, 32> codec_bundle_sha256{};
  bool translation_warp = false;
};

struct ForcedLtrFrames {
  int recovery_frame = -1;
  int reference_frame = -1;
};

struct UlbvcFrameMetadata {
  bool explicit_metadata = false;
  int model_q_index = -1;
  uint32_t unit_flags = 0;
  uint32_t short_ref_frame_id = ulbvc::transport::kUlbvcNoReference;
  uint32_t long_ref_frame_id = ulbvc::transport::kUlbvcNoReference;
  int64_t pts = 0;
  bool translation_warp = false;
  int8_t kx = 0;
  int8_t ky = 0;
  std::vector<ulbvc::transport::UlbvcTlv> tlvs;
};

void ValidateUlbvcBitstreamHeader(const UlbvcBitstreamHeader& header);
void ValidateUlbvcDecoderOutputShape(const UlbvcBitstreamHeader& header,
                                    const std::vector<int64_t>& output_shape);
void ValidateUlbvcQIndexForSidecar(int q_index, int supported_q_index_count);
ForcedLtrFrames ResolveForcedLtrFrames(const UlbvcBitstreamHeader& header,
                                       int requested_recovery_frame,
                                       int requested_reference_frame);
uint64_t MaxUlbvcFramePayloadBytes(int width, int height);
void ValidateUlbvcFrameMetadata(int frame_index, ulbvc::codec::UlbvcFrameType frame_type, int q_index,
                               int expected_frame_index, bool require_i_frame);

constexpr std::size_t kUlbvcBitstreamFrameOverheadBytes = 17;
constexpr std::size_t kOfficialUlbvcFrameOverheadBytes = 8;

class OfficialUlbvcBitstreamWriter {
 public:
  explicit OfficialUlbvcBitstreamWriter(const std::filesystem::path& path);
  OfficialUlbvcBitstreamWriter(const OfficialUlbvcBitstreamWriter&) = delete;
  OfficialUlbvcBitstreamWriter& operator=(const OfficialUlbvcBitstreamWriter&) = delete;
  ~OfficialUlbvcBitstreamWriter();

  void WriteFrame(int q_index, const std::vector<uint8_t>& payload);
  void Close();

 private:
  std::filesystem::path path_;
  std::ofstream output_;
  bool closed_ = false;
};

class OfficialUlbvcBitstreamReader {
 public:
  explicit OfficialUlbvcBitstreamReader(const std::filesystem::path& path);
  OfficialUlbvcBitstreamReader(const OfficialUlbvcBitstreamReader&) = delete;
  OfficialUlbvcBitstreamReader& operator=(const OfficialUlbvcBitstreamReader&) = delete;
  ~OfficialUlbvcBitstreamReader() = default;

  bool ReadFrame(int* frame_index, int* q_index, std::vector<uint8_t>* payload);

 private:
  std::filesystem::path path_;
  std::ifstream input_;
  int next_frame_index_ = 0;
};

class UlbvcBitstreamWriter {
 public:
  UlbvcBitstreamWriter(const std::filesystem::path& path, const UlbvcBitstreamHeader& header);
  UlbvcBitstreamWriter(const UlbvcBitstreamWriter&) = delete;
  UlbvcBitstreamWriter& operator=(const UlbvcBitstreamWriter&) = delete;
  ~UlbvcBitstreamWriter();

  void WriteFrame(int frame_index, ulbvc::codec::UlbvcFrameType frame_type, int q_index,
                  const std::vector<uint8_t>& payload);
  void WriteFrame(int frame_index, ulbvc::codec::UlbvcFrameType frame_type, int q_index,
                  const UlbvcFrameMetadata& metadata, const std::vector<uint8_t>& payload);
  void SwitchConfiguration(uint32_t config_id, const UlbvcBitstreamHeader& header);
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

class UlbvcBitstreamReader {
 public:
  explicit UlbvcBitstreamReader(const std::filesystem::path& path);
  UlbvcBitstreamReader(const UlbvcBitstreamReader&) = delete;
  UlbvcBitstreamReader& operator=(const UlbvcBitstreamReader&) = delete;
  ~UlbvcBitstreamReader();

  const UlbvcBitstreamHeader& header() const { return header_; }
  const UlbvcFrameMetadata& last_frame_metadata() const { return last_frame_metadata_; }
  bool ReadFrame(int* frame_index, ulbvc::codec::UlbvcFrameType* frame_type, int* q_index,
                 std::vector<uint8_t>* payload);

 private:
  std::filesystem::path path_;
  std::ifstream input_;
  UlbvcBitstreamHeader header_;
  uint32_t version_ = 0;
  int expected_frame_index_ = 0;
  uint64_t max_payload_size_ = 0;
  bool media_unit_format_ = false;
  uint32_t config_id_ = 0;
  std::vector<uint8_t> active_config_unit_;
  uint32_t pending_config_id_ = 0;
  uint32_t last_ltr_frame_id_ = ulbvc::transport::kUlbvcNoReference;
  UlbvcFrameMetadata last_frame_metadata_;
};

}  // namespace ulbvc::io

#endif  // ULBVC_IO_ULBVC_BITSTREAM_H_
