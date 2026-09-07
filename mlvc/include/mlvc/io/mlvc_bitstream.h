#ifndef MLVC_IO_MLVC_BITSTREAM_H_
#define MLVC_IO_MLVC_BITSTREAM_H_

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "mlvc/codec/mlvc_entropy.h"

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
};

constexpr std::size_t kMlvcBitstreamFrameOverheadBytes = 17;
constexpr std::size_t kOfficialMlvcFrameOverheadBytes = 8;

class OfficialMlvcBitstreamWriter {
 public:
  explicit OfficialMlvcBitstreamWriter(const std::filesystem::path& path);
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
  void Close();

 private:
  std::filesystem::path path_;
  std::ofstream output_;
  bool closed_ = false;
};

class MlvcBitstreamReader {
 public:
  explicit MlvcBitstreamReader(const std::filesystem::path& path);
  MlvcBitstreamReader(const MlvcBitstreamReader&) = delete;
  MlvcBitstreamReader& operator=(const MlvcBitstreamReader&) = delete;
  ~MlvcBitstreamReader();

  const MlvcBitstreamHeader& header() const { return header_; }
  bool ReadFrame(int* frame_index, mlvc::codec::MlvcFrameType* frame_type, int* q_index,
                 std::vector<uint8_t>* payload);

 private:
  std::filesystem::path path_;
  std::ifstream input_;
  MlvcBitstreamHeader header_;
  uint32_t version_ = 0;
};

}  // namespace mlvc::io

#endif  // MLVC_IO_MLVC_BITSTREAM_H_
