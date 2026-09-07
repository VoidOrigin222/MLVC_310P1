#include <mlvc/io/mlvc_bitstream.h>

#include <array>
#include <cstring>
#include <limits>
#include <vector>

#include "mlvc/core/status.h"

namespace mlvc::io {
namespace {

constexpr std::array<char, 8> kMagic = {'M', 'L', 'V', 'C', 'B', 'S', 'T', '1'};
constexpr uint32_t kVersion = 3;

template <typename T>
void WritePod(std::ostream& out, const T& value, const char* what) {
  out.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(T)));
  Check(out.good(), std::string("failed to write mlvc bitstream ") + what);
}

template <typename T>
T ReadPod(std::istream& in, const char* what) {
  T value{};
  in.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
  Check(in.good(), std::string("failed to read mlvc bitstream ") + what);
  return value;
}

}  // namespace

OfficialMlvcBitstreamWriter::OfficialMlvcBitstreamWriter(const std::filesystem::path& path)
    : path_(path) {
  if (!path_.parent_path().empty()) {
    std::filesystem::create_directories(path_.parent_path());
  }
  output_.open(path_, std::ios::binary | std::ios::trunc);
  Check(output_.good(), "failed to open official MLVC bitstream output: " + path_.string());
}

OfficialMlvcBitstreamWriter::~OfficialMlvcBitstreamWriter() { Close(); }

void OfficialMlvcBitstreamWriter::WriteFrame(int q_index, const std::vector<uint8_t>& payload) {
  Check(!closed_, "write on closed official MLVC bitstream");
  Check(payload.size() <= static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()),
        "official MLVC frame payload is too large");
  WritePod(output_, static_cast<int32_t>(q_index), "q_index");
  WritePod(output_, static_cast<uint32_t>(payload.size()), "payload_size");
  output_.write(reinterpret_cast<const char*>(payload.data()),
                static_cast<std::streamsize>(payload.size()));
  Check(output_.good(), "failed to write official MLVC payload");
}

void OfficialMlvcBitstreamWriter::Close() {
  if (closed_) {
    return;
  }
  if (output_.is_open()) {
    output_.flush();
    Check(output_.good(), "failed to flush official MLVC bitstream: " + path_.string());
    output_.close();
  }
  closed_ = true;
}

OfficialMlvcBitstreamReader::OfficialMlvcBitstreamReader(const std::filesystem::path& path)
    : path_(path) {
  input_.open(path_, std::ios::binary);
  Check(input_.good(), "failed to open official MLVC bitstream input: " + path_.string());
}

bool OfficialMlvcBitstreamReader::ReadFrame(int* frame_index, int* q_index,
                                            std::vector<uint8_t>* payload) {
  Check(frame_index != nullptr && q_index != nullptr && payload != nullptr,
        "official MLVC frame outputs are required");
  if (input_.peek() == std::char_traits<char>::eof()) {
    return false;
  }
  *frame_index = next_frame_index_++;
  *q_index = static_cast<int>(ReadPod<int32_t>(input_, "q_index"));
  const uint32_t bytes = ReadPod<uint32_t>(input_, "payload_size");
  payload->resize(bytes);
  input_.read(reinterpret_cast<char*>(payload->data()), static_cast<std::streamsize>(bytes));
  Check(input_.good(), "failed to read official MLVC payload");
  return true;
}

MlvcBitstreamWriter::MlvcBitstreamWriter(const std::filesystem::path& path,
                                         const MlvcBitstreamHeader& header)
    : path_(path) {
  if (!path_.parent_path().empty()) {
    std::filesystem::create_directories(path_.parent_path());
  }
  output_.open(path_, std::ios::binary | std::ios::trunc);
  Check(output_.good(), "failed to open mlvc bitstream output: " + path_.string());
  output_.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
  Check(output_.good(), "failed to write mlvc bitstream magic");
  WritePod(output_, static_cast<uint32_t>(header.version == 0 ? kVersion : header.version),
           "version");
  WritePod(output_, static_cast<int32_t>(header.width), "width");
  WritePod(output_, static_cast<int32_t>(header.height), "height");
  WritePod(output_, header.fps, "fps");
  WritePod(output_, static_cast<int32_t>(header.q_index), "q_index");
  WritePod(output_, static_cast<int32_t>(header.gop), "gop");
  WritePod(output_, static_cast<int32_t>(header.reset_interval), "reset_interval");
  WritePod(output_, static_cast<int32_t>(header.ltr_start_idx), "ltr_start_idx");
  WritePod(output_, static_cast<int32_t>(header.ltr_period), "ltr_period");
  WritePod(output_, static_cast<int32_t>(header.ltr_qp_shift), "ltr_qp_shift");
  WritePod(output_, header.target_bitrate_bps, "target_bitrate_bps");
  WritePod(output_, static_cast<uint32_t>(header.flags), "flags");
}

MlvcBitstreamWriter::~MlvcBitstreamWriter() { Close(); }

void MlvcBitstreamWriter::WriteFrame(int frame_index, mlvc::codec::MlvcFrameType frame_type,
                                     int q_index, const std::vector<uint8_t>& payload) {
  Check(!closed_, "write on closed mlvc bitstream");
  WritePod(output_, static_cast<int32_t>(frame_index), "frame_index");
  WritePod(output_, static_cast<uint8_t>(frame_type), "frame_type");
  WritePod(output_, static_cast<int32_t>(q_index), "q_index");
  const uint64_t bytes = static_cast<uint64_t>(payload.size());
  WritePod(output_, bytes, "payload_size");
  output_.write(reinterpret_cast<const char*>(payload.data()),
                static_cast<std::streamsize>(payload.size()));
  Check(output_.good(), "failed to write mlvc bitstream payload");
}

void MlvcBitstreamWriter::Close() {
  if (closed_) {
    return;
  }
  if (output_.is_open()) {
    output_.flush();
    Check(output_.good(), "failed to flush mlvc bitstream output: " + path_.string());
    output_.close();
  }
  closed_ = true;
}

MlvcBitstreamReader::MlvcBitstreamReader(const std::filesystem::path& path) : path_(path) {
  input_.open(path_, std::ios::binary);
  Check(input_.good(), "failed to open mlvc bitstream input: " + path_.string());
  std::array<char, 8> magic{};
  input_.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  Check(input_.good(), "failed to read mlvc bitstream magic: " + path_.string());
  Check(magic == kMagic, "invalid mlvc bitstream magic: " + path_.string());
  const uint32_t version = ReadPod<uint32_t>(input_, "version");
  Check(version >= 2 && version <= kVersion, "unsupported mlvc bitstream version");
  version_ = version;
  header_.version = version_;
  header_.width = static_cast<int>(ReadPod<int32_t>(input_, "width"));
  header_.height = static_cast<int>(ReadPod<int32_t>(input_, "height"));
  header_.fps = ReadPod<double>(input_, "fps");
  header_.q_index = static_cast<int>(ReadPod<int32_t>(input_, "q_index"));
  if (version_ >= 3) {
    header_.gop = static_cast<int>(ReadPod<int32_t>(input_, "gop"));
    header_.reset_interval = static_cast<int>(ReadPod<int32_t>(input_, "reset_interval"));
    header_.ltr_start_idx = static_cast<int>(ReadPod<int32_t>(input_, "ltr_start_idx"));
    header_.ltr_period = static_cast<int>(ReadPod<int32_t>(input_, "ltr_period"));
    header_.ltr_qp_shift = static_cast<int>(ReadPod<int32_t>(input_, "ltr_qp_shift"));
    header_.target_bitrate_bps = ReadPod<double>(input_, "target_bitrate_bps");
    header_.flags = ReadPod<uint32_t>(input_, "flags");
  }
}

MlvcBitstreamReader::~MlvcBitstreamReader() = default;

bool MlvcBitstreamReader::ReadFrame(int* frame_index, mlvc::codec::MlvcFrameType* frame_type,
                                    int* q_index, std::vector<uint8_t>* payload) {
  Check(frame_index != nullptr, "mlvc bitstream frame index output is required");
  Check(frame_type != nullptr && q_index != nullptr && payload != nullptr,
        "mlvc bitstream frame outputs are required");
  if (input_.peek() == std::char_traits<char>::eof()) {
    return false;
  }
  *frame_index = static_cast<int>(ReadPod<int32_t>(input_, "frame_index"));
  const uint8_t type_byte = ReadPod<uint8_t>(input_, "frame_type");
  if (version_ >= 3) {
    Check(type_byte <= static_cast<uint8_t>(mlvc::codec::MlvcFrameType::kLtrRecovery),
          "invalid mlvc frame type");
    *frame_type = static_cast<mlvc::codec::MlvcFrameType>(type_byte);
  } else {
    *frame_type =
        type_byte != 0 ? mlvc::codec::MlvcFrameType::kIFrame : mlvc::codec::MlvcFrameType::kPFrame;
  }
  *q_index = static_cast<int>(ReadPod<int32_t>(input_, "q_index"));
  const uint64_t bytes = ReadPod<uint64_t>(input_, "payload_size");
  payload->resize(static_cast<std::size_t>(bytes));
  input_.read(reinterpret_cast<char*>(payload->data()), static_cast<std::streamsize>(bytes));
  Check(input_.good(), "failed to read mlvc bitstream payload");
  return true;
}

}  // namespace mlvc::io
