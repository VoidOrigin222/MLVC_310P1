#include <mlvc/io/bitstream_io.h>

#include <algorithm>
#include <fstream>
#include <istream>

#include "mlvc/bitstream/bitstream.h"
#include "mlvc/core/status.h"

namespace mlvc::io {

class FileBitstreamInput::Impl {
 public:
  explicit Impl(const std::filesystem::path& path) : input(path, std::ios::binary) {}

  std::ifstream input;
};

FileBitstreamInput::FileBitstreamInput(std::filesystem::path path) : path_(std::move(path)) {
  {
    std::ifstream input(path_, std::ios::binary | std::ios::ate);
    mlvc::Check(input.good(), "failed to open input bitstream: " + path_.string());
    total_bytes_ = static_cast<uint64_t>(input.tellg());
  }
  impl_ = new Impl(path_);
  mlvc::Check(impl_->input.good(), "failed to open input bitstream: " + path_.string());
}

FileBitstreamInput::~FileBitstreamInput() { delete impl_; }

std::istream& FileBitstreamInput::stream() { return impl_->input; }

bool FileBitstreamInput::AtEnd() { return impl_->input.peek() == std::char_traits<char>::eof(); }

std::optional<std::streampos> FileBitstreamInput::Tell() {
  const std::streampos position = impl_->input.tellg();
  if (position == std::streampos(-1)) {
    return std::nullopt;
  }
  return position;
}

std::optional<uint64_t> FileBitstreamInput::TotalBytes() const { return total_bytes_; }

std::string FileBitstreamInput::Description() const { return path_.string(); }

std::optional<BitstreamSummary> ScanBitstreamSummary(std::istream& input, int frame_num,
                                                     std::optional<uint64_t> total_bytes) {
  mlvc::SpsHelper sps_helper;
  BitstreamSummary summary;
  while (input.peek() != std::char_traits<char>::eof()) {
    if (frame_num > 0 && summary.frames >= frame_num) {
      break;
    }
    const std::streampos unit_start = input.tellg();
    mlvc::NalHeader header = mlvc::ReadHeader(input);
    while (header.nal_type == mlvc::NalType::kSps) {
      const mlvc::Sps sps = mlvc::ReadSpsRemaining(input, header.sps_id);
      sps_helper.Add(sps);
      if (summary.width == 0 || summary.height == 0) {
        summary.width = sps.width;
        summary.height = sps.height;
      }
      if (input.peek() == std::char_traits<char>::eof()) {
        break;
      }
      header = mlvc::ReadHeader(input);
    }
    if (header.nal_type == mlvc::NalType::kSps) {
      if (total_bytes.has_value()) {
        summary.bytes = *total_bytes;
      }
      break;
    }

    const auto sps = sps_helper.Get(header.sps_id);
    mlvc::Check(sps.has_value(), "frame NAL references missing SPS");
    summary.width = sps->width;
    summary.height = sps->height;
    (void)mlvc::ReadIpRemaining(input);
    const std::streampos unit_end = input.tellg();
    summary.bytes += static_cast<uint64_t>(std::max<std::streamoff>(unit_end - unit_start, 0));
    ++summary.frames;
  }
  if (summary.frames == 0) {
    return std::nullopt;
  }
  return summary;
}

std::optional<BitstreamSummary> ScanBitstreamFile(const std::filesystem::path& input_path,
                                                  int frame_num) {
  std::ifstream input(input_path, std::ios::binary | std::ios::ate);
  if (!input.good()) {
    return std::nullopt;
  }
  const uint64_t total_bytes = static_cast<uint64_t>(input.tellg());
  input.seekg(0, std::ios::beg);
  return ScanBitstreamSummary(input, frame_num, total_bytes);
}

uint64_t StreamByteDistance(std::optional<std::streampos> begin, std::optional<std::streampos> end,
                            uint64_t fallback) {
  if (!begin.has_value() || !end.has_value()) {
    return fallback;
  }
  return static_cast<uint64_t>(std::max<std::streamoff>(*end - *begin, 0));
}

void WriteBytesFile(const std::string& bytes, const std::filesystem::path& path) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  mlvc::Check(output.good(), "failed to open byte output: " + path.string());
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  mlvc::Check(output.good(), "failed to write byte output: " + path.string());
}

}  // namespace mlvc::io
