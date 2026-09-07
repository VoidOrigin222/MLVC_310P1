#ifndef MLVC_IO_BITSTREAM_IO_H_
#define MLVC_IO_BITSTREAM_IO_H_

#include <cstdint>
#include <filesystem>
#include <ios>
#include <iosfwd>
#include <optional>
#include <string>

namespace mlvc::io {

struct BitstreamSummary {
  int frames = 0;
  uint64_t bytes = 0;
  int width = 0;
  int height = 0;
};

class BitstreamInput {
 public:
  virtual ~BitstreamInput() = default;

  virtual std::istream& stream() = 0;
  virtual bool AtEnd() = 0;
  virtual std::optional<std::streampos> Tell() = 0;
  virtual std::optional<uint64_t> TotalBytes() const = 0;
  virtual std::string Description() const = 0;
};

class FileBitstreamInput final : public BitstreamInput {
 public:
  explicit FileBitstreamInput(std::filesystem::path path);
  ~FileBitstreamInput() override;

  std::istream& stream() override;
  bool AtEnd() override;
  std::optional<std::streampos> Tell() override;
  std::optional<uint64_t> TotalBytes() const override;
  std::string Description() const override;

 private:
  std::filesystem::path path_;
  std::optional<uint64_t> total_bytes_;
  class Impl;
  Impl* impl_ = nullptr;
};

std::optional<BitstreamSummary> ScanBitstreamSummary(std::istream& input, int frame_num,
                                                     std::optional<uint64_t> total_bytes);
std::optional<BitstreamSummary> ScanBitstreamFile(const std::filesystem::path& input_path,
                                                  int frame_num);
uint64_t StreamByteDistance(std::optional<std::streampos> begin, std::optional<std::streampos> end,
                            uint64_t fallback);
void WriteBytesFile(const std::string& bytes, const std::filesystem::path& path);

}  // namespace mlvc::io

#endif  // MLVC_IO_BITSTREAM_IO_H_
