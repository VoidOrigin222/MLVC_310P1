#ifndef MLVC_ENTROPY_MLVC_OFFICIAL_ENTROPY_H_
#define MLVC_ENTROPY_MLVC_OFFICIAL_ENTROPY_H_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace mlvc {

class Profiler;

std::vector<int8_t> NarrowMlvcSymbols(const std::vector<int32_t>& values);

class MlvcOfficialEntropyEncoder {
 public:
  explicit MlvcOfficialEntropyEncoder(const std::filesystem::path& model_directory);
  ~MlvcOfficialEntropyEncoder();
  MlvcOfficialEntropyEncoder(const MlvcOfficialEntropyEncoder&) = delete;
  MlvcOfficialEntropyEncoder& operator=(const MlvcOfficialEntropyEncoder&) = delete;

  void Encode(const std::vector<int8_t>& z, const std::vector<int8_t>& y0,
              const std::vector<int8_t>& y1, const std::vector<uint8_t>& scales0,
              const std::vector<uint8_t>& scales1, int q_index, int z_channels, int z_height,
              int z_width, std::vector<uint8_t>* payload) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class MlvcOfficialEntropyDecoder {
 public:
  explicit MlvcOfficialEntropyDecoder(const std::filesystem::path& model_directory);
  ~MlvcOfficialEntropyDecoder();
  MlvcOfficialEntropyDecoder(const MlvcOfficialEntropyDecoder&) = delete;
  MlvcOfficialEntropyDecoder& operator=(const MlvcOfficialEntropyDecoder&) = delete;

  void SetStream(const std::vector<uint8_t>& payload, Profiler* profiler = nullptr);
  std::vector<int32_t> DecodeZInt32(int q_index, int z_channels, int z_height, int z_width,
                                    Profiler* profiler = nullptr);
  std::vector<int8_t> DecodeZ(int q_index, int z_channels, int z_height, int z_width,
                              Profiler* profiler = nullptr);
  std::vector<int32_t> DecodeYInt32(const std::vector<uint8_t>& scales, bool eof,
                                    Profiler* profiler = nullptr);
  std::vector<int8_t> DecodeY(const std::vector<uint8_t>& scales, bool eof,
                              Profiler* profiler = nullptr);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mlvc

#endif  // MLVC_ENTROPY_MLVC_OFFICIAL_ENTROPY_H_
