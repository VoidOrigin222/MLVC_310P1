#ifndef ULBVC_ENTROPY_ULBVC_OFFICIAL_ENTROPY_H_
#define ULBVC_ENTROPY_ULBVC_OFFICIAL_ENTROPY_H_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include "ulbvc/runtime/model_manifest.h"

namespace ulbvc {

class Profiler;

std::vector<int8_t> NarrowUlbvcSymbols(const std::vector<int32_t>& values);

class UlbvcOfficialEntropyEncoder {
 public:
  explicit UlbvcOfficialEntropyEncoder(const ModelManifest& manifest);
  ~UlbvcOfficialEntropyEncoder();
  UlbvcOfficialEntropyEncoder(const UlbvcOfficialEntropyEncoder&) = delete;
  UlbvcOfficialEntropyEncoder& operator=(const UlbvcOfficialEntropyEncoder&) = delete;

  void Encode(const std::vector<int8_t>& z, const std::vector<int8_t>& y0,
              const std::vector<int8_t>& y1, const std::vector<uint8_t>& scales0,
              const std::vector<uint8_t>& scales1, int q_index, int z_channels, int z_height,
              int z_width, std::vector<uint8_t>* payload) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class UlbvcOfficialEntropyDecoder {
 public:
  explicit UlbvcOfficialEntropyDecoder(const ModelManifest& manifest);
  ~UlbvcOfficialEntropyDecoder();
  UlbvcOfficialEntropyDecoder(const UlbvcOfficialEntropyDecoder&) = delete;
  UlbvcOfficialEntropyDecoder& operator=(const UlbvcOfficialEntropyDecoder&) = delete;

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

}  // namespace ulbvc

#endif  // ULBVC_ENTROPY_ULBVC_OFFICIAL_ENTROPY_H_
