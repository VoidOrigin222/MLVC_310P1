#ifndef MLVC_ENTROPY_ENTROPY_CODEC_H_
#define MLVC_ENTROPY_ENTROPY_CODEC_H_

#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

#include "mlvc/entropy/rans_core.h"
#include "mlvc/entropy/sidecar.h"

namespace mlvc {

constexpr float kScaleMin = 0.11f;
constexpr float kScaleMax = 16.0f;
constexpr int kScaleLevel = 128;

std::vector<int16_t> BuildIndexesEncoder(const std::vector<int8_t>& symbols,
                                         const std::vector<float>& scales,
                                         std::optional<float> force_zero_thres);
void BuildIndexesEncoderInto(const std::vector<int8_t>& symbols, const std::vector<float>& scales,
                             std::optional<float> force_zero_thres, std::vector<int16_t>* encoded);
std::vector<uint8_t> BuildIndexesDecoder(const std::vector<float>& scales,
                                         std::optional<float> force_zero_thres,
                                         std::vector<uint8_t>* skip_condition);
void BuildIndexesDecoderInto(const std::vector<float>& scales,
                             std::optional<float> force_zero_thres, std::vector<uint8_t>* indexes,
                             std::vector<uint8_t>* skip_condition);
class EntropyEncoder {
 public:
  EntropyEncoder(const RuntimeSidecar& sidecar, const std::string& prefix, bool use_two_parts,
                 int z_channel_override = -1);

  void Reserve(std::size_t max_y_parts, std::size_t max_y_symbols_per_part,
               std::size_t max_encoded_stream_bytes);
  void Reset();
  void EncodeZ(std::vector<int8_t> symbols, int qp, int height, int width);
  void EncodeZView(const std::vector<int8_t>& symbols, int qp, int height, int width);
  void EncodeY(const std::vector<int8_t>& symbols, const std::vector<float>& scales,
               std::optional<float> force_zero_thres);
  void EncodeYIndexesView(const std::vector<int16_t>& indexes);
  std::vector<uint8_t> Flush();
  void FlushInto(std::vector<uint8_t>* output);

 private:
  RansEncoderCore encoder0_;
  RansEncoderCore encoder1_;
  bool use_two_parts_ = false;
  int z_channel_ = 0;
  int z_group_ = 0;
  int gaussian_group_ = 0;
  std::size_t next_y_scratch_slot_ = 0;
  std::deque<std::vector<int8_t>> split_z_scratch_;
  std::vector<std::vector<int16_t>> y_index_scratch_;
};

class EntropyDecoder {
 public:
  EntropyDecoder(const RuntimeSidecar& sidecar, const std::string& prefix, bool use_two_parts,
                 int z_channel_override = -1);

  void Reserve(std::size_t max_y_symbols, std::size_t max_stream_bytes);
  void SetStream(const std::vector<uint8_t>& stream);
  std::vector<int8_t> DecodeZ(int qp, int height, int width);
  void DecodeZInto(int qp, int height, int width, std::vector<int8_t>* output);
  std::vector<int8_t> DecodeZ(std::size_t total_size, int qp, int height, int width);
  void DecodeZInto(std::size_t total_size, int qp, int height, int width,
                   std::vector<int8_t>* output);
  std::vector<int8_t> DecodeY(const std::vector<float>& scales,
                              std::optional<float> force_zero_thres);
  void DecodeYInto(const std::vector<float>& scales, std::optional<float> force_zero_thres,
                   std::vector<int8_t>* output);
  std::vector<int8_t> DecodeYIndexes(const std::vector<uint8_t>& indexes);
  void DecodeYIndexesInto(const std::vector<uint8_t>& indexes, std::vector<int8_t>* output);

 private:
  RansDecoderCore decoder0_;
  RansDecoderCore decoder1_;
  bool use_two_parts_ = false;
  int z_channel_ = 0;
  int z_group_ = 0;
  int gaussian_group_ = 0;
  std::vector<uint8_t> index_scratch_;
  std::vector<uint8_t> skip_scratch_;
  std::vector<int8_t> compact_decode_scratch_;
  std::vector<uint8_t> reversed_stream_scratch_;
};

}  // namespace mlvc

#endif  // MLVC_ENTROPY_ENTROPY_CODEC_H_
