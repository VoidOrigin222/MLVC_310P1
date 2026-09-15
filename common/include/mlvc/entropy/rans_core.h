#ifndef MLVC_ENTROPY_RANS_CORE_H_
#define MLVC_ENTROPY_RANS_CORE_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "mlvc/entropy/rans_byte.h"

namespace mlvc {

struct RansSymbol {
  uint16_t start = 0;
  uint16_t range = 0;
};

struct CdfGroup {
  std::vector<std::vector<int32_t>> cdfs;
  std::vector<int32_t> cdf_lengths;
  std::vector<int32_t> offsets;
  std::vector<std::vector<RansSymbol>> symbols;
};

class RansEncoderCore {
 public:
  int AddCdf(std::shared_ptr<CdfGroup> group);
  void ReservePending(std::size_t capacity);
  void ReserveOutputBytes(std::size_t capacity);
  void Reset();
  void EncodeY(std::vector<int16_t> symbols, int cdf_group_index);
  void EncodeY(const std::vector<int16_t>* symbols, std::size_t begin, std::size_t count,
               int cdf_group_index);
  void EncodeZ(std::vector<int8_t> symbols, int cdf_group_index, int start_offset,
               int per_channel_size);
  void EncodeZ(const std::vector<int8_t>* symbols, std::size_t begin, std::size_t count,
               int cdf_group_index, int start_offset, int per_channel_size);
  void Flush();
  const std::vector<uint8_t>& stream() const { return stream_; }

 private:
  struct PendingTask {
    enum class Type { kY, kZ };
    Type type = Type::kY;
    std::vector<int16_t> symbols_y;
    std::vector<int8_t> symbols_z;
    const std::vector<int16_t>* borrowed_symbols_y = nullptr;
    const std::vector<int8_t>* borrowed_symbols_z = nullptr;
    std::size_t symbol_begin = 0;
    std::size_t symbol_count = 0;
    int cdf_group_index = 0;
    int start_offset = 0;
    int per_channel_size = 0;
  };

  static std::size_t SymbolCount(const PendingTask& task);
  static int16_t YSymbolAt(const PendingTask& task, std::size_t index);
  static int8_t ZSymbolAt(const PendingTask& task, std::size_t index);
  void EncodeOneSymbol(uint8_t*& ptr, RansState& state, int32_t symbol, int32_t cdf_size,
                       int32_t offset, const std::vector<RansSymbol>& symbols);
  void EncodeYInternal(uint8_t*& ptr, RansState& state, const PendingTask& task);
  void EncodeZInternal(uint8_t*& ptr, RansState& state, const PendingTask& task);

  std::vector<std::shared_ptr<CdfGroup>> cdf_groups_;
  std::vector<PendingTask> pending_;
  std::vector<uint8_t> stream_;
  std::vector<uint8_t> output_scratch_;
};

class RansDecoderCore {
 public:
  int AddCdf(std::shared_ptr<CdfGroup> group);
  void ReserveStream(std::size_t capacity);
  void SetStream(std::vector<uint8_t> stream);
  void SetStreamView(const std::vector<uint8_t>& stream);
  std::vector<int8_t> DecodeY(const std::vector<uint8_t>& indexes, int cdf_group_index);
  void DecodeYInto(const std::vector<uint8_t>& indexes, std::size_t begin, std::size_t count,
                   int cdf_group_index, std::vector<int8_t>* output, std::size_t output_begin);
  std::vector<int8_t> DecodeZ(int total_size, int cdf_group_index, int start_offset,
                              int per_channel_size);
  void DecodeZInto(int total_size, int cdf_group_index, int start_offset, int per_channel_size,
                   std::vector<int8_t>* output, std::size_t output_begin);

 private:
  int8_t DecodeOneSymbol(const std::vector<int32_t>& cdf, int32_t cdf_size, int32_t offset);

  std::vector<std::shared_ptr<CdfGroup>> cdf_groups_;
  std::vector<uint8_t> stream_;
  RansState state_ = 0;
  uint8_t* ptr_ = nullptr;
};

std::shared_ptr<CdfGroup> MakeCdfGroup(const std::vector<int32_t>& cdfs,
                                       const std::vector<int64_t>& cdf_shape,
                                       const std::vector<int32_t>& cdf_lengths,
                                       const std::vector<int32_t>& offsets);

}  // namespace mlvc

#endif  // MLVC_ENTROPY_RANS_CORE_H_
