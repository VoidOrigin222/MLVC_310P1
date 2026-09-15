#include "mlvc/entropy/rans_core.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>

#include "mlvc/core/status.h"
#include "mlvc/framework/profile_range.h"

namespace mlvc {
namespace {

constexpr uint16_t kBypassPrecision = 2;
constexpr uint16_t kMaxBypassValue = (1 << kBypassPrecision) - 1;

void RansEncPutBits(RansState& state, uint8_t*& ptr, uint32_t value) {
  MLVC_RANS_ASSERT(kBypassPrecision <= 8);
  MLVC_RANS_ASSERT(value < (1u << kBypassPrecision));

  constexpr uint32_t kFreq = 1 << (kRansScaleBits - kBypassPrecision);
  constexpr uint32_t kMaxState = kFreq << kRansEncRenormShiftBits;
  while (state >= kMaxState) {
    *(--ptr) = static_cast<uint8_t>(state & 0xff);
    state >>= 8;
  }

  state = (state << kBypassPrecision) | value;
}

uint32_t RansDecGetBits(RansState& state, uint8_t*& ptr) {
  const uint32_t value = state & ((1u << kBypassPrecision) - 1);
  state = state >> kBypassPrecision;
  if (state < kRansByteL) {
    state = (state << 8) | *ptr++;
    MLVC_RANS_ASSERT(state >= kRansByteL);
  }
  return value;
}

}  // namespace

std::shared_ptr<CdfGroup> MakeCdfGroup(const std::vector<int32_t>& cdfs,
                                       const std::vector<int64_t>& cdf_shape,
                                       const std::vector<int32_t>& cdf_lengths,
                                       const std::vector<int32_t>& offsets) {
  Check(cdf_shape.size() == 2, "CDF array must be rank 2");
  const int64_t rows = cdf_shape[0];
  const int64_t cols = cdf_shape[1];
  Check(rows >= 0 && cols >= 0, "CDF shape must be non-negative");
  Check(static_cast<std::size_t>(rows * cols) == cdfs.size(), "CDF data size mismatch");
  Check(static_cast<std::size_t>(rows) == cdf_lengths.size(), "CDF length size mismatch");
  Check(static_cast<std::size_t>(rows) == offsets.size(), "CDF offset size mismatch");

  auto group = std::make_shared<CdfGroup>();
  group->cdf_lengths = cdf_lengths;
  group->offsets = offsets;
  group->cdfs.resize(static_cast<std::size_t>(rows));
  group->symbols.resize(static_cast<std::size_t>(rows));
  for (int64_t row = 0; row < rows; ++row) {
    const auto begin = cdfs.begin() + row * cols;
    group->cdfs[row] = std::vector<int32_t>(begin, begin + cols);
    group->symbols[row].resize(static_cast<std::size_t>(cols));
    for (int64_t col = 0; col < cols - 1; ++col) {
      const int32_t start = group->cdfs[row][col];
      const int32_t range = group->cdfs[row][col + 1] - group->cdfs[row][col];
      group->symbols[row][col] =
          RansSymbol{static_cast<uint16_t>(start), static_cast<uint16_t>(range)};
    }
  }
  return group;
}

int RansEncoderCore::AddCdf(std::shared_ptr<CdfGroup> group) {
  cdf_groups_.push_back(std::move(group));
  return static_cast<int>(cdf_groups_.size()) - 1;
}

void RansEncoderCore::ReservePending(std::size_t capacity) { pending_.reserve(capacity); }

void RansEncoderCore::ReserveOutputBytes(std::size_t capacity) {
  output_scratch_.reserve(capacity);
  stream_.reserve(capacity);
}

void RansEncoderCore::Reset() {
  pending_.clear();
  stream_.clear();
}

void RansEncoderCore::EncodeY(std::vector<int16_t> symbols, int cdf_group_index) {
  const std::size_t symbol_count = symbols.size();
  pending_.push_back(PendingTask{PendingTask::Type::kY,
                                 std::move(symbols),
                                 {},
                                 nullptr,
                                 nullptr,
                                 0,
                                 symbol_count,
                                 cdf_group_index,
                                 0,
                                 0});
}

void RansEncoderCore::EncodeY(const std::vector<int16_t>* symbols, std::size_t begin,
                              std::size_t count, int cdf_group_index) {
  Check(symbols != nullptr, "EncodeY shared symbols must not be null");
  Check(begin <= symbols->size() && count <= symbols->size() - begin, "EncodeY symbol range");
  PendingTask task;
  task.type = PendingTask::Type::kY;
  task.borrowed_symbols_y = symbols;
  task.symbol_begin = begin;
  task.symbol_count = count;
  task.cdf_group_index = cdf_group_index;
  pending_.push_back(std::move(task));
}

void RansEncoderCore::EncodeZ(std::vector<int8_t> symbols, int cdf_group_index, int start_offset,
                              int per_channel_size) {
  const std::size_t symbol_count = symbols.size();
  PendingTask task;
  task.type = PendingTask::Type::kZ;
  task.symbols_z = std::move(symbols);
  task.symbol_count = symbol_count;
  task.cdf_group_index = cdf_group_index;
  task.start_offset = start_offset;
  task.per_channel_size = per_channel_size;
  pending_.push_back(std::move(task));
}

void RansEncoderCore::EncodeZ(const std::vector<int8_t>* symbols, std::size_t begin,
                              std::size_t count, int cdf_group_index, int start_offset,
                              int per_channel_size) {
  Check(symbols != nullptr, "EncodeZ shared symbols must not be null");
  Check(begin <= symbols->size() && count <= symbols->size() - begin, "EncodeZ symbol range");
  PendingTask task;
  task.type = PendingTask::Type::kZ;
  task.borrowed_symbols_z = symbols;
  task.symbol_begin = begin;
  task.symbol_count = count;
  task.cdf_group_index = cdf_group_index;
  task.start_offset = start_offset;
  task.per_channel_size = per_channel_size;
  pending_.push_back(std::move(task));
}

std::size_t RansEncoderCore::SymbolCount(const PendingTask& task) {
  if (task.symbol_count != 0) {
    return task.symbol_count;
  }
  return task.type == PendingTask::Type::kY ? task.symbols_y.size() : task.symbols_z.size();
}

int16_t RansEncoderCore::YSymbolAt(const PendingTask& task, std::size_t index) {
  if (task.borrowed_symbols_y != nullptr) {
    return task.borrowed_symbols_y->at(task.symbol_begin + index);
  }
  return task.symbols_y.at(task.symbol_begin + index);
}

int8_t RansEncoderCore::ZSymbolAt(const PendingTask& task, std::size_t index) {
  if (task.borrowed_symbols_z != nullptr) {
    return task.borrowed_symbols_z->at(task.symbol_begin + index);
  }
  return task.symbols_z.at(task.symbol_begin + index);
}

void RansEncoderCore::EncodeOneSymbol(uint8_t*& ptr, RansState& state, int32_t symbol,
                                      int32_t cdf_size, int32_t offset,
                                      const std::vector<RansSymbol>& symbols) {
  const int32_t max_value = cdf_size - 2;
  int32_t value = symbol - offset;

  uint32_t raw_value = 0;
  if (value < 0) {
    raw_value = -2 * value - 1;
    value = max_value;
  } else if (value >= max_value) {
    raw_value = 2 * (value - max_value);
    value = max_value;
  }

  if (value == max_value) {
    std::array<uint16_t, 32> bypass_bins{};
    int bypass_bin_count = 0;
    int32_t bypass_count = 0;
    while ((raw_value >> (bypass_count * kBypassPrecision)) != 0) {
      ++bypass_count;
    }

    int32_t remaining = bypass_count;
    while (remaining >= kMaxBypassValue) {
      bypass_bins.at(bypass_bin_count++) = kMaxBypassValue;
      remaining -= kMaxBypassValue;
    }
    bypass_bins.at(bypass_bin_count++) = static_cast<uint16_t>(remaining);

    for (int32_t i = 0; i < bypass_count; ++i) {
      const int32_t bin = (raw_value >> (i * kBypassPrecision)) & kMaxBypassValue;
      bypass_bins.at(bypass_bin_count++) = static_cast<uint16_t>(bin);
    }

    for (int i = bypass_bin_count - 1; i >= 0; --i) {
      RansEncPutBits(state, ptr, bypass_bins.at(i));
    }
  }
  RansEncPut(state, ptr, symbols[value].start, symbols[value].range);
}

void RansEncoderCore::EncodeYInternal(uint8_t*& ptr, RansState& state, const PendingTask& task) {
  MLVC_PROFILE_RANGE_FUNCTION();
  const CdfGroup& group = *cdf_groups_.at(task.cdf_group_index);
  for (int i = static_cast<int>(SymbolCount(task)) - 1; i >= 0; --i) {
    const int32_t combined_symbol = YSymbolAt(task, static_cast<std::size_t>(i));
    const int32_t cdf_index = combined_symbol & 0xff;
    const int32_t symbol = combined_symbol >> 8;
    EncodeOneSymbol(ptr, state, symbol, group.cdf_lengths[cdf_index], group.offsets[cdf_index],
                    group.symbols[cdf_index]);
  }
}

void RansEncoderCore::EncodeZInternal(uint8_t*& ptr, RansState& state, const PendingTask& task) {
  MLVC_PROFILE_RANGE_FUNCTION();
  const CdfGroup& group = *cdf_groups_.at(task.cdf_group_index);
  for (int i = static_cast<int>(SymbolCount(task)) - 1; i >= 0; --i) {
    const int32_t cdf_index = i / task.per_channel_size + task.start_offset;
    EncodeOneSymbol(ptr, state, ZSymbolAt(task, static_cast<std::size_t>(i)),
                    group.cdf_lengths[cdf_index], group.offsets[cdf_index],
                    group.symbols[cdf_index]);
  }
}

void RansEncoderCore::Flush() {
  MLVC_PROFILE_RANGE_FUNCTION();
  RansState state;
  RansEncInit(state);

  int32_t total_symbol_size = 0;
  for (const PendingTask& task : pending_) {
    total_symbol_size += static_cast<int32_t>(SymbolCount(task));
  }

  if (total_symbol_size == 0) {
    stream_.clear();
    return;
  }

  const std::size_t output_capacity = static_cast<std::size_t>(total_symbol_size) * 8 + 1024;
  output_scratch_.resize(output_capacity);
  uint8_t* ptr_end = output_scratch_.data() + output_scratch_.size();
  uint8_t* ptr = ptr_end;
  for (auto it = pending_.rbegin(); it != pending_.rend(); ++it) {
    if (it->type == PendingTask::Type::kY) {
      EncodeYInternal(ptr, state, *it);
    } else {
      EncodeZInternal(ptr, state, *it);
    }
  }

  RansEncFlush(state, ptr);
  const auto bytes = static_cast<std::size_t>(std::distance(ptr, ptr_end));
  stream_.assign(ptr, ptr + bytes);
}

int RansDecoderCore::AddCdf(std::shared_ptr<CdfGroup> group) {
  cdf_groups_.push_back(std::move(group));
  return static_cast<int>(cdf_groups_.size()) - 1;
}

void RansDecoderCore::ReserveStream(std::size_t capacity) { stream_.reserve(capacity); }

void RansDecoderCore::SetStream(std::vector<uint8_t> stream) {
  stream_ = std::move(stream);
  ptr_ = stream_.data();
  RansDecInit(state_, ptr_);
}

void RansDecoderCore::SetStreamView(const std::vector<uint8_t>& stream) {
  ptr_ = const_cast<uint8_t*>(stream.data());
  RansDecInit(state_, ptr_);
}

int8_t RansDecoderCore::DecodeOneSymbol(const std::vector<int32_t>& cdf, int32_t cdf_size,
                                        int32_t offset) {
  const int32_t max_value = cdf_size - 2;
  const int32_t cum_freq = static_cast<int32_t>(RansDecGet(state_));

  int symbol = 1;
  while (cdf[symbol++] <= cum_freq) {
  }
  symbol -= 2;

  RansDecAdvance(state_, ptr_, cdf[symbol], cdf[symbol + 1] - cdf[symbol]);

  int32_t value = static_cast<int32_t>(symbol);
  if (value == max_value) {
    int32_t bin = static_cast<int32_t>(RansDecGetBits(state_, ptr_));
    int32_t bypass_count = bin;
    while (bin == kMaxBypassValue) {
      bin = static_cast<int32_t>(RansDecGetBits(state_, ptr_));
      bypass_count += bin;
    }

    int32_t raw_value = 0;
    for (int i = 0; i < bypass_count; ++i) {
      bin = static_cast<int32_t>(RansDecGetBits(state_, ptr_));
      raw_value |= bin << (i * kBypassPrecision);
    }
    value = raw_value >> 1;
    if ((raw_value & 1) != 0) {
      value = -value - 1;
    } else {
      value += max_value;
    }
  }

  return static_cast<int8_t>(value + offset);
}

std::vector<int8_t> RansDecoderCore::DecodeY(const std::vector<uint8_t>& indexes,
                                             int cdf_group_index) {
  std::vector<int8_t> decoded(indexes.size());
  DecodeYInto(indexes, 0, indexes.size(), cdf_group_index, &decoded, 0);
  return decoded;
}

void RansDecoderCore::DecodeYInto(const std::vector<uint8_t>& indexes, std::size_t begin,
                                  std::size_t count, int cdf_group_index,
                                  std::vector<int8_t>* output, std::size_t output_begin) {
  Check(output != nullptr, "DecodeYInto output must not be null");
  Check(begin <= indexes.size() && count <= indexes.size() - begin, "DecodeYInto input range");
  Check(output_begin <= output->size() && count <= output->size() - output_begin,
        "DecodeYInto output range");
  const CdfGroup& group = *cdf_groups_.at(cdf_group_index);
  for (std::size_t i = 0; i < count; ++i) {
    const int32_t cdf_index = indexes.at(begin + i);
    (*output)[output_begin + i] = DecodeOneSymbol(
        group.cdfs[cdf_index], group.cdf_lengths[cdf_index], group.offsets[cdf_index]);
  }
}

std::vector<int8_t> RansDecoderCore::DecodeZ(int total_size, int cdf_group_index, int start_offset,
                                             int per_channel_size) {
  std::vector<int8_t> decoded(static_cast<std::size_t>(total_size));
  DecodeZInto(total_size, cdf_group_index, start_offset, per_channel_size, &decoded, 0);
  return decoded;
}

void RansDecoderCore::DecodeZInto(int total_size, int cdf_group_index, int start_offset,
                                  int per_channel_size, std::vector<int8_t>* output,
                                  std::size_t output_begin) {
  Check(output != nullptr, "DecodeZInto output must not be null");
  Check(total_size >= 0, "DecodeZInto total_size must be non-negative");
  const std::size_t count = static_cast<std::size_t>(total_size);
  Check(output_begin <= output->size() && count <= output->size() - output_begin,
        "DecodeZInto output range");
  const CdfGroup& group = *cdf_groups_.at(cdf_group_index);
  for (int i = 0; i < total_size; ++i) {
    const int32_t cdf_index = i / per_channel_size + start_offset;
    (*output)[output_begin + static_cast<std::size_t>(i)] = DecodeOneSymbol(
        group.cdfs[cdf_index], group.cdf_lengths[cdf_index], group.offsets[cdf_index]);
  }
}

}  // namespace mlvc
