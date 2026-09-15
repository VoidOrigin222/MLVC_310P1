#include "mlvc/entropy/entropy_codec.h"

#include <algorithm>
#include <cmath>

#include "mlvc/core/status.h"
#include "mlvc/framework/profile_range.h"

namespace mlvc {
namespace {

const float kLogScaleMin = std::log(kScaleMin);
const float kLogScaleMax = std::log(kScaleMax);
const float kLogScaleStep = (kLogScaleMax - kLogScaleMin) / (kScaleLevel - 1);
const float kLogStepRecip = 1.0f / kLogScaleStep;

uint8_t ScaleIndex(float scale) {
  const float clipped = std::min(std::max(scale, kScaleMin), kScaleMax);
  return static_cast<uint8_t>((std::log(clipped) - kLogScaleMin) * kLogStepRecip);
}

std::vector<uint8_t> FirstPart(const std::vector<uint8_t>& stream0,
                               const std::vector<uint8_t>& stream1) {
  int identical_bytes = 0;
  int check_bytes = static_cast<int>(std::min(stream0.size(), stream1.size()));
  check_bytes = std::min(check_bytes, 8);
  for (int i = 0; i < check_bytes; ++i) {
    if (stream0[stream0.size() - 1 - i] != 0) {
      break;
    }
    if (stream1[stream1.size() - 1 - i] != 0) {
      break;
    }
    ++identical_bytes;
  }
  if (identical_bytes == 0 && !stream0.empty() && !stream1.empty() &&
      stream0.back() == stream1.back()) {
    identical_bytes = 1;
  }

  std::vector<uint8_t> output;
  output.reserve(stream0.size() + stream1.size() - identical_bytes);
  output.insert(output.end(), stream0.begin(), stream0.end());
  output.insert(output.end(), stream1.rbegin() + identical_bytes, stream1.rend());
  return output;
}

void FirstPartInto(const std::vector<uint8_t>& stream0, const std::vector<uint8_t>& stream1,
                   std::vector<uint8_t>* output) {
  Check(output != nullptr, "first-part output must not be null");
  int identical_bytes = 0;
  int check_bytes = static_cast<int>(std::min(stream0.size(), stream1.size()));
  check_bytes = std::min(check_bytes, 8);
  for (int i = 0; i < check_bytes; ++i) {
    if (stream0[stream0.size() - 1 - i] != 0) {
      break;
    }
    if (stream1[stream1.size() - 1 - i] != 0) {
      break;
    }
    ++identical_bytes;
  }
  if (identical_bytes == 0 && !stream0.empty() && !stream1.empty() &&
      stream0.back() == stream1.back()) {
    identical_bytes = 1;
  }

  output->clear();
  output->reserve(stream0.size() + stream1.size() - identical_bytes);
  output->insert(output->end(), stream0.begin(), stream0.end());
  output->insert(output->end(), stream1.rbegin() + identical_bytes, stream1.rend());
}

}  // namespace

std::vector<int16_t> BuildIndexesEncoder(const std::vector<int8_t>& symbols,
                                         const std::vector<float>& scales,
                                         std::optional<float> force_zero_thres) {
  std::vector<int16_t> encoded;
  BuildIndexesEncoderInto(symbols, scales, force_zero_thres, &encoded);
  return encoded;
}

void BuildIndexesEncoderInto(const std::vector<int8_t>& symbols, const std::vector<float>& scales,
                             std::optional<float> force_zero_thres, std::vector<int16_t>* encoded) {
  MLVC_PROFILE_RANGE_FUNCTION();
  Check(encoded != nullptr, "encoded output must not be null");
  Check(symbols.size() == scales.size(), "symbols/scales size mismatch");
  encoded->clear();
  encoded->reserve(symbols.size());
  for (std::size_t i = 0; i < symbols.size(); ++i) {
    const float clipped = std::min(std::max(scales[i], kScaleMin), kScaleMax);
    if (force_zero_thres.has_value() && clipped <= *force_zero_thres) {
      continue;
    }
    const int32_t symbol = static_cast<int32_t>(symbols[i]);
    const int32_t index = static_cast<int32_t>(ScaleIndex(clipped));
    encoded->push_back(static_cast<int16_t>((symbol << 8) + index));
  }
}

void BuildIndexesDecoderInto(const std::vector<float>& scales,
                             std::optional<float> force_zero_thres, std::vector<uint8_t>* indexes,
                             std::vector<uint8_t>* skip_condition) {
  Check(indexes != nullptr, "indexes output must not be null");
  indexes->clear();
  indexes->reserve(scales.size());
  if (skip_condition != nullptr) {
    skip_condition->clear();
  }
  for (float scale : scales) {
    const float clipped = std::min(std::max(scale, kScaleMin), kScaleMax);
    const bool keep = !force_zero_thres.has_value() || clipped > *force_zero_thres;
    if (skip_condition != nullptr) {
      skip_condition->push_back(keep ? 1 : 0);
    }
    if (keep) {
      indexes->push_back(ScaleIndex(clipped));
    }
  }
}

std::vector<uint8_t> BuildIndexesDecoder(const std::vector<float>& scales,
                                         std::optional<float> force_zero_thres,
                                         std::vector<uint8_t>* skip_condition) {
  std::vector<uint8_t> indexes;
  BuildIndexesDecoderInto(scales, force_zero_thres, &indexes, skip_condition);
  return indexes;
}

EntropyEncoder::EntropyEncoder(const RuntimeSidecar& sidecar, const std::string& prefix,
                               bool use_two_parts, int z_channel_override)
    : use_two_parts_(use_two_parts),
      z_channel_(z_channel_override >= 0 ? z_channel_override : sidecar.z_channel(prefix)) {
  auto z_group = sidecar.MakeCdfGroupForPrefix(prefix, true);
  auto gaussian_group = sidecar.MakeCdfGroupForPrefix(prefix, false);
  z_group_ = encoder0_.AddCdf(z_group);
  gaussian_group_ = encoder0_.AddCdf(gaussian_group);
  encoder1_.AddCdf(z_group);
  encoder1_.AddCdf(gaussian_group);
}

void EntropyEncoder::Reserve(std::size_t max_y_parts, std::size_t max_y_symbols_per_part,
                             std::size_t max_encoded_stream_bytes) {
  y_index_scratch_.resize(max_y_parts);
  for (std::vector<int16_t>& scratch : y_index_scratch_) {
    scratch.reserve(max_y_symbols_per_part);
  }
  split_z_scratch_.resize(1);
  split_z_scratch_.clear();
  encoder0_.ReservePending(max_y_parts + 1);
  encoder1_.ReservePending(max_y_parts + 1);
  encoder0_.ReserveOutputBytes(max_encoded_stream_bytes);
  encoder1_.ReserveOutputBytes(max_encoded_stream_bytes);
}

void EntropyEncoder::Reset() {
  encoder0_.Reset();
  encoder1_.Reset();
  split_z_scratch_.clear();
  next_y_scratch_slot_ = 0;
}

void EntropyEncoder::EncodeZ(std::vector<int8_t> symbols, int qp, int height, int width) {
  MLVC_PROFILE_RANGE_FUNCTION();
  const int per_channel_size = height * width;
  if (use_two_parts_) {
    const int split = static_cast<int>(symbols.size()) / 2;
    const int channel_half = split / per_channel_size;
    split_z_scratch_.push_back(std::move(symbols));
    const std::vector<int8_t>& split_symbols = split_z_scratch_.back();
    encoder0_.EncodeZ(&split_symbols, 0, static_cast<std::size_t>(split), z_group_, qp * z_channel_,
                      per_channel_size);
    encoder1_.EncodeZ(&split_symbols, static_cast<std::size_t>(split),
                      split_symbols.size() - static_cast<std::size_t>(split), z_group_,
                      qp * z_channel_ + channel_half, per_channel_size);
    return;
  }
  encoder0_.EncodeZ(std::move(symbols), z_group_, qp * z_channel_, per_channel_size);
}

void EntropyEncoder::EncodeZView(const std::vector<int8_t>& symbols, int qp, int height,
                                 int width) {
  const int per_channel_size = height * width;
  if (use_two_parts_) {
    const int split = static_cast<int>(symbols.size()) / 2;
    const int channel_half = split / per_channel_size;
    encoder0_.EncodeZ(&symbols, 0, static_cast<std::size_t>(split), z_group_, qp * z_channel_,
                      per_channel_size);
    encoder1_.EncodeZ(&symbols, static_cast<std::size_t>(split),
                      symbols.size() - static_cast<std::size_t>(split), z_group_,
                      qp * z_channel_ + channel_half, per_channel_size);
    return;
  }
  encoder0_.EncodeZ(&symbols, 0, symbols.size(), z_group_, qp * z_channel_, per_channel_size);
}

void EntropyEncoder::EncodeY(const std::vector<int8_t>& symbols, const std::vector<float>& scales,
                             std::optional<float> force_zero_thres) {
  MLVC_PROFILE_RANGE_FUNCTION();
  if (next_y_scratch_slot_ == y_index_scratch_.size()) {
    y_index_scratch_.emplace_back();
  }
  std::vector<int16_t>& encoded = y_index_scratch_[next_y_scratch_slot_++];
  BuildIndexesEncoderInto(symbols, scales, force_zero_thres, &encoded);
  if (use_two_parts_) {
    const int split = static_cast<int>(encoded.size()) / 2;
    encoder0_.EncodeY(&encoded, 0, static_cast<std::size_t>(split), gaussian_group_);
    encoder1_.EncodeY(&encoded, static_cast<std::size_t>(split),
                      encoded.size() - static_cast<std::size_t>(split), gaussian_group_);
    return;
  }
  encoder0_.EncodeY(&encoded, 0, encoded.size(), gaussian_group_);
}

void EntropyEncoder::EncodeYIndexesView(const std::vector<int16_t>& indexes) {
  MLVC_PROFILE_RANGE_FUNCTION();
  if (next_y_scratch_slot_ == y_index_scratch_.size()) {
    y_index_scratch_.emplace_back();
  }
  std::vector<int16_t>& encoded = y_index_scratch_[next_y_scratch_slot_++];
  encoded.assign(indexes.begin(), indexes.end());
  if (use_two_parts_) {
    const int split = static_cast<int>(encoded.size()) / 2;
    encoder0_.EncodeY(&encoded, 0, static_cast<std::size_t>(split), gaussian_group_);
    encoder1_.EncodeY(&encoded, static_cast<std::size_t>(split),
                      encoded.size() - static_cast<std::size_t>(split), gaussian_group_);
    return;
  }
  encoder0_.EncodeY(&encoded, 0, encoded.size(), gaussian_group_);
}

std::vector<uint8_t> EntropyEncoder::Flush() {
  MLVC_PROFILE_RANGE_FUNCTION();
  encoder0_.Flush();
  encoder1_.Flush();
  if (use_two_parts_) {
    return FirstPart(encoder0_.stream(), encoder1_.stream());
  }
  return encoder0_.stream();
}

void EntropyEncoder::FlushInto(std::vector<uint8_t>* output) {
  MLVC_PROFILE_RANGE_FUNCTION();
  Check(output != nullptr, "entropy flush output must not be null");
  encoder0_.Flush();
  encoder1_.Flush();
  if (use_two_parts_) {
    FirstPartInto(encoder0_.stream(), encoder1_.stream(), output);
    return;
  }
  output->assign(encoder0_.stream().begin(), encoder0_.stream().end());
}

EntropyDecoder::EntropyDecoder(const RuntimeSidecar& sidecar, const std::string& prefix,
                               bool use_two_parts, int z_channel_override)
    : use_two_parts_(use_two_parts),
      z_channel_(z_channel_override >= 0 ? z_channel_override : sidecar.z_channel(prefix)) {
  auto z_group = sidecar.MakeCdfGroupForPrefix(prefix, true);
  auto gaussian_group = sidecar.MakeCdfGroupForPrefix(prefix, false);
  z_group_ = decoder0_.AddCdf(z_group);
  gaussian_group_ = decoder0_.AddCdf(gaussian_group);
  decoder1_.AddCdf(z_group);
  decoder1_.AddCdf(gaussian_group);
}

void EntropyDecoder::Reserve(std::size_t max_y_symbols, std::size_t max_stream_bytes) {
  index_scratch_.reserve(max_y_symbols);
  skip_scratch_.reserve(max_y_symbols);
  compact_decode_scratch_.reserve(max_y_symbols);
  reversed_stream_scratch_.reserve(max_stream_bytes);
  decoder0_.ReserveStream(max_stream_bytes);
  decoder1_.ReserveStream(max_stream_bytes);
}

void EntropyDecoder::SetStream(const std::vector<uint8_t>& stream) {
  decoder0_.SetStreamView(stream);
  if (use_two_parts_) {
    reversed_stream_scratch_.assign(stream.rbegin(), stream.rend());
    decoder1_.SetStreamView(reversed_stream_scratch_);
  }
}

std::vector<int8_t> EntropyDecoder::DecodeZ(int qp, int height, int width) {
  std::vector<int8_t> decoded;
  DecodeZInto(qp, height, width, &decoded);
  return decoded;
}

void EntropyDecoder::DecodeZInto(int qp, int height, int width, std::vector<int8_t>* output) {
  const int per_channel_size = height * width;
  const int total_size = z_channel_ * per_channel_size;
  DecodeZInto(static_cast<std::size_t>(total_size), qp, height, width, output);
}

std::vector<int8_t> EntropyDecoder::DecodeZ(std::size_t total_size, int qp, int height, int width) {
  std::vector<int8_t> decoded;
  DecodeZInto(total_size, qp, height, width, &decoded);
  return decoded;
}

void EntropyDecoder::DecodeZInto(std::size_t total_size, int qp, int height, int width,
                                 std::vector<int8_t>* output) {
  MLVC_PROFILE_RANGE_FUNCTION();
  Check(output != nullptr, "DecodeZInto output must not be null");
  const int per_channel_size = height * width;
  Check(per_channel_size > 0, "DecodeZInto invalid spatial shape");
  Check(total_size % static_cast<std::size_t>(per_channel_size) == 0,
        "DecodeZInto total size does not match spatial shape");
  output->resize(total_size);
  if (use_two_parts_) {
    const std::size_t split = total_size / 2;
    const int channel_half = split / per_channel_size;
    decoder0_.DecodeZInto(split, z_group_, qp * z_channel_, per_channel_size, output, 0);
    decoder1_.DecodeZInto(total_size - split, z_group_, qp * z_channel_ + channel_half,
                          per_channel_size, output, static_cast<std::size_t>(split));
    return;
  }
  decoder0_.DecodeZInto(total_size, z_group_, qp * z_channel_, per_channel_size, output, 0);
}

std::vector<int8_t> EntropyDecoder::DecodeY(const std::vector<float>& scales,
                                            std::optional<float> force_zero_thres) {
  std::vector<int8_t> output;
  DecodeYInto(scales, force_zero_thres, &output);
  return output;
}

void EntropyDecoder::DecodeYInto(const std::vector<float>& scales,
                                 std::optional<float> force_zero_thres,
                                 std::vector<int8_t>* output) {
  MLVC_PROFILE_RANGE_FUNCTION();
  Check(output != nullptr, "DecodeYInto output must not be null");
  BuildIndexesDecoderInto(scales, force_zero_thres, &index_scratch_, &skip_scratch_);
  compact_decode_scratch_.resize(index_scratch_.size());
  if (use_two_parts_) {
    const std::size_t split = index_scratch_.size() / 2;
    decoder0_.DecodeYInto(index_scratch_, 0, split, gaussian_group_, &compact_decode_scratch_, 0);
    decoder1_.DecodeYInto(index_scratch_, split, index_scratch_.size() - split, gaussian_group_,
                          &compact_decode_scratch_, split);
  } else {
    decoder0_.DecodeYInto(index_scratch_, 0, index_scratch_.size(), gaussian_group_,
                          &compact_decode_scratch_, 0);
  }

  if (!force_zero_thres.has_value()) {
    output->assign(compact_decode_scratch_.begin(), compact_decode_scratch_.end());
    return;
  }

  output->assign(scales.size(), 0);
  std::size_t decoded_index = 0;
  for (std::size_t i = 0; i < skip_scratch_.size(); ++i) {
    if (skip_scratch_[i] != 0) {
      (*output)[i] = compact_decode_scratch_.at(decoded_index++);
    }
  }
}

std::vector<int8_t> EntropyDecoder::DecodeYIndexes(const std::vector<uint8_t>& indexes) {
  std::vector<int8_t> output;
  DecodeYIndexesInto(indexes, &output);
  return output;
}

void EntropyDecoder::DecodeYIndexesInto(const std::vector<uint8_t>& indexes,
                                        std::vector<int8_t>* output) {
  MLVC_PROFILE_RANGE_FUNCTION();
  Check(output != nullptr, "DecodeYIndexesInto output must not be null");
  compact_decode_scratch_.resize(indexes.size());
  if (use_two_parts_) {
    const std::size_t split = indexes.size() / 2;
    decoder0_.DecodeYInto(indexes, 0, split, gaussian_group_, &compact_decode_scratch_, 0);
    decoder1_.DecodeYInto(indexes, split, indexes.size() - split, gaussian_group_,
                          &compact_decode_scratch_, split);
  } else {
    decoder0_.DecodeYInto(indexes, 0, indexes.size(), gaussian_group_, &compact_decode_scratch_, 0);
  }
  output->assign(compact_decode_scratch_.begin(), compact_decode_scratch_.end());
}

}  // namespace mlvc
