#include <mlvc/core/status.h>
#include <mlvc/entropy/mlvc_official_entropy.h>
#include <mlvc/framework/profiler.h>
#include <msrtc_rans/EntropyCoder.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <system_error>

namespace mlvc {
namespace {

struct Pmf {
  std::vector<int32_t> lengths;
  std::vector<int32_t> offsets;
  std::vector<int32_t> table;
};

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream input(path);
  Check(input.good(), "failed to open MLVC PMF: " + path.string());
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::vector<int32_t> ParseIntegerArray(const std::string& json, const std::string& key) {
  const std::string name = '"' + key + '"';
  std::size_t pos = json.find(name);
  Check(pos != std::string::npos, "missing MLVC PMF key: " + key);
  pos = json.find('[', pos + name.size());
  Check(pos != std::string::npos, "missing MLVC PMF array: " + key);
  ++pos;

  std::vector<int32_t> values;
  while (pos < json.size()) {
    while (pos < json.size() &&
           (std::isspace(static_cast<unsigned char>(json[pos])) || json[pos] == ',')) {
      ++pos;
    }
    if (pos >= json.size() || json[pos] == ']') {
      break;
    }
    bool negative = false;
    if (json[pos] == '-') {
      negative = true;
      ++pos;
    }
    int64_t value = 0;
    bool has_digit = false;
    while (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos]))) {
      value = value * 10 + static_cast<int64_t>(json[pos++] - '0');
      has_digit = true;
    }
    Check(has_digit, "invalid integer in MLVC PMF array: " + key);
    const int64_t signed_value = negative ? -value : value;
    Check(signed_value >= std::numeric_limits<int32_t>::min() &&
              signed_value <= std::numeric_limits<int32_t>::max(),
          "MLVC PMF integer out of range: " + key);
    values.push_back(static_cast<int32_t>(signed_value));
  }
  Check(!values.empty(), "empty MLVC PMF array: " + key);
  return values;
}

Pmf LoadPmf(const std::filesystem::path& path) {
  const std::string json = ReadText(path);
  return {ParseIntegerArray(json, "pmf_lengths"), ParseIntegerArray(json, "pmf_offsets"),
          ParseIntegerArray(json, "pmf_table")};
}

void CheckRans(const std::error_code& error, const std::string& operation) {
  Check(!error, operation + ": " + error.message());
}

std::vector<int32_t> ToInt32(const std::vector<int8_t>& values) {
  return {values.begin(), values.end()};
}

std::vector<int32_t> ToInt32(const std::vector<uint8_t>& values) {
  return {values.begin(), values.end()};
}

std::vector<int32_t> BuildZIndexes(int q_index, int channels, int height, int width) {
  Check(q_index >= 0 && q_index < 64, "MLVC q_index must be in [0, 63]");
  Check(channels > 0 && height > 0 && width > 0, "invalid MLVC z shape");
  const std::size_t plane = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
  std::vector<int32_t> indexes(static_cast<std::size_t>(channels) * plane);
  for (int channel = 0; channel < channels; ++channel) {
    std::fill_n(indexes.begin() + static_cast<std::ptrdiff_t>(channel * plane), plane,
                q_index * channels + channel);
  }
  return indexes;
}

}  // namespace

std::vector<int8_t> NarrowMlvcSymbols(const std::vector<int32_t>& values) {
  std::vector<int8_t> output(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    Check(values[i] >= std::numeric_limits<int8_t>::min() &&
              values[i] <= std::numeric_limits<int8_t>::max(),
          "decoded MLVC symbol is outside int8 range");
    output[i] = static_cast<int8_t>(values[i]);
  }
  return output;
}

class MlvcOfficialEntropyEncoder::Impl {
 public:
  explicit Impl(const std::filesystem::path& model_directory) {
    const Pmf gaussian = LoadPmf(model_directory / "gaussian_pmf.json");
    const Pmf bit_estimator = LoadPmf(model_directory / "bit_estimator_pmf.json");
    CheckRans(gaussian_.Initialize(msrtc_rans::RansVariant::RansByte, gaussian.lengths,
                                   gaussian.offsets, gaussian.table, 16, 2),
              "initialize MLVC Gaussian encoder");
    CheckRans(z_.Initialize(msrtc_rans::RansVariant::RansByte, bit_estimator.lengths,
                            bit_estimator.offsets, bit_estimator.table, 16, 2),
              "initialize MLVC bit-estimator encoder");
  }

  msrtc_rans::EntropyEncoder gaussian_;
  msrtc_rans::EntropyEncoder z_;
};

MlvcOfficialEntropyEncoder::MlvcOfficialEntropyEncoder(const std::filesystem::path& model_directory)
    : impl_(std::make_unique<Impl>(model_directory)) {}

MlvcOfficialEntropyEncoder::~MlvcOfficialEntropyEncoder() = default;

void MlvcOfficialEntropyEncoder::Encode(const std::vector<int8_t>& z, const std::vector<int8_t>& y0,
                                        const std::vector<int8_t>& y1,
                                        const std::vector<uint8_t>& scales0,
                                        const std::vector<uint8_t>& scales1, int q_index,
                                        int z_channels, int z_height, int z_width,
                                        std::vector<uint8_t>* payload) const {
  Check(payload != nullptr, "MLVC payload output is required");
  Check(y0.size() == scales0.size() && y1.size() == scales1.size(), "MLVC y/scales size mismatch");
  const std::vector<int32_t> y0_values = ToInt32(y0);
  const std::vector<int32_t> y1_values = ToInt32(y1);
  const std::vector<int32_t> z_values = ToInt32(z);
  const std::vector<int32_t> scale0_indexes = ToInt32(scales0);
  const std::vector<int32_t> scale1_indexes = ToInt32(scales1);
  const std::vector<int32_t> z_indexes = BuildZIndexes(q_index, z_channels, z_height, z_width);
  Check(z_indexes.size() == z_values.size(), "MLVC z shape/symbol size mismatch");

  msrtc_rans::HeapResizableBuffer buffer;
  msrtc_rans::RansEncoderStream stream;
  CheckRans(stream.Initialize(msrtc_rans::RansVariant::RansByte, buffer),
            "initialize MLVC rANS stream");
  // Official mlvc-main order is y_raw_1, y_raw_0, then z_raw.
  CheckRans(impl_->gaussian_.Encode(stream, scale1_indexes, y1_values), "encode MLVC y_raw_1");
  CheckRans(impl_->gaussian_.Encode(stream, scale0_indexes, y0_values), "encode MLVC y_raw_0");
  CheckRans(impl_->z_.Encode(stream, z_indexes, z_values), "encode MLVC z_raw");
  const msrtc_rans::span<const std::byte> encoded = stream.Flush();
  payload->resize(encoded.size());
  for (std::size_t i = 0; i < encoded.size(); ++i) {
    (*payload)[i] = std::to_integer<uint8_t>(encoded[i]);
  }
}

class MlvcOfficialEntropyDecoder::Impl {
 public:
  explicit Impl(const std::filesystem::path& model_directory) {
    const Pmf gaussian = LoadPmf(model_directory / "gaussian_pmf.json");
    const Pmf bit_estimator = LoadPmf(model_directory / "bit_estimator_pmf.json");
    CheckRans(gaussian_.Initialize(msrtc_rans::RansVariant::RansByte, gaussian.lengths,
                                   gaussian.offsets, gaussian.table, 16, 2),
              "initialize MLVC Gaussian decoder");
    CheckRans(z_.Initialize(msrtc_rans::RansVariant::RansByte, bit_estimator.lengths,
                            bit_estimator.offsets, bit_estimator.table, 16, 2),
              "initialize MLVC bit-estimator decoder");
    CheckRans(stream_.Initialize(msrtc_rans::RansVariant::RansByte),
              "initialize MLVC rANS decoder stream");
  }

  msrtc_rans::EntropyDecoder gaussian_;
  msrtc_rans::EntropyDecoder z_;
  msrtc_rans::RansDecoderStream stream_;
  std::vector<std::byte> payload_;
};

MlvcOfficialEntropyDecoder::MlvcOfficialEntropyDecoder(const std::filesystem::path& model_directory)
    : impl_(std::make_unique<Impl>(model_directory)) {}

MlvcOfficialEntropyDecoder::~MlvcOfficialEntropyDecoder() = default;

void MlvcOfficialEntropyDecoder::SetStream(const std::vector<uint8_t>& payload,
                                           Profiler* profiler) {
  if (impl_->stream_.IsOpen()) {
    impl_->stream_.Close();
  }
  {
    ScopedCpuTimer timer(profiler, "entropy.payload_copy");
    impl_->payload_.resize(payload.size());
    for (std::size_t i = 0; i < payload.size(); ++i) {
      impl_->payload_[i] = static_cast<std::byte>(payload[i]);
    }
  }
  {
    ScopedCpuTimer timer(profiler, "entropy.stream_open");
    CheckRans(impl_->stream_.Open(impl_->payload_), "open MLVC rANS payload");
  }
}

std::vector<int32_t> MlvcOfficialEntropyDecoder::DecodeZInt32(
    int q_index, int z_channels, int z_height, int z_width, Profiler* profiler) {
  std::vector<int32_t> indexes;
  {
    ScopedCpuTimer timer(profiler, "entropy.z.index_build");
    indexes = BuildZIndexes(q_index, z_channels, z_height, z_width);
  }
  std::vector<int32_t> values;
  {
    ScopedCpuTimer timer(profiler, "entropy.z.output_allocate");
    values.resize(indexes.size());
  }
  {
    ScopedCpuTimer timer(profiler, "entropy.z.rans_decode");
    CheckRans(impl_->z_.Decode(values, indexes, impl_->stream_), "decode MLVC z_raw");
  }
  return values;
}

std::vector<int8_t> MlvcOfficialEntropyDecoder::DecodeZ(int q_index, int z_channels, int z_height,
                                                        int z_width, Profiler* profiler) {
  std::vector<int32_t> values =
      DecodeZInt32(q_index, z_channels, z_height, z_width, profiler);
  ScopedCpuTimer timer(profiler, "entropy.z.int32_to_int8");
  return NarrowMlvcSymbols(values);
}

std::vector<int32_t> MlvcOfficialEntropyDecoder::DecodeYInt32(
    const std::vector<uint8_t>& scales, bool eof, Profiler* profiler) {
  const char* const part = eof ? "y1" : "y0";
  std::vector<int32_t> indexes;
  std::vector<int32_t> values;
  {
    ScopedCpuTimer timer(profiler, std::string("entropy.") + part + ".index_convert");
    indexes = ToInt32(scales);
  }
  {
    ScopedCpuTimer timer(profiler, std::string("entropy.") + part + ".output_allocate");
    values.resize(indexes.size());
  }
  {
    ScopedCpuTimer timer(profiler, std::string("entropy.") + part + ".rans_decode");
    CheckRans(impl_->gaussian_.Decode(values, indexes, impl_->stream_), "decode MLVC y_raw");
  }
  if (eof) {
    ScopedCpuTimer timer(profiler, "entropy.y1.stream_finalize");
    Check(impl_->stream_.CheckEOF(), "MLVC rANS payload has trailing or incomplete data");
    impl_->stream_.Close();
  }
  return values;
}

std::vector<int8_t> MlvcOfficialEntropyDecoder::DecodeY(const std::vector<uint8_t>& scales,
                                                        bool eof, Profiler* profiler) {
  const char* const part = eof ? "y1" : "y0";
  std::vector<int32_t> values = DecodeYInt32(scales, eof, profiler);
  ScopedCpuTimer timer(profiler, std::string("entropy.") + part + ".int32_to_int8");
  return NarrowMlvcSymbols(values);
}

}  // namespace mlvc
