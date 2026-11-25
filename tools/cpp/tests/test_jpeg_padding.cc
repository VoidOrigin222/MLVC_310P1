#include <ulbvc/io/jpeg_padding.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace {
using Bytes = std::vector<uint8_t>;
using ulbvc::io::JpegPaddingStatus;
void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void Marker(Bytes* out, uint8_t marker, const Bytes& body) {
  const auto length = body.size() + 2;
  out->insert(out->end(), {0xff, marker, uint8_t(length >> 8), uint8_t(length)});
  out->insert(out->end(), body.begin(), body.end());
}
Bytes Fixture(bool correct, bool stuffed = false) {
  Bytes out{0xff, 0xd8};
  Bytes quantization(65, 1);
  quantization[0] = 0;
  Marker(&out, 0xdb, quantization);
  Marker(&out, 0xc0,
         stuffed ? Bytes{8, 0, 8, 0, 8, 1, 1, 0x11, 0}
                 : Bytes{8, 0, 16, 0, 16, 3, 1, 0x21, 0, 2, 0x11, 0, 3, 0x11, 0});
  Bytes tables(18, 0);
  tables[1] = 1;  // DC code 0.
  tables[17] = stuffed ? 2 : 0;
  tables.push_back(0x10);
  tables.insert(tables.end(), 16, 0);
  if (stuffed) {
    tables[19] = 1;  // AC code 0: ZRL, code 10: run 14 / magnitude 1.
    tables[20] = 1;
    tables.insert(tables.end(), {0xf0, 0xe1});
  } else {
    tables[20] = 1;  // AC code 00: EOB.
    tables.push_back(0);
  }
  Marker(&out, 0xc4, tables);
  if (!stuffed) Marker(&out, 0xdd, {0, 1});
  Marker(&out, 0xda, stuffed ? Bytes{1, 1, 0, 0, 63, 0} : Bytes{3, 1, 0, 2, 0, 3, 0, 0, 63, 0});
  if (stuffed) {
    out.insert(out.end(), {2, uint8_t(correct ? 0xff : 0x80)});
    if (correct) out.push_back(0);
  } else {
    out.insert(out.end(), {0, uint8_t(correct ? 15 : 0), 0xff, 0xd0, 0, uint8_t(correct ? 15 : 0)});
  }
  out.insert(out.end(), {0xff, 0xd9, 0, 0, 0, 0});
  return out;
}
Bytes Read(const char* path) {
  std::ifstream in(path, std::ios::binary);
  Check(in.good(), "cannot read fixture");
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
void Compare(const Bytes& source, const Bytes& expected) {
  const auto original = source;
  Bytes result;
  const auto first = ulbvc::io::NormalizeJpegPadding(source.data(), source.size(), &result);
  Check(first.status == JpegPaddingStatus::kNormalized, "expected padding normalization");
  Check(first.changed_segments > 0 && result == expected, "coefficient/stuffing/header mismatch");
  Check(source == original, "source was mutated");
  const auto second = ulbvc::io::NormalizeJpegPadding(expected.data(), expected.size(), &result);
  Check(second.status == JpegPaddingStatus::kUnchanged && result.empty(),
        "normalization not idempotent");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Compare(Fixture(false), Fixture(true));
    Compare(Fixture(false, true), Fixture(true, true));
    const auto source = Fixture(false);
    Bytes output;
    // Every truncation before EOI must fall back without publishing partial patches.
    for (std::size_t bytes = 0; bytes < source.size() - 4; ++bytes) {
      const auto r = ulbvc::io::NormalizeJpegPadding(source.data(), bytes, &output);
      Check(r.status != JpegPaddingStatus::kNormalized && output.empty(),
            "truncated input was modified");
    }
    for (std::size_t pos = 0; pos < source.size(); ++pos) {
      auto bad = source;
      bad[pos] ^= 0xff;
      const auto r = ulbvc::io::NormalizeJpegPadding(bad.data(), bad.size(), &output);
      if (r.status != JpegPaddingStatus::kNormalized)
        Check(output.empty(), "partial output on fallback");
      else {
        Bytes again;
        Check(ulbvc::io::NormalizeJpegPadding(output.data(), output.size(), &again).status ==
                  JpegPaddingStatus::kUnchanged,
              "modified input was not idempotent");
      }
    }
    // Optional real camera originals and their independent Python reference corrections.
    Check(argc % 2 == 1, "expected SOURCE EXPECTED pairs");
    for (int i = 1; i < argc; i += 2) Compare(Read(argv[i]), Read(argv[i + 1]));
    std::cout << "jpeg_padding: passed (boundary/stuffing/idempotence/fallback/fixtures)\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
