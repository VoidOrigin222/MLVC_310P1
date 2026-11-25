#ifndef ULBVC_IO_JPEG_PADDING_H_
#define ULBVC_IO_JPEG_PADDING_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ulbvc::io {

enum class JpegPaddingStatus { kUnchanged, kNormalized, kUnsupported, kInvalid };
struct JpegPaddingResult {
  JpegPaddingStatus status = JpegPaddingStatus::kInvalid;
  std::size_t changed_segments = 0;
};

// Supports single-scan 8-bit baseline Huffman JPEG. Only segment padding bits
// and required FF/00 stuffing change; coefficients and the input stay intact.
// Output is empty on unchanged/unsupported/invalid input: callers use the source.
JpegPaddingResult NormalizeJpegPadding(const uint8_t* data, std::size_t size,
                                       std::vector<uint8_t>* output);

}  // namespace ulbvc::io
#endif
