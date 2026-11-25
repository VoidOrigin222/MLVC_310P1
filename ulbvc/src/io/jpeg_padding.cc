#include <ulbvc/io/jpeg_padding.h>

#include <algorithm>
#include <array>
#include <limits>

namespace ulbvc::io {
namespace {
struct FastCode {
  uint8_t length = 0;
  uint8_t symbol = 0;
};
struct Huffman {
  bool valid = false;
  std::array<FastCode, 512> fast{};
  std::array<int, 17> minimum{}, maximum{}, offset{};
  std::array<uint8_t, 256> symbols{};
};
struct Component {
  int id = 0, h = 0, v = 0;
};
struct ScanComponent {
  int blocks = 0, dc = 0, ac = 0;
};
struct Patch {
  std::size_t position;
  uint8_t value;
};

unsigned Be16(const uint8_t* p) { return (unsigned(p[0]) << 8) | p[1]; }

bool BuildTable(const uint8_t* counts, const uint8_t* symbols, bool ac, Huffman* table) {
  *table = Huffman{};
  table->maximum.fill(-1);
  int code = 0, offset = 0;
  for (int length = 1; length <= 16; ++length) {
    const int count = counts[length - 1];
    // An all-ones Huffman code is reserved for byte alignment.
    if (code + count >= (1 << length)) return false;
    table->minimum[length] = code;
    table->offset[length] = offset;
    if (count) table->maximum[length] = code + count - 1;
    for (int i = 0; i < count; ++i) {
      const uint8_t symbol = symbols[offset + i];
      if ((!ac && symbol > 11) ||
          (ac && ((symbol & 15) > 10 || (!(symbol & 15) && symbol != 0 && symbol != 0xf0))))
        return false;
      table->symbols[offset + i] = symbol;
      if (length <= 9) {
        const int start = (code + i) << (9 - length);
        for (int j = 0; j < (1 << (9 - length)); ++j)
          table->fast[start + j] = {static_cast<uint8_t>(length), symbol};
      }
    }
    offset += count;
    code = (code + count) << 1;
  }
  table->valid = offset > 0;
  return table->valid;
}

class Bits {
 public:
  explicit Bits(const std::vector<uint8_t>& bytes) : bytes_(bytes) {}
  std::size_t remaining() const { return bytes_.size() * 8 - position_; }
  bool Skip(unsigned count) {
    if (count > remaining()) return false;
    position_ += count;
    return true;
  }
  int Symbol(const Huffman& table) {
    const std::size_t left = remaining();
    if (!left) return -1;
    const auto index = position_ / 8;
    uint32_t window = uint32_t(bytes_[index]) << 16;
    if (index + 1 < bytes_.size()) window |= uint32_t(bytes_[index + 1]) << 8;
    if (index + 2 < bytes_.size()) window |= bytes_[index + 2];
    window = ((window << (position_ & 7)) >> 8) & 0xffff;
    if (left >= 9) {
      const auto fast = table.fast[window >> 7];
      if (fast.length) {
        position_ += fast.length;
        return fast.symbol;
      }
    }
    for (int length = left >= 9 ? 10 : 1; length <= 16 && unsigned(length) <= left; ++length) {
      const int code = window >> (16 - length);
      if (code >= table.minimum[length] && code <= table.maximum[length]) {
        position_ += length;
        return table.symbols[table.offset[length] + code - table.minimum[length]];
      }
    }
    return -1;
  }

 private:
  const std::vector<uint8_t>& bytes_;
  std::size_t position_ = 0;
};

bool WalkSegment(const std::vector<uint8_t>& bytes, std::size_t mcus,
                 const std::vector<ScanComponent>& scan, const std::array<Huffman, 8>& tables,
                 unsigned* padding) {
  Bits bits(bytes);
  for (std::size_t mcu = 0; mcu < mcus; ++mcu) {
    for (const auto& component : scan) {
      for (int block = 0; block < component.blocks; ++block) {
        const int dc = bits.Symbol(tables[component.dc]);
        if (dc < 0 || !bits.Skip(dc)) return false;
        int coefficient = 1;
        while (coefficient < 64) {
          const int symbol = bits.Symbol(tables[4 + component.ac]);
          if (symbol < 0) return false;
          if (!symbol) break;
          if (symbol == 0xf0) {
            coefficient += 16;
            if (coefficient > 64) return false;
          } else {
            coefficient += symbol >> 4;
            if (coefficient >= 64 || !bits.Skip(symbol & 15)) return false;
            ++coefficient;
          }
        }
      }
    }
  }
  if (bits.remaining() > 7) return false;
  *padding = static_cast<unsigned>(bits.remaining());
  return true;
}
}  // namespace

JpegPaddingResult NormalizeJpegPadding(const uint8_t* data, std::size_t size,
                                       std::vector<uint8_t>* output) {
  if (!output) return {};
  output->clear();
  if (!data || size < 4 || size > 32 * 1024 * 1024 || data[0] != 0xff || data[1] != 0xd8) return {};
  std::array<Huffman, 8> tables;
  std::vector<Component> components;
  std::vector<ScanComponent> scan;
  unsigned width = 0, height = 0, restart = 0;
  int max_h = 0, max_v = 0;
  std::size_t p = 2;
  while (p < size) {
    if (data[p++] != 0xff) return {};
    while (p < size && data[p] == 0xff) ++p;
    if (p >= size) return {};
    const unsigned marker = data[p++];
    if (p + 2 > size) return {};
    const auto length = Be16(data + p);
    if (length < 2 || length > size - p) return {};
    const auto* body = data + p + 2;
    const auto bytes = length - 2;
    p += length;
    if (marker == 0xc0) {
      if (!components.empty() || bytes < 6) return {};
      if (body[0] != 8 || (body[5] != 1 && body[5] != 3))
        return {JpegPaddingStatus::kUnsupported, 0};
      height = Be16(body + 1);
      width = Be16(body + 3);
      if (!width || !height || uint64_t(width) * height > 64 * 1024 * 1024 ||
          bytes != unsigned(6 + 3 * body[5]))
        return {};
      int blocks = 0;
      for (unsigned q = 6; q < bytes; q += 3) {
        Component c{body[q], body[q + 1] >> 4, body[q + 1] & 15};
        if (!c.h || !c.v || c.h > 4 || c.v > 4 || body[q + 2] > 3 ||
            std::any_of(components.begin(), components.end(),
                        [&](const auto& v) { return v.id == c.id; }))
          return {};
        components.push_back(c);
        max_h = std::max(max_h, c.h);
        max_v = std::max(max_v, c.v);
        blocks += c.h * c.v;
      }
      if (blocks > 10) return {JpegPaddingStatus::kUnsupported, 0};
    } else if (marker == 0xc4) {
      std::size_t q = 0;
      while (q < bytes) {
        if (bytes - q < 17) return {};
        const unsigned key = body[q++];
        if ((key >> 4) > 1 || (key & 15) > 3) return {};
        const auto* counts = body + q;
        unsigned count = 0;
        for (int i = 0; i < 16; ++i) count += counts[i];
        q += 16;
        if (!count || count > 256 || count > bytes - q ||
            !BuildTable(counts, body + q, key >> 4, &tables[(key >> 4) * 4 + (key & 15)]))
          return {};
        q += count;
      }
    } else if (marker == 0xdd) {
      if (bytes != 2) return {};
      restart = Be16(body);
    } else if (marker == 0xda) {
      if (components.empty() || !bytes) return {};
      if (body[0] != components.size()) return {JpegPaddingStatus::kUnsupported, 0};
      if (bytes != 1 + 2 * components.size() + 3) return {};
      if (body[bytes - 3] || body[bytes - 2] != 63 || body[bytes - 1])
        return {JpegPaddingStatus::kUnsupported, 0};
      std::array<bool, 3> seen{};
      for (unsigned q = 1; q < 1 + 2 * components.size(); q += 2) {
        const auto found = std::find_if(components.begin(), components.end(),
                                        [&](const auto& c) { return c.id == body[q]; });
        if (found == components.end()) return {};
        const auto index = found - components.begin();
        if (seen[index]) return {};
        seen[index] = true;
        const unsigned dc = body[q + 1] >> 4, ac = body[q + 1] & 15;
        if (dc > 3 || ac > 3 || !tables[dc].valid || !tables[4 + ac].valid) return {};
        scan.push_back({components.size() == 1 ? 1 : found->h * found->v, int(dc), int(ac)});
      }
      break;
    } else if (marker != 0xdb && marker != 0xfe && !(marker >= 0xe0 && marker <= 0xef)) {
      return {JpegPaddingStatus::kUnsupported, 0};
    }
  }
  if (scan.empty()) return {};
  const unsigned mcu_width = components.size() == 1 ? 8 : max_h * 8;
  const unsigned mcu_height = components.size() == 1 ? 8 : max_v * 8;
  std::size_t remaining =
      std::size_t((width + mcu_width - 1) / mcu_width) * ((height + mcu_height - 1) / mcu_height);
  std::vector<uint8_t> raw;
  raw.reserve(restart ? 4096 : size - p);
  std::vector<Patch> patches;
  unsigned next_restart = 0;
  bool ended = false;
  while (p < size && remaining) {
    raw.clear();
    std::size_t last_byte = 0;
    unsigned marker = 0;
    while (p < size) {
      const auto position = p;
      const uint8_t value = data[p++];
      if (value == 0xff) {
        if (p >= size) return {};
        if (data[p] != 0) {
          while (p < size && data[p] == 0xff) ++p;
          if (p >= size) return {};
          marker = data[p++];
          break;
        }
        ++p;
      }
      last_byte = position;
      raw.push_back(value);
    }
    const auto mcus = restart ? std::min<std::size_t>(restart, remaining) : remaining;
    if (raw.empty() || !marker) return {};
    if (remaining > mcus) {
      if (marker != 0xd0 + next_restart) return {};
      next_restart = (next_restart + 1) & 7;
    } else if (marker != 0xd9) {
      return {JpegPaddingStatus::kUnsupported, 0};
    }
    unsigned padding = 0;
    if (!WalkSegment(raw, mcus, scan, tables, &padding)) return {};
    if (padding) {
      const auto fixed = static_cast<uint8_t>(raw.back() | ((1u << padding) - 1));
      if (fixed != raw.back()) patches.push_back({last_byte, fixed});
    }
    remaining -= mcus;
    ended = marker == 0xd9;
  }
  if (!ended || remaining) return {};
  if (patches.empty()) return {JpegPaddingStatus::kUnchanged, 0};
  output->reserve(size + patches.size());
  std::size_t copied = 0;
  for (const auto& patch : patches) {
    output->insert(output->end(), data + copied, data + patch.position);
    output->push_back(patch.value);
    if (patch.value == 0xff) output->push_back(0);
    copied = patch.position + 1;
  }
  output->insert(output->end(), data + copied, data + size);
  return {JpegPaddingStatus::kNormalized, patches.size()};
}
}  // namespace ulbvc::io
