#include "mlvc/bitstream/bitstream.h"

#include <algorithm>

#include "mlvc/core/status.h"

namespace mlvc {
namespace {

uint8_t ReadByte(std::istream& in) {
  char value = 0;
  in.read(&value, 1);
  Check(in.good(), "unexpected end of bitstream");
  return static_cast<uint8_t>(value);
}

void WriteByte(std::ostream& out, uint8_t value) {
  const char byte = static_cast<char>(value);
  out.write(&byte, 1);
  Check(out.good(), "failed to write bitstream byte");
}

}  // namespace

std::pair<int, bool> SpsHelper::GetSpsId(const Sps& target) {
  int max_id = -1;
  for (const Sps& record : records_) {
    if (record.height == target.height && record.width == target.width &&
        record.use_ada_i == target.use_ada_i && record.ec_part == target.ec_part) {
      return {record.sps_id, false};
    }
    max_id = std::max(max_id, record.sps_id);
  }
  Check(max_id < 15, "too many SPS records");
  Sps record = target;
  record.sps_id = max_id + 1;
  records_.push_back(record);
  return {record.sps_id, true};
}

void SpsHelper::Add(const Sps& record) {
  for (Sps& current : records_) {
    if (current.sps_id == record.sps_id) {
      current = record;
      return;
    }
  }
  records_.push_back(record);
}

std::optional<Sps> SpsHelper::Get(int sps_id) const {
  for (const Sps& record : records_) {
    if (record.sps_id == sps_id) {
      return record;
    }
  }
  return std::nullopt;
}

void WriteUIntAdaptive(std::ostream& out, uint32_t value) {
  if (value < (1u << 7)) {
    WriteByte(out, static_cast<uint8_t>((value & 0xff) | (0x00 << 7)));
    return;
  }
  if (value < (1u << 14)) {
    const uint8_t v0 = static_cast<uint8_t>(value & 0xff);
    const uint8_t v1 = static_cast<uint8_t>(((value >> 8) & 0xff) | (0x02 << 6));
    WriteByte(out, v1);
    WriteByte(out, v0);
    return;
  }
  Check(value < (1u << 30), "adaptive integer is too large");
  const uint8_t v0 = static_cast<uint8_t>(value & 0xff);
  const uint8_t v1 = static_cast<uint8_t>((value >> 8) & 0xff);
  const uint8_t v2 = static_cast<uint8_t>((value >> 16) & 0xff);
  const uint8_t v3 = static_cast<uint8_t>(((value >> 24) & 0xff) | (0x03 << 6));
  WriteByte(out, v3);
  WriteByte(out, v2);
  WriteByte(out, v1);
  WriteByte(out, v0);
}

uint32_t ReadUIntAdaptive(std::istream& in) {
  uint32_t v3 = ReadByte(in);
  if ((v3 >> 7) == 0) {
    return v3;
  }
  const uint32_t v2 = ReadByte(in);
  if ((v3 >> 6) == 0x02) {
    return ((v3 & 0x3f) << 8) + v2;
  }
  v3 &= 0x3f;
  const uint32_t v1 = ReadByte(in);
  const uint32_t v0 = ReadByte(in);
  return (v3 << 24) + (v2 << 16) + (v1 << 8) + v0;
}

void WriteSps(std::ostream& out, const Sps& sps) {
  Check(sps.sps_id >= 0 && sps.sps_id < 16, "sps_id must be in [0, 15]");
  WriteByte(out, static_cast<uint8_t>((static_cast<int>(NalType::kSps) << 4) + sps.sps_id));
  WriteUIntAdaptive(out, static_cast<uint32_t>(sps.height));
  WriteUIntAdaptive(out, static_cast<uint32_t>(sps.width));
  WriteByte(out, static_cast<uint8_t>((sps.ec_part << 2) + sps.use_ada_i));
}

void WriteIp(std::ostream& out, bool is_i_frame, int sps_id, int qp,
             const std::vector<uint8_t>& bitstream) {
  const NalType nal_type = is_i_frame ? NalType::kI : NalType::kP;
  WriteByte(out, static_cast<uint8_t>((static_cast<int>(nal_type) << 4) + sps_id));
  WriteByte(out, static_cast<uint8_t>(qp));
  WriteUIntAdaptive(out, static_cast<uint32_t>(bitstream.size()));
  out.write(reinterpret_cast<const char*>(bitstream.data()),
            static_cast<std::streamsize>(bitstream.size()));
  Check(out.good(), "failed to write bitstream payload");
}

NalHeader ReadHeader(std::istream& in) {
  const uint8_t flag = ReadByte(in);
  const int nal_type = flag >> 4;
  Check(nal_type >= 0 && nal_type <= 2, "unsupported NAL type");
  return NalHeader{static_cast<NalType>(nal_type), flag & 0x0f};
}

Sps ReadSpsRemaining(std::istream& in, int sps_id) {
  Sps sps;
  sps.sps_id = sps_id;
  sps.height = static_cast<int>(ReadUIntAdaptive(in));
  sps.width = static_cast<int>(ReadUIntAdaptive(in));
  const uint8_t flag = ReadByte(in);
  sps.ec_part = (flag >> 2) & 0x01;
  sps.use_ada_i = flag & 0x01;
  return sps;
}

std::pair<int, std::vector<uint8_t>> ReadIpRemaining(std::istream& in) {
  const int qp = ReadByte(in);
  const uint32_t stream_length = ReadUIntAdaptive(in);
  std::vector<uint8_t> stream(stream_length);
  in.read(reinterpret_cast<char*>(stream.data()), stream_length);
  Check(in.good(), "failed to read bitstream payload");
  return {qp, std::move(stream)};
}

}  // namespace mlvc
