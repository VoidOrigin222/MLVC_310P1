#ifndef MLVC_BITSTREAM_BITSTREAM_H_
#define MLVC_BITSTREAM_BITSTREAM_H_

#include <cstdint>
#include <optional>
#include <sstream>
#include <vector>

namespace mlvc {

enum class NalType : uint8_t {
  kSps = 0,
  kI = 1,
  kP = 2,
};

struct Sps {
  int sps_id = -1;
  int height = 0;
  int width = 0;
  int ec_part = 0;
  int use_ada_i = 0;
};

struct NalHeader {
  NalType nal_type = NalType::kSps;
  int sps_id = 0;
};

class SpsHelper {
 public:
  std::pair<int, bool> GetSpsId(const Sps& target);
  void Add(const Sps& record);
  std::optional<Sps> Get(int sps_id) const;

 private:
  std::vector<Sps> records_;
};

void WriteUIntAdaptive(std::ostream& out, uint32_t value);
uint32_t ReadUIntAdaptive(std::istream& in);
void WriteSps(std::ostream& out, const Sps& sps);
void WriteIp(std::ostream& out, bool is_i_frame, int sps_id, int qp,
             const std::vector<uint8_t>& bitstream);
NalHeader ReadHeader(std::istream& in);
Sps ReadSpsRemaining(std::istream& in, int sps_id);
std::pair<int, std::vector<uint8_t>> ReadIpRemaining(std::istream& in);

}  // namespace mlvc

#endif  // MLVC_BITSTREAM_BITSTREAM_H_
