#ifndef MLVC_ENTROPY_SIDECAR_H_
#define MLVC_ENTROPY_SIDECAR_H_

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "mlvc/core/types.h"
#include "mlvc/entropy/rans_core.h"

namespace mlvc {

struct SidecarArray {
  std::string name;
  DataType dtype = DataType::kFloat16;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;

  std::size_t ElementCount() const;
  std::vector<int32_t> AsInt32() const;
  std::vector<uint16_t> AsFloat16Bits() const;
  float ScalarFloat32() const;
};

class RuntimeSidecar {
 public:
  static RuntimeSidecar Load(const std::filesystem::path& path);

  const SidecarArray& Get(const std::string& name) const;
  const std::map<std::string, SidecarArray>& arrays() const { return arrays_; }
  float force_zero_thres() const;
  float python_fast_force_zero_thres() const;
  int z_channel(const std::string& prefix) const;
  int ShiftedQp(int base_qp, int frame_adaptation_index) const;
  const uint16_t* QScaleData(const std::string& name, int qp) const;
  std::shared_ptr<CdfGroup> MakeCdfGroupForPrefix(const std::string& prefix, bool z_table) const;

 private:
  std::map<std::string, SidecarArray> arrays_;
};

}  // namespace mlvc

#endif  // MLVC_ENTROPY_SIDECAR_H_
