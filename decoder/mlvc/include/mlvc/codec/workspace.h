#ifndef MLVC_CODEC_WORKSPACE_H_
#define MLVC_CODEC_WORKSPACE_H_

#include <map>
#include <string>

#include "mlvc/core/buffer.h"

namespace mlvc {

struct WorkspaceBufferPlan {
  std::map<std::string, std::size_t> acl_bytes;
  std::map<std::string, std::size_t> pinned_bytes;
};

class CodecWorkspace {
 public:
  explicit CodecWorkspace(WorkspaceBufferPlan plan);

  AclBuffer& Acl(const std::string& name);
  PinnedHostBuffer& Pinned(const std::string& name);

 private:
  std::map<std::string, AclBuffer> acl_buffers_;
  std::map<std::string, PinnedHostBuffer> pinned_buffers_;
};

}  // namespace mlvc

#endif  // MLVC_CODEC_WORKSPACE_H_
