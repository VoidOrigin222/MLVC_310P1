#include "mlvc/codec/workspace.h"

#include <utility>

#include "mlvc/core/status.h"

namespace mlvc {

CodecWorkspace::CodecWorkspace(WorkspaceBufferPlan plan) {
  for (const auto& [name, bytes] : plan.acl_bytes) {
    acl_buffers_.emplace(name, AclBuffer(bytes));
  }
  for (const auto& [name, bytes] : plan.pinned_bytes) {
    pinned_buffers_.emplace(name, PinnedHostBuffer(bytes));
  }
}

AclBuffer& CodecWorkspace::Acl(const std::string& name) {
  auto it = acl_buffers_.find(name);
  Check(it != acl_buffers_.end(), "unknown ACL workspace buffer: " + name);
  return it->second;
}

PinnedHostBuffer& CodecWorkspace::Pinned(const std::string& name) {
  auto it = pinned_buffers_.find(name);
  Check(it != pinned_buffers_.end(), "unknown pinned workspace buffer: " + name);
  return it->second;
}

}  // namespace mlvc
