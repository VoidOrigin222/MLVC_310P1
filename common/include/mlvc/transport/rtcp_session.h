#ifndef MLVC_TRANSPORT_RTCP_SESSION_H_
#define MLVC_TRANSPORT_RTCP_SESSION_H_

#include <cstdint>
#include <string>
#include <vector>

namespace mlvc::transport {

// Builds and validates the minimal RFC 3550 compound RTCP packet used to end
// an RTP sender session: RR + SDES/CNAME + BYE.  The media stream does not use
// the optional MLVC EOS media unit as its RTP session terminator.
std::vector<uint8_t> EncodeRtcpSessionBye(uint32_t ssrc, const std::string& cname);
uint32_t DecodeRtcpSessionBye(const std::vector<uint8_t>& packet);

}  // namespace mlvc::transport

#endif  // MLVC_TRANSPORT_RTCP_SESSION_H_
