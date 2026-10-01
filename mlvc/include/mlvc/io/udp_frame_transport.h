#ifndef MLVC_IO_UDP_FRAME_TRANSPORT_H_
#define MLVC_IO_UDP_FRAME_TRANSPORT_H_

#include <cstddef>
#include <cstdint>
#include <array>
#include <chrono>
#include <deque>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "mlvc/io/mlvc_bitstream.h"
#include "mlvc/transport/rtp_message_receiver.h"
#include "mlvc/transport/rtp_mlvc.h"
#include "mlvc/transport/udp_message_transport.h"
#include "mlvc/transport/mlvc_media_unit.h"

namespace mlvc::io {

// UDP framing follows VideoTrans: EB90 magic, 1024-byte fragments, and CDDE
// tail. A reassembled message starts with a one-byte MLVC message kind.
class UdpMlvcSender {
 public:
  UdpMlvcSender(const std::string& host, uint16_t port,
                mlvc::transport::UdpSendOptions options = {});
  ~UdpMlvcSender();
  void SendHeader(const MlvcBitstreamHeader& header);
  void SendFrame(int frame_index, mlvc::codec::MlvcFrameType frame_type, int q_index,
                 const std::vector<uint8_t>& payload);
  void SendEnd();
  void Close();

 private:
  mlvc::transport::UdpMessageSender sender_;
  uint64_t max_payload_size_ = 0;
  int expected_frame_index_ = 0;
  bool header_sent_ = false;
  bool closed_ = false;
};

class UdpMlvcReceiver {
 public:
  explicit UdpMlvcReceiver(uint16_t port);
  ~UdpMlvcReceiver();
  MlvcBitstreamHeader ReceiveHeader();
  bool ReceiveFrame(int* frame_index, mlvc::codec::MlvcFrameType* frame_type, int* q_index,
                    std::vector<uint8_t>* payload);
  const MlvcFrameMetadata& last_frame_metadata() const { return last_frame_metadata_; }

 private:
  mlvc::transport::UdpMessageReceiver receiver_;
  MlvcBitstreamHeader header_;
  uint64_t max_payload_size_ = 0;
  int expected_frame_index_ = 0;
  bool header_received_ = false;
  MlvcFrameMetadata last_frame_metadata_;
};

class RtpMlvcSender {
 public:
  RtpMlvcSender(const std::string& host, uint16_t port, uint64_t pacing_rate_bps = 0,
                std::size_t max_burst_bytes = 4096,
                std::size_t max_queue_bytes = 4u * 1024u * 1024u,
                uint64_t max_queue_delay_ms = 1000, uint8_t payload_type = 96);
  ~RtpMlvcSender();
  void SendHeader(const MlvcBitstreamHeader& header);
  // Starts a new protocol configuration.  The next frame must be a random
  // access I-frame; frame IDs remain monotonic across the switch.
  void SendConfiguration(uint32_t config_id, const MlvcBitstreamHeader& header);
  void SendFrame(int frame_index, mlvc::codec::MlvcFrameType frame_type, int q_index,
                 const std::vector<uint8_t>& payload);
  void SendFrame(int frame_index, mlvc::codec::MlvcFrameType frame_type, int q_index,
                 const MlvcFrameMetadata& metadata, const std::vector<uint8_t>& payload);
  void SendEnd();
  void Close();
  bool ConsumeRandomAccessRequest();
  bool PopMlvcControl(mlvc::transport::MlvcControlMessage* message);
  uint16_t rtcp_local_port() const;
  void SendMlvcControlResponse(const mlvc::transport::MlvcControlMessage& request, bool accepted,
                               const std::string& reason);

 private:
  mlvc::transport::RtpMlvcSender sender_;
  std::vector<uint8_t> session_config_;
  uint32_t config_id_ = 1;
  uint32_t timestamp_offset_ = 0;
  double fps_ = 30.0;
  uint64_t max_payload_size_ = 0;
  int expected_frame_index_ = 0;
  bool header_sent_ = false;
  bool configuration_switch_pending_ = false;
  bool closed_ = false;
};

class RtpMlvcReceiver {
 public:
  explicit RtpMlvcReceiver(uint16_t port,
                           std::array<uint8_t, 32> expected_bundle_hash = {},
                           uint8_t payload_type = 96);
  ~RtpMlvcReceiver();
  uint16_t local_port() const { return receiver_.local_port(); }
  MlvcBitstreamHeader ReceiveHeader();
  bool ReceiveFrame(int* frame_index, mlvc::codec::MlvcFrameType* frame_type, int* q_index,
                    std::vector<uint8_t>* payload);
 const MlvcFrameMetadata& last_frame_metadata() const { return last_frame_metadata_; }

 private:
  void HandleConfigurationUnit(const std::vector<uint8_t>& unit);
  void HandleEndUnit(const std::vector<uint8_t>& unit);
  void QueueConfigurationUnit(std::vector<uint8_t> unit, uint32_t config_id);
  void BufferFrameUnit(const std::vector<uint8_t>& unit, uint32_t frame_id);
  bool PopReadyFrame(std::vector<uint8_t>* unit);
  bool TryStartAtRandomAccess();
  bool HasPendingCurrentFrameAfterExpected() const;
  std::optional<uint32_t> FindRecoveryFrame() const;
  void MarkDependencyGap();
  void DiscardPendingFramesBefore(uint32_t frame_id);
  void RememberDecodedFrame(uint32_t frame_id, bool store_as_ltr);

  mlvc::transport::RtpMessageReceiver receiver_;
  MlvcBitstreamHeader header_;
  uint32_t config_id_ = 0;
  std::array<uint8_t, 32> expected_bundle_hash_{};
  std::vector<uint8_t> active_config_unit_;
  uint32_t pending_config_id_ = 0;
  uint64_t max_payload_size_ = 0;
  int expected_frame_index_ = 0;
  bool header_received_ = false;
  std::optional<uint32_t> eos_frame_count_;
  std::optional<std::vector<uint8_t>> pending_eos_unit_;
  std::optional<std::chrono::steady_clock::time_point> reorder_deadline_;
  bool waiting_for_recovery_ = false;
  std::size_t pending_frame_bytes_ = 0;
  std::deque<std::vector<uint8_t>> pending_config_units_;
  std::map<uint32_t, std::vector<uint8_t>> pending_frames_;
  std::map<uint32_t, std::pair<std::size_t, uint32_t>> recent_frames_;
  std::optional<uint32_t> available_short_reference_id_;
  std::set<uint32_t> ltr_frame_ids_;
  MlvcFrameMetadata last_frame_metadata_;
};

}  // namespace mlvc::io

#endif  // MLVC_IO_UDP_FRAME_TRANSPORT_H_
