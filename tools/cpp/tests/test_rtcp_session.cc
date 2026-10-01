#include "mlvc/transport/rtcp_session.h"

#include <cassert>
#include <chrono>
#include <iostream>
#include <stdexcept>

using namespace mlvc::transport;

int main() {
  try {
    const auto packet = mlvc::transport::EncodeRtcpSessionBye(0x01020304u, "mlvc-test");
    assert(mlvc::transport::DecodeRtcpSessionBye(packet) == 0x01020304u);
    auto malformed = packet;
    malformed[3] = 0;
    bool rejected = false;
    try {
      (void)mlvc::transport::DecodeRtcpSessionBye(malformed);
    } catch (const std::exception&) {
      rejected = true;
    }
    assert(rejected);

    const RtcpReportBlock block{0x11223344u, 7, -3, 99, 1234, 0x01020304u,
                                0x05060708u};
    RtcpSenderReport sr;
    sr.sender_ssrc = 0x01020304u;
    sr.ntp_seconds = 10;
    sr.ntp_fraction = 20;
    sr.rtp_timestamp = 30;
    sr.packet_count = 40;
    sr.octet_count = 50;
    sr.reports.push_back(block);
    mlvc::transport::MlvcControlMessage command;
    command.flags = mlvc::transport::kMlvcControlAtomic;
    command.transaction_id = 42;
    command.media_ssrc = 0x55667788u;
    command.apply_after_frame_id = 123;
    command.tlvs.push_back({0x8002, {17}});
    command.tlvs.push_back({0x0009, {0, 0, 0, 123}});
    const auto app = mlvc::transport::EncodeMlvcRtcpApp(sr.sender_ssrc, command);
    const auto decoded_app = mlvc::transport::DecodeMlvcRtcpApp(app);
    assert(decoded_app.control_ssrc == sr.sender_ssrc);
    assert(decoded_app.transaction_id == command.transaction_id);
    assert(decoded_app.tlvs.size() == 2 && decoded_app.tlvs[0].value[0] == 17);

    const auto nack = EncodeRtcpNack(RtcpNack{sr.sender_ssrc, 0x99u, 1234, 0x20});
    const auto pli = EncodeRtcpPli(RtcpPli{sr.sender_ssrc, 0x99u});
    const auto fir = EncodeRtcpFir(RtcpFir{sr.sender_ssrc, 0x99u, 8});
    const auto tmmbr = EncodeRtcpTmmbr(RtcpTmmbr{sr.sender_ssrc, 0x99u, 4000000, 37});
    const auto compound = mlvc::transport::EncodeRtcpSenderReport(
        sr, "mlvc-test", {nack, pli, fir, tmmbr, app});
    const auto contents = mlvc::transport::DecodeRtcpCompound(compound);
    assert(contents.sender_report.has_value());
    assert(contents.sender_report->reports[0].cumulative_lost == -3);
    assert(contents.cname == "mlvc-test");
    assert(contents.nacks.size() == 1 && contents.nacks[0].pid == 1234);
    assert(contents.plis.size() == 1 && contents.plis[0].media_ssrc == 0x99u);
    assert(contents.firs.size() == 1 && contents.firs[0].sequence_number == 8);
    assert(contents.tmmbrs.size() == 1 && contents.tmmbrs[0].overhead_bytes == 37);
    assert(contents.mlvc_controls.size() == 1 &&
           contents.mlvc_controls[0].transaction_id == command.transaction_id);

    auto bad_app = app;
    bad_app[3] = 0;  // Declared length no longer covers the APP envelope.
    bool bad_app_rejected = false;
    try {
      (void)mlvc::transport::DecodeMlvcRtcpApp(bad_app);
    } catch (const std::exception&) {
      bad_app_rejected = true;
    }
    assert(bad_app_rejected);

    auto padded_app = command;
    padded_app.tlvs.resize(1);
    auto bad_tlv = mlvc::transport::EncodeMlvcRtcpApp(sr.sender_ssrc, padded_app);
    bad_tlv.back() = 1;  // Non-zero 32-bit alignment padding must be rejected.
    bool bad_tlv_rejected = false;
    try {
      (void)mlvc::transport::DecodeMlvcRtcpApp(bad_tlv);
    } catch (const std::exception&) {
      bad_tlv_rejected = true;
    }
    assert(bad_tlv_rejected);

    mlvc::transport::MlvcControlMessage non_atomic = command;
    non_atomic.flags = 0;
    bool non_atomic_rejected = false;
    try {
      (void)mlvc::transport::EncodeMlvcRtcpApp(sr.sender_ssrc, non_atomic);
    } catch (const std::exception&) {
      non_atomic_rejected = true;
    }
    assert(non_atomic_rejected);

#ifndef _WIN32
    RtcpUdpEndpoint receiver_endpoint(0);
    RtcpUdpEndpoint sender_endpoint(0, "127.0.0.1", receiver_endpoint.local_port());
    sender_endpoint.Send(compound);
    std::vector<uint8_t> received_compound;
    assert(receiver_endpoint.Receive(&received_compound, std::chrono::milliseconds(500)));
    assert(DecodeRtcpCompound(received_compound).cname == "mlvc-test");
    receiver_endpoint.Close();
    sender_endpoint.Close();
    bool closed_rejected = false;
    try {
      (void)receiver_endpoint.Receive(&received_compound, std::chrono::milliseconds(1));
    } catch (const std::exception&) {
      closed_rejected = true;
    }
    assert(closed_rejected);
#endif
    std::cout << "rtcp session test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rtcp session test failed: " << error.what() << "\n";
    return 1;
  }
}
