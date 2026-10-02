#include "mlvc/io/mlvc_bitstream.h"
#include "mlvc/io/udp_frame_transport.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace {
using mlvc::codec::MlvcFrameType;
void Require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
template <class Function> void Reject(Function function, const char* message) {
  bool rejected = false;
  try { function(); } catch (const std::exception&) { rejected = true; }
  Require(rejected, message);
}
mlvc::io::MlvcBitstreamHeader Header(bool warp) {
  mlvc::io::MlvcBitstreamHeader header;
  header.width = header.height = 64;
  header.version = 4;
  header.codec_bundle_sha256.fill(0x42);
  header.translation_warp = warp;
  return header;
}
mlvc::io::MlvcFrameMetadata Metadata(int index, bool warp) {
  mlvc::io::MlvcFrameMetadata metadata;
  metadata.explicit_metadata = true;
  metadata.translation_warp = warp;
  metadata.model_q_index = 10;
  metadata.pts = index * 3000;
  if (index == 0) {
    metadata.unit_flags = mlvc::transport::kEfuRandomAccess | mlvc::transport::kEfuResetReference;
  } else {
    metadata.short_ref_frame_id = index - 1;
    if (warp) { metadata.kx = -128; metadata.ky = 127; }
  }
  return metadata;
}
void ExpectMetadata(const mlvc::io::MlvcFrameMetadata& metadata, int index, bool warp) {
  Require(metadata.explicit_metadata && metadata.translation_warp == warp,
          "decoded frame lost warp capability");
  Require(metadata.kx == (warp && index ? -128 : 0) && metadata.ky == (warp && index ? 127 : 0),
          "decoded frame lost signed geometry");
}
void WriteBytes(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  Require(out.good(), "failed to prepare malformed file fixture");
}
void ExpectFileReject(const std::filesystem::path& path) {
  Reject([&] {
    mlvc::io::MlvcBitstreamReader reader(path);
    int index, q;
    MlvcFrameType type;
    std::vector<uint8_t> payload;
    while (reader.ReadFrame(&index, &type, &q, &payload)) {}
  }, "file reader accepted missing or unexpected warp flag");
}
}  // namespace

int main() {
  const auto path = std::filesystem::temp_directory_path() /
      ("mlvc-warp-transport-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
  try {
    const std::vector<uint8_t> payload(5000, 0x39);
    uintmax_t legacy_size = 0;
    for (bool warp : {false, true}) {
      const auto header = Header(warp);
      {
        mlvc::io::MlvcBitstreamWriter writer(path, header);
        Reject([&] { writer.WriteFrame(0, MlvcFrameType::kIFrame, 10, Metadata(0, !warp), payload); },
               "writer accepted mismatching warp mode");
        writer.WriteFrame(0, MlvcFrameType::kIFrame, 10, Metadata(0, warp), payload);
        writer.WriteFrame(1, MlvcFrameType::kPFrame, 10, Metadata(1, warp), payload);
        Reject([&] { writer.SwitchConfiguration(2, Header(!warp)); }, "writer allowed warp capability switch");
        writer.Close();
      }
      const auto size = std::filesystem::file_size(path);
      if (!warp) legacy_size = size;
      else Require(size == legacy_size + 10, "file must add 8 bytes capability and 2 bytes P geometry");
      {
        mlvc::io::MlvcBitstreamReader reader(path);
        Require(reader.header().translation_warp == warp, "file header lost warp mode");
        for (int expected = 0; expected < 2; ++expected) {
          int index, q;
          MlvcFrameType type;
          std::vector<uint8_t> decoded;
          Require(reader.ReadFrame(&index, &type, &q, &decoded), "file frame missing");
          Require(index == expected && q == 10 && decoded == payload, "file entropy payload changed");
          ExpectMetadata(reader.last_frame_metadata(), expected, warp);
        }
      }
      // Flip only the EFU mode flag. Payload CRC stays valid, so it must be
      // the active SCU capability check that prevents the wrong decoder path.
      std::ifstream in(path, std::ios::binary);
      std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), {});
      const std::size_t scu_size = (static_cast<std::size_t>(bytes[20]) << 24) |
          (static_cast<std::size_t>(bytes[21]) << 16) | (static_cast<std::size_t>(bytes[22]) << 8) | bytes[23];
      bytes[16 + scu_size + 19] ^= mlvc::transport::kEfuTranslationWarp;
      WriteBytes(path, bytes);
      ExpectFileReject(path);

      mlvc::io::RtpMlvcReceiver receiver(0, header.codec_bundle_sha256);
      mlvc::io::RtpMlvcSender sender("127.0.0.1", receiver.local_port());
      sender.SendHeader(header);
      Reject([&] { sender.SendFrame(0, MlvcFrameType::kIFrame, 10, Metadata(0, !warp), payload); },
             "RTP sender accepted mismatching warp mode");
      sender.SendFrame(0, MlvcFrameType::kIFrame, 10, Metadata(0, warp), payload);
      sender.SendFrame(1, MlvcFrameType::kPFrame, 10, Metadata(1, warp), payload);
      sender.SendEnd();
      Require(receiver.ReceiveHeader().translation_warp == warp, "RTP header lost capability");
      for (int expected = 0; expected < 2; ++expected) {
        int index, q;
        MlvcFrameType type;
        std::vector<uint8_t> decoded;
        Require(receiver.ReceiveFrame(&index, &type, &q, &decoded), "RTP frame missing");
        Require(index == expected && q == 10 && decoded == payload, "RTP entropy payload changed");
        ExpectMetadata(receiver.last_frame_metadata(), expected, warp);
      }
      int index, q;
      MlvcFrameType type;
      std::vector<uint8_t> decoded;
      Require(!receiver.ReceiveFrame(&index, &type, &q, &decoded), "RTP EOS missing");
      sender.Close();
    }
    mlvc::io::UdpMlvcSender udp("127.0.0.1", 19999);
    Reject([&] { udp.SendHeader(Header(true)); }, "legacy UDP accepted unsupported warp mode");
    for (bool capability : {false, true}) {
      mlvc::transport::MlvcScu scu;
      scu.coded_width = scu.visible_width = 64;
      scu.coded_height = scu.visible_height = 64;
      scu.codec_bundle_sha256.fill(0x42);
      scu.translation_warp = capability;
      mlvc::transport::MlvcEfu iframe;
      iframe.unit_flags |= mlvc::transport::kEfuRandomAccess | mlvc::transport::kEfuResetReference;
      if (!capability) iframe.unit_flags |= mlvc::transport::kEfuTranslationWarp;
      iframe.entropy_payload = {3, 4, 5};
      mlvc::io::RtpMlvcReceiver receiver(0, scu.codec_bundle_sha256);
      mlvc::transport::RtpMlvcSender sender("127.0.0.1", receiver.local_port());
      sender.SetSessionConfig(1, mlvc::transport::SerializeScu(scu));
      sender.SendUnit(mlvc::transport::RtpUnitType::kEfu, 0, 1, 0, 0,
                      mlvc::transport::SerializeEfu(iframe));
      sender.Flush();
      Require(receiver.ReceiveHeader().translation_warp == capability, "negative RTP initial SCU lost");
      Reject([&] {
        int index, q;
        MlvcFrameType type;
        std::vector<uint8_t> decoded;
        receiver.ReceiveFrame(&index, &type, &q, &decoded);
      }, "RTP reader accepted missing or unexpected warp flag");
      sender.Close();
    }
    std::filesystem::remove(path);
    std::cout << "translation warp file/RTP transport tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::filesystem::remove(path);
    std::cerr << error.what() << '\n';
    return 1;
  }
}
