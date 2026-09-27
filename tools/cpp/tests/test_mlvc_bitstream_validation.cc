#include <mlvc/codec/mlvc_entropy.h>
#include <mlvc/core/status.h>
#include <mlvc/io/mlvc_bitstream.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::streamoff kV4HeaderBytes =
    8 + sizeof(uint32_t) + 2 * sizeof(int32_t) + sizeof(double) + 6 * sizeof(int32_t) +
    sizeof(double) + sizeof(uint32_t) + 2 * sizeof(int32_t);
constexpr std::streamoff kFramePayloadSizeOffset =
    kV4HeaderBytes + sizeof(int32_t) + sizeof(uint8_t) + sizeof(int32_t);

template <typename Function>
void ExpectReject(Function&& function, const char* description) {
  bool rejected = false;
  try {
    function();
  } catch (const std::exception&) {
    rejected = true;
  }
  mlvc::Check(rejected, std::string("expected validation failure: ") + description);
}

void ExpectReaderReject(const std::filesystem::path& path, const char* description) {
  ExpectReject(
      [&] {
        mlvc::io::MlvcBitstreamReader reader(path);
        int frame_index = 0;
        mlvc::codec::MlvcFrameType frame_type = mlvc::codec::MlvcFrameType::kPFrame;
        int q_index = 0;
        std::vector<uint8_t> payload;
        while (reader.ReadFrame(&frame_index, &frame_type, &q_index, &payload)) {
        }
      },
      description);
}

void WriteBigEndian32(std::fstream* stream, std::streamoff position, uint32_t value) {
  const char bytes[4] = {static_cast<char>(value >> 24), static_cast<char>(value >> 16),
                         static_cast<char>(value >> 8), static_cast<char>(value)};
  stream->seekp(position);
  stream->write(bytes, sizeof(bytes));
}

}  // namespace

int main() {
  try {
    mlvc::io::MlvcBitstreamHeader header;
    header.version = 3;
    header.width = 1920;
    header.height = 1080;
    header.fps = 25.0;
    header.q_index = 18;
    header.gop = 96;
    header.reset_interval = 32;
    header.ltr_start_idx = 8;
    header.ltr_period = 64;
    header.ltr_qp_shift = 8;
    header.target_bitrate_bps = 1000000.0;
    for (std::size_t i = 0; i < header.codec_bundle_sha256.size(); ++i) {
      header.codec_bundle_sha256[i] = static_cast<uint8_t>(0xc0u + i);
    }
    mlvc::io::ValidateMlvcBitstreamHeader(header);
    mlvc::io::ValidateMlvcDecoderOutputShape(header, {1, 3, 1088, 1920});
    ExpectReject(
        [&] { mlvc::io::ValidateMlvcDecoderOutputShape(header, {1, 3, 720, 1280}); },
        "decoder output shape smaller than stream resolution");
    ExpectReject(
        [&] { mlvc::io::ValidateMlvcDecoderOutputShape(header, {1, 1, 1088, 1920}); },
        "decoder output channel shape");
    ExpectReject(
        [&] {
          auto invalid = header;
          invalid.width = 32;
          mlvc::io::ValidateMlvcBitstreamHeader(invalid);
        },
        "small width");
    ExpectReject(
        [&] {
          auto invalid = header;
          invalid.fps = std::numeric_limits<double>::quiet_NaN();
          mlvc::io::ValidateMlvcBitstreamHeader(invalid);
        },
        "non-finite fps");
    ExpectReject(
        [&] {
          auto invalid = header;
          invalid.ltr_qp_shift = 64;
          mlvc::io::ValidateMlvcBitstreamHeader(invalid);
        },
        "ltr qp shift");
    ExpectReject(
        [&] {
          auto invalid = header;
          invalid.version = 5;
          mlvc::io::ValidateMlvcBitstreamHeader(invalid);
        },
        "unsupported version");

    const auto roundtrip_path =
        std::filesystem::temp_directory_path() / "mlvc_header_v4_roundtrip.mlvc";
    auto v4 = header;
    v4.version = 4;
    v4.forced_ltr_recovery_frame = 40;
    v4.forced_ltr_reference_frame = 8;
    {
      mlvc::io::MlvcBitstreamWriter writer(roundtrip_path, v4);
      writer.WriteFrame(0, mlvc::codec::MlvcFrameType::kIFrame, 18, {1, 2, 3});
      writer.SwitchConfiguration(2, v4);
      writer.WriteFrame(1, mlvc::codec::MlvcFrameType::kIFrame, 18, {4, 5});
      writer.Close();
    }
    {
      mlvc::io::MlvcBitstreamReader reader(roundtrip_path);
      mlvc::Check(reader.header().forced_ltr_recovery_frame == 40 &&
                      reader.header().forced_ltr_reference_frame == 8,
                  "forced LTR metadata did not survive bitstream header roundtrip");
      int frame_index = -1;
      mlvc::codec::MlvcFrameType frame_type = mlvc::codec::MlvcFrameType::kPFrame;
      int q_index = -1;
      std::vector<uint8_t> payload;
      mlvc::Check(reader.ReadFrame(&frame_index, &frame_type, &q_index, &payload) &&
                      frame_index == 0 && payload == std::vector<uint8_t>({1, 2, 3}),
                  "first MLVC-ES frame did not roundtrip");
      mlvc::Check(reader.ReadFrame(&frame_index, &frame_type, &q_index, &payload) &&
                      frame_index == 1 && payload == std::vector<uint8_t>({4, 5}),
                  "configuration-switch MLVC-ES frame did not roundtrip");
      mlvc::Check(!reader.ReadFrame(&frame_index, &frame_type, &q_index, &payload),
                  "MLVC-ES reader did not reach end of stream");
    }

    ExpectReject(
        [&] {
          mlvc::io::MlvcBitstreamWriter writer(
              std::filesystem::temp_directory_path() / "mlvc_writer_gap.mlvc", v4);
          writer.WriteFrame(1, mlvc::codec::MlvcFrameType::kIFrame, 18, {1});
        },
        "writer accepts non-zero first frame index");
    ExpectReject(
        [&] {
          mlvc::io::MlvcBitstreamWriter writer(
              std::filesystem::temp_directory_path() / "mlvc_writer_first_p.mlvc", v4);
          writer.WriteFrame(0, mlvc::codec::MlvcFrameType::kPFrame, 18, {1});
        },
        "writer accepts non-I first frame");

    const auto pts_path = std::filesystem::temp_directory_path() / "mlvc_writer_pts.mlvc";
    {
      auto pts_header = v4;
      pts_header.fps = 25.0;
      mlvc::io::MlvcBitstreamWriter writer(pts_path, pts_header);
      writer.WriteFrame(0, mlvc::codec::MlvcFrameType::kIFrame, 18, {1});
      writer.WriteFrame(1, mlvc::codec::MlvcFrameType::kIFrame, 18, {2});
      writer.Close();
    }
    {
      mlvc::io::MlvcBitstreamReader reader(pts_path);
      int frame_index = -1;
      mlvc::codec::MlvcFrameType frame_type = mlvc::codec::MlvcFrameType::kPFrame;
      int q_index = -1;
      std::vector<uint8_t> payload;
      mlvc::Check(reader.ReadFrame(&frame_index, &frame_type, &q_index, &payload) &&
                      reader.last_frame_metadata().pts == 0,
                  "first MLVC-ES PTS is not zero");
      mlvc::Check(reader.ReadFrame(&frame_index, &frame_type, &q_index, &payload) &&
                      reader.last_frame_metadata().pts == 3600,
                  "default MLVC-ES PTS does not follow the configured frame rate");
    }

    ExpectReject(
        [&] {
          mlvc::io::MlvcBitstreamWriter writer(
              std::filesystem::temp_directory_path() / "mlvc_writer_missing_refs.mlvc", v4);
          writer.WriteFrame(0, mlvc::codec::MlvcFrameType::kIFrame, 18, {1});
          writer.WriteFrame(1, mlvc::codec::MlvcFrameType::kPFrame, 18, {2});
        },
        "writer infers inter-frame reference metadata");

    const auto corrupt_path = std::filesystem::temp_directory_path() / "mlvc_corrupt_payload.mlvc";
    std::filesystem::copy_file(roundtrip_path, corrupt_path,
                               std::filesystem::copy_options::overwrite_existing);
    {
      std::fstream corrupt(corrupt_path, std::ios::binary | std::ios::in | std::ios::out);
      corrupt.seekp(kFramePayloadSizeOffset);
      const uint64_t oversized = std::numeric_limits<uint64_t>::max();
      corrupt.write(reinterpret_cast<const char*>(&oversized), sizeof(oversized));
    }
    ExpectReaderReject(corrupt_path, "oversized corrupted payload");

    const auto truncated_path = std::filesystem::temp_directory_path() / "mlvc_truncated_payload.mlvc";
    std::filesystem::copy_file(roundtrip_path, truncated_path,
                               std::filesystem::copy_options::overwrite_existing);
    std::filesystem::resize_file(truncated_path, std::filesystem::file_size(truncated_path) - 1);
    ExpectReaderReject(truncated_path, "truncated payload");

    // The MLVC-ES reader must use the SCU frame limit before allocating an
    // EFU buffer, even when the claimed unit length is otherwise protocol-valid.
    const auto oversized_efu_path =
        std::filesystem::temp_directory_path() / "mlvc_oversized_efu_before_allocation.mlvc";
    std::filesystem::copy_file(roundtrip_path, oversized_efu_path,
                               std::filesystem::copy_options::overwrite_existing);
    uint32_t initial_scu_length = 0;
    {
      std::ifstream input(oversized_efu_path, std::ios::binary);
      input.seekg(20);
      unsigned char bytes[4]{};
      input.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
      mlvc::Check(input.good(), "failed to read MLVC-ES SCU length in test");
      initial_scu_length = (static_cast<uint32_t>(bytes[0]) << 24) |
                           (static_cast<uint32_t>(bytes[1]) << 16) |
                           (static_cast<uint32_t>(bytes[2]) << 8) | bytes[3];
    }
    {
      std::fstream corrupt(oversized_efu_path, std::ios::binary | std::ios::in | std::ios::out);
      const std::streamoff efu_offset = 16 + initial_scu_length;
      const uint32_t oversized_payload = 16u * 1024u * 1024u + 1u;
      WriteBigEndian32(&corrupt, efu_offset + 4, 52u + oversized_payload);
      WriteBigEndian32(&corrupt, efu_offset + 40, oversized_payload);
    }
    ExpectReaderReject(oversized_efu_path, "MLVC-ES payload above the SCU resolution limit");

    const auto optional_unit_path =
        std::filesystem::temp_directory_path() / "mlvc_optional_media_unit.mlvc";
    {
      std::ifstream input(roundtrip_path, std::ios::binary);
      std::vector<char> contents((std::istreambuf_iterator<char>(input)),
                                 std::istreambuf_iterator<char>());
      std::vector<char> extension(20, 0);
      extension[0] = 1;
      extension[1] = 0x05;
      extension[3] = 20;
      extension[7] = 20;
      contents.insert(contents.begin() + 16 + initial_scu_length, extension.begin(),
                      extension.end());
      std::ofstream output(optional_unit_path, std::ios::binary | std::ios::trunc);
      output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
      mlvc::Check(output.good(), "failed to write optional media-unit test file");
    }
    {
      mlvc::io::MlvcBitstreamReader reader(optional_unit_path);
      int frame_index = -1;
      mlvc::codec::MlvcFrameType frame_type = mlvc::codec::MlvcFrameType::kPFrame;
      int q_index = -1;
      std::vector<uint8_t> payload;
      mlvc::Check(reader.ReadFrame(&frame_index, &frame_type, &q_index, &payload) &&
                      frame_index == 0,
                  "reader did not skip an unknown optional media unit");
    }

    const auto frame_gap_path = std::filesystem::temp_directory_path() / "mlvc_frame_gap.mlvc";
    std::filesystem::copy_file(roundtrip_path, frame_gap_path,
                               std::filesystem::copy_options::overwrite_existing);
    {
      std::fstream corrupt(frame_gap_path, std::ios::binary | std::ios::in | std::ios::out);
      corrupt.seekp(kV4HeaderBytes);
      const int32_t wrong_index = 1;
      corrupt.write(reinterpret_cast<const char*>(&wrong_index), sizeof(wrong_index));
    }
    ExpectReaderReject(frame_gap_path, "non-zero first frame index");

    const auto header_only = mlvc::io::ResolveForcedLtrFrames(v4, -1, -1);
    mlvc::Check(header_only.recovery_frame == 40 && header_only.reference_frame == 8,
                "decoder did not use forced LTR metadata from the v4 header");
    const auto matching = mlvc::io::ResolveForcedLtrFrames(v4, 40, 8);
    mlvc::Check(matching.recovery_frame == 40 && matching.reference_frame == 8,
                "matching decoder forced LTR configuration was not accepted");

    ExpectReject(
        [&] { mlvc::codec::ValidateForcedLtrConfiguration(96, 8, 16, 40, -1); },
        "unpaired forced LTR configuration");
    ExpectReject(
        [&] { mlvc::codec::ValidateForcedLtrConfiguration(96, 8, 16, 40, 40); },
        "LTR reference not before recovery");
    ExpectReject(
        [&] { mlvc::codec::ValidateForcedLtrConfiguration(96, 8, 16, 104, 8); },
        "cross-GOP LTR reference");
    ExpectReject(
        [&] { mlvc::codec::ValidateForcedLtrConfiguration(96, 8, 16, 96, 8); },
        "I-frame forced LTR recovery");
    ExpectReject(
        [&] { mlvc::codec::ValidateForcedLtrConfiguration(96, 8, 16, 40, 9); },
        "reference frame that is not an LTR capture");
    mlvc::codec::ValidateForcedLtrConfiguration(96, 8, 16, 40, 8);

    ExpectReject(
        [&] {
          auto invalid = header;
          invalid.forced_ltr_recovery_frame = 40;
          invalid.forced_ltr_reference_frame = 8;
          mlvc::io::ValidateMlvcBitstreamHeader(invalid);
        },
        "forced LTR metadata in a pre-v4 header");
    ExpectReject([&] { mlvc::io::ResolveForcedLtrFrames(v4, 40, -1); },
                 "unpaired decoder override");
    ExpectReject([&] { mlvc::io::ResolveForcedLtrFrames(v4, 40, 16); },
                 "decoder override conflicting with the header");

    const uint64_t payload_limit = mlvc::io::MaxMlvcFramePayloadBytes(header.width, header.height);
    mlvc::Check(payload_limit == 16ULL * 1024ULL * 1024ULL, "1080p payload limit is incorrect");
    mlvc::io::ValidateMlvcFrameMetadata(0, mlvc::codec::MlvcFrameType::kIFrame, 18, 0, true);
    mlvc::io::ValidateMlvcFrameMetadata(1, mlvc::codec::MlvcFrameType::kPFrame, 20, 1, false);
    ExpectReject(
        [&] {
          mlvc::io::ValidateMlvcFrameMetadata(2, mlvc::codec::MlvcFrameType::kPFrame, 20, 1, false);
        },
        "frame number gap");
    ExpectReject(
        [&] {
          mlvc::io::ValidateMlvcFrameMetadata(0, mlvc::codec::MlvcFrameType::kPFrame, 18, 0, true);
        },
        "first frame type");
    ExpectReject(
        [&] {
          mlvc::io::ValidateMlvcFrameMetadata(1, static_cast<mlvc::codec::MlvcFrameType>(99), 18, 1,
                                              false);
        },
        "invalid frame type");
    ExpectReject(
        [&] {
          mlvc::io::ValidateMlvcFrameMetadata(1, mlvc::codec::MlvcFrameType::kPFrame, 64, 1, false);
        },
        "frame qp");
    mlvc::io::ValidateMlvcQIndexForSidecar(31, 32);
    ExpectReject([&] { mlvc::io::ValidateMlvcQIndexForSidecar(32, 32); },
                 "sidecar Q index range");

    std::filesystem::remove(roundtrip_path);
    std::filesystem::remove(corrupt_path);
    std::filesystem::remove(truncated_path);
    std::filesystem::remove(frame_gap_path);
    std::filesystem::remove(oversized_efu_path);
    std::filesystem::remove(optional_unit_path);

    std::cout << "mlvc bitstream validation test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_mlvc_bitstream_validation failed: " << error.what() << "\n";
    return 1;
  }
}
