#include <mlvc/codec/mlvc_entropy.h>
#include <mlvc/core/status.h>
#include <mlvc/io/mlvc_bitstream.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

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
    mlvc::io::ValidateMlvcBitstreamHeader(header);

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
      writer.Close();
    }
    mlvc::io::MlvcBitstreamReader reader(roundtrip_path);
    mlvc::Check(reader.header().forced_ltr_recovery_frame == 40 &&
                    reader.header().forced_ltr_reference_frame == 8,
                "forced LTR metadata did not survive bitstream header roundtrip");

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

    std::cout << "mlvc bitstream validation test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_mlvc_bitstream_validation failed: " << error.what() << "\n";
    return 1;
  }
}
