#include <mlvc/application/pipeline/ordered_future_window.h>
#include <mlvc/application/stream/mlvc_entropy_decode.h>
#include <mlvc/core/status.h>
#include <mlvc/entropy/mlvc_official_entropy.h>
#include <mlvc/framework/entropy_worker.h>
#include <mlvc/framework/profiler.h>
#include <mlvc/io/mlvc_bitstream.h>
#include <mlvc/runtime/model_manifest.h>

#include <chrono>
#include <cstddef>
#include <deque>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Packet {
  int frame_index = -1;
  mlvc::codec::MlvcFrameType frame_type = mlvc::codec::MlvcFrameType::kPFrame;
  int q_index = 0;
  std::vector<uint8_t> payload;
};

void CheckTensorEqual(const mlvc::codec::TensorData& expected,
                      const mlvc::codec::TensorData& actual, const std::string& name,
                      int frame_index) {
  const std::string prefix = "frame " + std::to_string(frame_index) + " " + name;
  mlvc::Check(expected.shape.dims() == actual.shape.dims(), prefix + " shape mismatch");
  mlvc::Check(expected.dtype == actual.dtype, prefix + " dtype mismatch");
  mlvc::Check(expected.bytes == actual.bytes, prefix + " byte mismatch");
}

void CheckFrameEqual(const mlvc::codec::DecodedEntropyFrame& expected,
                     const mlvc::codec::DecodedEntropyFrame& actual) {
  mlvc::Check(expected.frame_index == actual.frame_index, "entropy frame index mismatch");
  mlvc::Check(expected.frame_type == actual.frame_type, "entropy frame type mismatch");
  mlvc::Check(expected.q_index == actual.q_index, "entropy q_index mismatch");
  CheckTensorEqual(expected.z_raw, actual.z_raw, "z_raw", expected.frame_index);
  CheckTensorEqual(expected.y_raw_0, actual.y_raw_0, "y_raw_0", expected.frame_index);
  CheckTensorEqual(expected.y_raw_1, actual.y_raw_1, "y_raw_1", expected.frame_index);
}

std::vector<Packet> ReadPackets(const std::filesystem::path& bitstream_path, int frame_limit) {
  mlvc::io::MlvcBitstreamReader reader(bitstream_path);
  std::vector<Packet> packets;
  while (frame_limit <= 0 || static_cast<int>(packets.size()) < frame_limit) {
    Packet packet;
    if (!reader.ReadFrame(&packet.frame_index, &packet.frame_type, &packet.q_index,
                          &packet.payload)) {
      break;
    }
    packets.push_back(std::move(packet));
  }
  mlvc::Check(!packets.empty(), "entropy equivalence test received no bitstream frames");
  return packets;
}

mlvc::codec::DecodedEntropyFrame DecodePacket(
    mlvc::MlvcOfficialEntropyDecoder* decoder, const mlvc::ModelRecord& decoder_record,
    const Packet& packet, mlvc::Profiler* profiler) {
  return mlvc::codec::DecodeMlvcEntropyFrame(decoder, decoder_record, packet.frame_index,
                                              packet.frame_type, packet.q_index, packet.payload,
                                              profiler);
}

void CheckParallelEquivalence(const std::filesystem::path& model_directory,
                              const mlvc::ModelRecord& decoder_record,
                              const std::vector<Packet>& packets) {
  mlvc::MlvcOfficialEntropyDecoder sequential_decoder(model_directory);
  std::vector<std::unique_ptr<mlvc::MlvcOfficialEntropyDecoder>> parallel_decoders;
  parallel_decoders.emplace_back(
      std::make_unique<mlvc::MlvcOfficialEntropyDecoder>(model_directory));
  parallel_decoders.emplace_back(
      std::make_unique<mlvc::MlvcOfficialEntropyDecoder>(model_directory));
  mlvc::EntropyWorker worker(2);
  mlvc::Profiler profiler;
  mlvc::app::OrderedFutureWindow<mlvc::codec::DecodedEntropyFrame> window(2);
  std::deque<mlvc::codec::DecodedEntropyFrame> expected;

  for (const Packet& packet : packets) {
    expected.push_back(DecodePacket(&sequential_decoder, decoder_record, packet, &profiler));
    if (window.full()) {
      CheckFrameEqual(expected.front(), window.PopFront());
      expected.pop_front();
    }
    window.SubmitNext([&, packet](std::size_t slot) {
      return worker.SubmitValue<mlvc::codec::DecodedEntropyFrame>([&, slot, packet] {
        return DecodePacket(parallel_decoders.at(slot).get(), decoder_record, packet, &profiler);
      });
    });
  }
  while (!window.empty()) {
    CheckFrameEqual(expected.front(), window.PopFront());
    expected.pop_front();
  }
  mlvc::Check(expected.empty(), "parallel entropy results were not fully compared");
}

void CheckDetailedProfileEvents(const std::filesystem::path& model_directory,
                                const mlvc::ModelRecord& decoder_record,
                                const Packet& packet) {
  mlvc::MlvcOfficialEntropyDecoder decoder(model_directory);
  mlvc::Profiler profiler;
  (void)DecodePacket(&decoder, decoder_record, packet, &profiler);

  std::set<std::string> names;
  for (const mlvc::ProfileEvent& event : profiler.events()) {
    names.insert(event.name);
  }
  const std::vector<std::string> required = {
      "entropy.payload_copy",       "entropy.stream_open",
      "entropy.z.index_build",      "entropy.z.output_allocate",
      "entropy.z.rans_decode",      "entropy.z.int32_to_int8",
      "entropy.z.int8_to_fp16",     "entropy.scale_index_expand",
      "entropy.y0.index_convert",   "entropy.y0.output_allocate",
      "entropy.y0.rans_decode",     "entropy.y0.int32_to_int8",
      "entropy.y0.int8_to_fp16",    "entropy.y1.index_convert",
      "entropy.y1.output_allocate", "entropy.y1.rans_decode",
      "entropy.y1.stream_finalize", "entropy.y1.int32_to_int8",
      "entropy.y1.int8_to_fp16",
  };
  for (const std::string& name : required) {
    mlvc::Check(names.count(name) == 1, "missing entropy profile event: " + name);
  }
}

double MeasureSequential(const std::filesystem::path& model_directory,
                         const mlvc::ModelRecord& decoder_record,
                         const std::vector<Packet>& packets) {
  mlvc::MlvcOfficialEntropyDecoder decoder(model_directory);
  mlvc::Profiler profiler;
  const auto begin = std::chrono::steady_clock::now();
  for (const Packet& packet : packets) {
    (void)DecodePacket(&decoder, decoder_record, packet, &profiler);
  }
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
}

double MeasureParallel(const std::filesystem::path& model_directory,
                       const mlvc::ModelRecord& decoder_record,
                       const std::vector<Packet>& packets) {
  std::vector<std::unique_ptr<mlvc::MlvcOfficialEntropyDecoder>> decoders;
  decoders.emplace_back(std::make_unique<mlvc::MlvcOfficialEntropyDecoder>(model_directory));
  decoders.emplace_back(std::make_unique<mlvc::MlvcOfficialEntropyDecoder>(model_directory));
  mlvc::EntropyWorker worker(2);
  mlvc::Profiler profiler;
  mlvc::app::OrderedFutureWindow<mlvc::codec::DecodedEntropyFrame> window(2);
  const auto begin = std::chrono::steady_clock::now();
  for (const Packet& packet : packets) {
    if (window.full()) {
      (void)window.PopFront();
    }
    window.SubmitNext([&, packet](std::size_t slot) {
      return worker.SubmitValue<mlvc::codec::DecodedEntropyFrame>([&, slot, packet] {
        return DecodePacket(decoders.at(slot).get(), decoder_record, packet, &profiler);
      });
    });
  }
  while (!window.empty()) {
    (void)window.PopFront();
  }
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    mlvc::Check(argc == 3 || argc == 4,
                "usage: test_entropy_decode_equivalence <manifest> <bitstream> [frames]");
    const int frame_limit = argc == 4 ? std::stoi(argv[3]) : 32;
    mlvc::Check(frame_limit > 0, "entropy equivalence frame count must be positive");
    const mlvc::ModelManifest manifest = mlvc::ModelManifest::Load(argv[1]);
    const mlvc::ModelRecord& decoder_record = manifest.GetModel("MLVCDecoder");
    const std::vector<Packet> packets = ReadPackets(argv[2], frame_limit);

    CheckDetailedProfileEvents(manifest.directory(), decoder_record, packets.front());
    CheckParallelEquivalence(manifest.directory(), decoder_record, packets);
    const double sequential_seconds = MeasureSequential(manifest.directory(), decoder_record,
                                                        packets);
    const double parallel_seconds = MeasureParallel(manifest.directory(), decoder_record, packets);
    const double frames = static_cast<double>(packets.size());
    std::cout << "frames=" << packets.size() << "\n";
    std::cout << "sequential_entropy_fps=" << frames / sequential_seconds << "\n";
    std::cout << "parallel_entropy_fps=" << frames / parallel_seconds << "\n";
    std::cout << "entropy_outputs=byte_identical\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_entropy_decode_equivalence failed: " << error.what() << "\n";
    return 1;
  }
}
