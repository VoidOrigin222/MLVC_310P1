#include <mlvc/application/runtime/mlvc_codec_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

#include "mlvc/core/status.h"

namespace mlvc::app {

namespace {

constexpr std::array<uint32_t, 64> kSha256K = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
    0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
    0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
    0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

uint32_t RotR(uint32_t value, unsigned count) {
  return (value >> count) | (value << (32u - count));
}

void Sha256Block(const uint8_t* block, std::array<uint32_t, 8>* state) {
  std::array<uint32_t, 64> words{};
  for (std::size_t i = 0; i < 16; ++i) {
    words[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
               (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
               (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
               static_cast<uint32_t>(block[i * 4 + 3]);
  }
  for (std::size_t i = 16; i < words.size(); ++i) {
    const uint32_t s0 = RotR(words[i - 15], 7) ^ RotR(words[i - 15], 18) ^ (words[i - 15] >> 3);
    const uint32_t s1 = RotR(words[i - 2], 17) ^ RotR(words[i - 2], 19) ^ (words[i - 2] >> 10);
    words[i] = words[i - 16] + s0 + words[i - 7] + s1;
  }
  uint32_t a = (*state)[0], b = (*state)[1], c = (*state)[2], d = (*state)[3];
  uint32_t e = (*state)[4], f = (*state)[5], g = (*state)[6], h = (*state)[7];
  for (std::size_t i = 0; i < words.size(); ++i) {
    const uint32_t temp1 = h + (RotR(e, 6) ^ RotR(e, 11) ^ RotR(e, 25)) +
                           ((e & f) ^ (~e & g)) + kSha256K[i] + words[i];
    const uint32_t temp2 = (RotR(a, 2) ^ RotR(a, 13) ^ RotR(a, 22)) +
                           ((a & b) ^ (a & c) ^ (b & c));
    h = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
  }
  (*state)[0] += a; (*state)[1] += b; (*state)[2] += c; (*state)[3] += d;
  (*state)[4] += e; (*state)[5] += f; (*state)[6] += g; (*state)[7] += h;
}

std::string Sha256File(const std::filesystem::path& path, uint64_t expected_bytes,
                       const std::string& label) {
  std::ifstream input(path, std::ios::binary);
  mlvc::Check(input.good(), "failed to open " + label + " for integrity validation: " +
                                path.string());
  mlvc::Check(expected_bytes <= std::numeric_limits<uint64_t>::max() / 8,
             label + " size overflows SHA-256 length field: " + path.string());

  std::array<uint32_t, 8> state = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  std::array<uint8_t, 64> block{};
  std::array<uint8_t, 64 * 1024> input_buffer{};
  std::size_t block_size = 0;
  uint64_t bytes_read = 0;
  while (input) {
    input.read(reinterpret_cast<char*>(input_buffer.data()),
               static_cast<std::streamsize>(input_buffer.size()));
    const std::streamsize count = input.gcount();
    mlvc::Check(count >= 0, "failed to read " + label + " for integrity validation: " +
                              path.string());
    if (count == 0) {
      break;
    }
    const uint64_t chunk_size = static_cast<uint64_t>(count);
    mlvc::Check(bytes_read <= std::numeric_limits<uint64_t>::max() - chunk_size,
               label + " size overflows during integrity validation: " + path.string());
    bytes_read += chunk_size;

    std::size_t offset = 0;
    while (offset < static_cast<std::size_t>(count)) {
      const std::size_t copied = std::min(block.size() - block_size,
                                          static_cast<std::size_t>(count) - offset);
      std::copy_n(input_buffer.data() + offset, copied, block.data() + block_size);
      block_size += copied;
      offset += copied;
      if (block_size == block.size()) {
        Sha256Block(block.data(), &state);
        block_size = 0;
      }
    }
  }
  mlvc::Check(!input.bad(), "failed to read " + label + " for integrity validation: " +
                             path.string());
  mlvc::Check(bytes_read == expected_bytes,
             label + " size changed during integrity validation: " + path.string());

  block[block_size++] = 0x80;
  if (block_size > 56) {
    std::fill(block.begin() + block_size, block.end(), 0);
    Sha256Block(block.data(), &state);
    block_size = 0;
  }
  std::fill(block.begin() + block_size, block.begin() + 56, 0);
  const uint64_t bit_count = expected_bytes * 8;
  for (unsigned i = 0; i < 8; ++i) {
    block[56 + i] = static_cast<uint8_t>(bit_count >> (56 - i * 8));
  }
  Sha256Block(block.data(), &state);

  std::ostringstream digest;
  digest << std::hex << std::setfill('0');
  for (uint32_t word : state) digest << std::setw(8) << word;
  return digest.str();
}

void VerifyFile(const std::filesystem::path& path, uint64_t expected_bytes,
                const std::string& expected_sha256, const std::string& label) {
  std::error_code error;
  mlvc::Check(std::filesystem::is_regular_file(path, error) && !error,
              "manifest " + label + " is not a regular file: " + path.string());
  error.clear();
  const uintmax_t actual_bytes = std::filesystem::file_size(path, error);
  mlvc::Check(!error, "failed to stat manifest " + label + ": " + path.string());
  mlvc::Check(actual_bytes == expected_bytes,
              "manifest " + label + " byte count mismatch: " + path.string());
  mlvc::Check(Sha256File(path, expected_bytes, label) == expected_sha256,
              "manifest " + label + " sha256 mismatch: " + path.string());
}

void VerifyManifestFiles(const mlvc::ModelManifest& manifest) {
  mlvc::VerifyRuntimeArtifacts(manifest);
  const auto& sidecar = manifest.sidecar();
  const std::filesystem::path sidecar_path = sidecar.file.is_absolute()
                                                ? sidecar.file
                                                : manifest.directory() / sidecar.file;
  VerifyFile(sidecar_path, sidecar.bytes, sidecar.sha256, "sidecar");
  for (const auto& model : manifest.models()) {
    const std::filesystem::path model_path = model.model.is_absolute()
                                                 ? model.model
                                                 : manifest.directory() / model.model;
    VerifyFile(model_path, model.bytes, model.sha256,
               "model " + model.name);
  }
}

std::filesystem::path RuntimeArtifactPath(const mlvc::ModelManifest& manifest,
                                          const std::string& name) {
  const auto& artifact = manifest.GetRuntimeArtifact(name);
  return artifact.file.is_absolute() ? artifact.file : manifest.directory() / artifact.file;
}

}  // namespace

mlvc::ModelManifest MlvcCodecRuntime::LoadVerifiedManifest(
    const std::filesystem::path& manifest_path) {
  auto manifest = mlvc::ModelManifest::Load(manifest_path);
  VerifyManifestFiles(manifest);
  return manifest;
}

MlvcCodecRuntime::MlvcCodecRuntime(const std::filesystem::path& manifest_path, int device,
                                   mlvc::codec::StageOutputBindingMode binding_mode)
    : runtime_(device),
      models_(&runtime_, LoadVerifiedManifest(manifest_path)),
      sidecar_(mlvc::RuntimeSidecar::LoadQpShiftMetadata(
          RuntimeArtifactPath(models_.manifest(), "metadata.json"))),
      workspace_(&models_, binding_mode) {}

}  // namespace mlvc::app
