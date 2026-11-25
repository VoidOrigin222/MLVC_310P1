#include <ulbvc/codec/tensor_data.h>
#include <ulbvc/core/status.h>
#include <ulbvc/io/fp16_yuv444_to_nv12.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {
using ulbvc::io::Nv12Layout;

template <class Function>
void Reject(Function&& function, const char* reason) {
  bool rejected = false;
  try { function(); } catch (const std::exception&) { rejected = true; }
  ulbvc::Check(rejected, std::string("expected rejection: ") + reason);
}

void TestBorrowedFp16MatchesOwnedAndKeepsOwner(bool full_resolution = false) {
  auto owned = ulbvc::codec::MakeFp16Tensor(full_resolution ? std::vector<int64_t>{1, 3, 1088, 1920}
                                                       : std::vector<int64_t>{1, 3, 10, 16}, 0.0f);
  auto* halves = reinterpret_cast<uint16_t*>(owned.bytes.data());
  for (std::size_t i = 0; i < owned.Elements(); ++i) {
    const float value = static_cast<float>(static_cast<int>(i % 31) - 4) / 23.0f;
    halves[i] = static_cast<uint16_t>(ulbvc::codec::FloatToHalfBits(value));
  }
  const Nv12Layout layout = full_resolution ? Nv12Layout{1920, 1080, 1920, 1080}
                                           : Nv12Layout{14, 8, 20, 10};
  std::vector<uint8_t> expected;
  ulbvc::io::ConvertFp16Yuv444ToNv12Scalar(owned, layout, &expected);
  for (const auto location : {ulbvc::MemoryLocation::kCpu, ulbvc::MemoryLocation::kPinnedCpu}) {
    auto owner = std::make_shared<std::vector<uint8_t>>(owned.bytes);
    std::weak_ptr<std::vector<uint8_t>> weak = owner;
    ulbvc::codec::TensorData borrowed;
    borrowed.shape = owned.shape;
    borrowed.dtype = owned.dtype;
    // Exercise converter input views directly. AttachExternalBuffer deliberately
    // disallows CPU leases, while TensorData can represent a borrowed CPU view.
    borrowed.external_data = owner->data();
    borrowed.external_bytes = owner->size();
    borrowed.external_location = location;
    borrowed.external_owner = owner;
    owner.reset();
    std::vector<uint8_t> scalar, optimized;
    ulbvc::io::ConvertFp16Yuv444ToNv12Scalar(borrowed, layout, &scalar);
    ulbvc::io::ConvertFp16Yuv444ToNv12(borrowed, layout, &optimized);
    ulbvc::Check(scalar == expected && optimized == expected,
                "borrowed FP16 conversion differs from owned scalar bytes");
    ulbvc::Check(!weak.expired() && borrowed.has_external_buffer(), "conversion consumed source ownership");
    ulbvc::Check(weak.lock()->size() == owned.bytes.size() && *weak.lock() == owned.bytes,
                "conversion mutated the borrowed source");
    borrowed.ClearExternalBuffer();
    ulbvc::Check(weak.expired(), "borrowed owner was leaked after release");
  }
  auto short_owned = owned;
  short_owned.bytes.pop_back();
  auto check_both = [&](const ulbvc::codec::TensorData& bad, const char* reason) {
    Reject([&] { ulbvc::io::ConvertFp16Yuv444ToNv12Scalar(bad, layout, &expected); }, reason);
    Reject([&] { ulbvc::io::ConvertFp16Yuv444ToNv12(bad, layout, &expected); }, reason);
  };
  check_both(short_owned, "short owned FP16 buffer");
  ulbvc::codec::TensorData short_borrowed;
  short_borrowed.shape = owned.shape;
  short_borrowed.dtype = owned.dtype;
  short_borrowed.external_data = owned.bytes.data();
  short_borrowed.external_bytes = owned.bytes.size() - 2;
  short_borrowed.external_location = ulbvc::MemoryLocation::kCpu;
  check_both(short_borrowed, "short borrowed FP16 buffer");
  ulbvc::codec::TensorData device;
  device.shape = owned.shape;
  device.dtype = owned.dtype;
  device.external_data = owned.bytes.data();
  device.external_bytes = owned.bytes.size();
  device.external_location = ulbvc::MemoryLocation::kAcl;
  check_both(device, "device FP16 buffer must not be dereferenced by CPU conversion");
  auto wrong_type = owned;
  wrong_type.dtype = ulbvc::DataType::kInt8;
  check_both(wrong_type, "non-FP16 input");
}
}  // namespace

int main() {
  try {
    TestBorrowedFp16MatchesOwnedAndKeepsOwner();
    TestBorrowedFp16MatchesOwnedAndKeepsOwner(true);
    std::cout << "full-resolution and borrowed FP16 motion input tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_motion_nv12_inputs failed: " << error.what() << '\n';
    return 1;
  }
}
