#include <mlvc/codec/mlvc_entropy.h>
#include <mlvc/core/status.h>

#include <iostream>

int main() {
  try {
    constexpr int kGop = 96;
    constexpr int kResetInterval = 32;
    mlvc::Check(mlvc::codec::ShouldResetReferenceFeature(0, kGop, kResetInterval),
                "the first I frame must reset the reference feature");
    mlvc::Check(mlvc::codec::ShouldResetReferenceFeature(31, kGop, kResetInterval),
                "a reset-interval boundary P frame must reset the reference feature");
    mlvc::Check(!mlvc::codec::ShouldResetReferenceFeature(32, kGop, kResetInterval),
                "a normal P frame must preserve the reference feature");
    mlvc::Check(mlvc::codec::ShouldResetReferenceFeature(63, kGop, kResetInterval),
                "each reset-interval boundary P frame must reset the reference feature");
    mlvc::Check(mlvc::codec::ShouldResetReferenceFeature(96, kGop, kResetInterval),
                "a GOP-boundary I frame must reset the reference feature");
    std::cout << "reference reset schedule test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_reference_reset_schedule failed: " << error.what() << "\n";
    return 1;
  }
}
