#include <mlvc/motion/translation_math.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void Check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Operation>
void MustThrow(Operation operation) {
  bool threw = false;
  try { operation(); } catch (const std::exception&) { threw = true; }
  Check(threw, "invalid motion input was accepted");
}

}  // namespace

int main() {
  try {
    using mlvc::motion::EstimateTranslation;
    using mlvc::motion::NearestFeatureCell;
    Check(NearestFeatureCell(0) == 0, "zero displacement");
    for (int cell = 0; cell < 126; ++cell) {
      const double boundary = 8.0 * cell + 4.0;
      Check(NearestFeatureCell(boundary) == cell, "positive half-grid tie must round toward zero");
      Check(NearestFeatureCell(-boundary) == -cell, "negative half-grid tie must round toward zero");
      Check(NearestFeatureCell(boundary + 0.25) == cell + 1, "above positive half-grid");
      Check(NearestFeatureCell(-boundary - 0.25) == -cell - 1, "below negative half-grid");
    }
    Check(NearestFeatureCell(1016) == 127 && NearestFeatureCell(-1024) == -128,
          "signed side information endpoints");
    MustThrow([] { NearestFeatureCell(1024); });
    MustThrow([] { NearestFeatureCell(-1032); });
    MustThrow([] { NearestFeatureCell(std::numeric_limits<double>::quiet_NaN()); });

    // One 16x16 block outweighs three 4x4 blocks; future-reference vectors are excluded.
    auto area = EstimateTranslation({{-1, 4, 4, 0, 0, 4}, {-1, 4, 4, 0, 0, 4},
                                     {-1, 4, 4, 0, 0, 4}, {-1, 16, 16, -64, 32, 4},
                                     {1, 64, 64, 512, -512, 4}}, false);
    Check(area.tx == 16 && area.ty == -8 && area.kx == 2 && area.ky == -1 &&
              area.vector_count == 4, "area weighting, reference filter and vector sign");
    // Even-area ties select the lower sorted value, independently per axis.
    auto ties = EstimateTranslation({{-1, 8, 8, -48, 16, 4}, {-1, 8, 8, -16, -48, 4}}, false);
    Check(ties.tx == 4 && ties.ty == -4 && ties.kx == 0 && ties.ky == 0,
          "left-search median tie or independent axes");
    auto fractional = EstimateTranslation({{-1, 16, 16, -17, 17, 4}}, false);
    Check(fractional.tx == 4.25 && fractional.ty == -4.25 &&
              fractional.kx == 1 && fractional.ky == -1, "quarter-pixel vector scale");
    auto reset = EstimateTranslation({{-1, 16, 16, -1024, 1024, 4}}, true);
    Check(reset.tx == 0 && reset.ty == 0 && reset.kx == 0 && reset.ky == 0 &&
              reset.vector_count == 0, "GOP start must ignore motion");
    Check(EstimateTranslation({}, false).kx == 0, "all-intra P frame has no evidence");
    MustThrow([] { EstimateTranslation({{-1, 8, 8, -4, 4, 0}}, false); });
    MustThrow([] { EstimateTranslation({{-1, 0, 8, -4, 4, 4}}, false); });
    std::cout << "motion_translation status=ok\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "motion_translation status=failed: " << error.what() << '\n';
    return 1;
  }
}
