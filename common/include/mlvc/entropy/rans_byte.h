#ifndef MLVC_ENTROPY_RANS_BYTE_H_
#define MLVC_ENTROPY_RANS_BYTE_H_

#include <cstdint>

#ifdef assert
#define MLVC_RANS_ASSERT assert
#else
#define MLVC_RANS_ASSERT(x)
#endif

namespace mlvc {

constexpr int kRansScaleBits = 16;
constexpr int kRansShiftBits = 23;
constexpr uint32_t kRansByteL = (1u << kRansShiftBits);
constexpr int kRansEncRenormShiftBits = kRansShiftBits - kRansScaleBits + 8;
constexpr int kRansDecMask = ((1u << kRansScaleBits) - 1);

using RansState = uint32_t;

inline void RansEncInit(RansState& state) { state = kRansByteL; }

inline void RansEncRenorm(RansState& state, uint8_t*& ptr, uint32_t freq) {
  const uint32_t x_max = freq << kRansEncRenormShiftBits;
  while (state >= x_max) {
    *(--ptr) = static_cast<uint8_t>(state & 0xff);
    state >>= 8;
  }
}

inline void RansEncPut(RansState& state, uint8_t*& ptr, uint32_t start, uint32_t freq) {
  RansEncRenorm(state, ptr, freq);
  state = ((state / freq) << kRansScaleBits) + (state % freq) + start;
}

inline void RansEncFlush(const RansState& state, uint8_t*& ptr) {
  ptr -= 4;
  ptr[0] = static_cast<uint8_t>(state >> 0);
  ptr[1] = static_cast<uint8_t>(state >> 8);
  ptr[2] = static_cast<uint8_t>(state >> 16);
  ptr[3] = static_cast<uint8_t>(state >> 24);
}

inline void RansDecInit(RansState& state, uint8_t*& ptr) {
  state = (*ptr++) << 0;
  state |= (*ptr++) << 8;
  state |= (*ptr++) << 16;
  state |= (*ptr++) << 24;
}

inline uint32_t RansDecGet(RansState& state) { return state & kRansDecMask; }

inline void RansDecAdvance(RansState& state, uint8_t*& ptr, uint32_t start, uint32_t freq) {
  state = freq * (state >> kRansScaleBits) + (state & kRansDecMask) - start;
  while (state < kRansByteL) {
    state = (state << 8) | *ptr++;
  }
}

}  // namespace mlvc

#endif  // MLVC_ENTROPY_RANS_BYTE_H_
