#ifndef MLVC_CODEC_TRANSLATION_WARP_H_
#define MLVC_CODEC_TRANSLATION_WARP_H_

#include <mlvc/codec/tensor_data.h>
#include <mlvc/core/tensor.h>
#include <mlvc/runtime/model_manifest.h>

#include <utility>

namespace mlvc::codec {

// Copies integer-offset spatial slices on every leading plane. Uncovered
// locations retain their original value. Source storage is never modified.
TensorData ShiftTensorPreserveBoundary(const mlvc::TensorView& source, int kx, int ky);
TensorData ShiftTensorPreserveBoundary(const TensorData& source, int kx, int ky);
// Reuses owned output storage. Every output byte is assigned directly from
// the shifted slice or the original boundary, without clone-then-overwrite.
// Overlapping source/output storage is read from a temporary snapshot.
// Any external output buffer is detached; source lifetime is retained until
// copying finishes. The zero-offset result remains a complete byte copy.
void ShiftTensorPreserveBoundaryInto(const mlvc::TensorView& source, int kx, int ky,
                                    TensorData* output);
void ShiftTensorPreserveBoundaryInto(const TensorData& source, int kx, int ky,
                                    TensorData* output);
std::pair<TensorData, TensorData> ShiftFeatureAndMemory(const TensorData& feature,
                                                      const TensorData& memory, int kx, int ky);

// Only the audited FP16 1080p pair currently has a verified pointwise adaptor
// and concatenated feature/memory input. Fail closed for all other bundles.
void RequireTranslationWarpModels(const mlvc::ModelManifest& manifest);

}  // namespace mlvc::codec

#endif  // MLVC_CODEC_TRANSLATION_WARP_H_
