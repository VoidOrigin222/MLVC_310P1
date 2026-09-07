#ifndef MLVC_CODEC_DETAIL_QSCALE_CACHE_H_
#define MLVC_CODEC_DETAIL_QSCALE_CACHE_H_

#include <mlvc/codec/tensor_data.h>
#include <mlvc/core/status.h>
#include <mlvc/core/tensor_handle.h>
#include <mlvc/entropy/sidecar.h>
#include <mlvc/framework/profiler.h>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <mlvc/codec/detail/stage/constants.h>

namespace mlvc::codec {

TensorData MakeQScaleTensor(const mlvc::RuntimeSidecar& sidecar, const std::string& name, int qp,
                            const std::vector<int64_t>& shape);

class QScaleCache {
 public:
  explicit QScaleCache(const mlvc::RuntimeSidecar& sidecar)
      : sidecar_(sidecar),
        i_encoder_q_step_{MakeQScaleTensor(sidecar_, "i_encoder_q_scale", kBaseQp, {1, 368, 1, 1}),
                          std::nullopt},
        i_decoder_q_step_{MakeQScaleTensor(sidecar_, "i_decoder_q_scale", kBaseQp, {1, 368, 1, 1}),
                          std::nullopt} {}

  const TensorData& IEncoder() const { return i_encoder_q_step_.tensor; }
  const mlvc::TensorHandle* IEncoderHandle() const { return HandleOrNull(i_encoder_q_step_); }

  const TensorData& IDecoder(int qp) {
    if (qp == kBaseQp) {
      return i_decoder_q_step_.tensor;
    }
    return Cached(&i_decoder_q_steps_, "i_decoder_q_scale", qp, {1, 368, 1, 1}).tensor;
  }

  const mlvc::TensorHandle* IDecoderHandle(int qp) {
    if (qp == kBaseQp) {
      return HandleOrNull(i_decoder_q_step_);
    }
    return HandleOrNull(Cached(&i_decoder_q_steps_, "i_decoder_q_scale", qp, {1, 368, 1, 1}));
  }

  const TensorData& PFeature(int qp) {
    return Cached(&p_feature_q_steps_, "p_feature_q_scale", qp, {1, 256, 1, 1}).tensor;
  }

  const mlvc::TensorHandle* PFeatureHandle(int qp) {
    return HandleOrNull(Cached(&p_feature_q_steps_, "p_feature_q_scale", qp, {1, 256, 1, 1}));
  }

  const TensorData& PEncoder(int qp) {
    return Cached(&p_encoder_q_steps_, "p_encoder_q_scale", qp, {1, 256, 1, 1}).tensor;
  }

  const mlvc::TensorHandle* PEncoderHandle(int qp) {
    return HandleOrNull(Cached(&p_encoder_q_steps_, "p_encoder_q_scale", qp, {1, 256, 1, 1}));
  }

  const TensorData& PDecoder(int qp) {
    return Cached(&p_decoder_q_steps_, "p_decoder_q_scale", qp, {1, 256, 1, 1}).tensor;
  }

  const mlvc::TensorHandle* PDecoderHandle(int qp) {
    return HandleOrNull(Cached(&p_decoder_q_steps_, "p_decoder_q_scale", qp, {1, 256, 1, 1}));
  }

  const TensorData& PReconstruction(int qp) {
    return Cached(&p_reconstruction_q_steps_, "p_reconstruction_q_scale", qp, {1, 320, 1, 1})
        .tensor;
  }

  const mlvc::TensorHandle* PReconstructionHandle(int qp) {
    return HandleOrNull(
        Cached(&p_reconstruction_q_steps_, "p_reconstruction_q_scale", qp, {1, 320, 1, 1}));
  }

  void PreloadAllRows() {
    for (int qp = 0; qp < QpCount("i_encoder_q_scale"); ++qp) {
      if (qp != kBaseQp) {
        (void)Cached(&i_encoder_q_steps_, "i_encoder_q_scale", qp, {1, 368, 1, 1});
      }
    }
    for (int qp = 0; qp < QpCount("i_decoder_q_scale"); ++qp) {
      if (qp != kBaseQp) {
        (void)Cached(&i_decoder_q_steps_, "i_decoder_q_scale", qp, {1, 368, 1, 1});
      }
    }
    for (int qp = 0; qp < QpCount("p_feature_q_scale"); ++qp) {
      (void)Cached(&p_feature_q_steps_, "p_feature_q_scale", qp, {1, 256, 1, 1});
    }
    for (int qp = 0; qp < QpCount("p_encoder_q_scale"); ++qp) {
      (void)Cached(&p_encoder_q_steps_, "p_encoder_q_scale", qp, {1, 256, 1, 1});
    }
    for (int qp = 0; qp < QpCount("p_decoder_q_scale"); ++qp) {
      (void)Cached(&p_decoder_q_steps_, "p_decoder_q_scale", qp, {1, 256, 1, 1});
    }
    for (int qp = 0; qp < QpCount("p_reconstruction_q_scale"); ++qp) {
      (void)Cached(&p_reconstruction_q_steps_, "p_reconstruction_q_scale", qp, {1, 320, 1, 1});
    }
  }

  void UploadStaticInputs(void* stream = nullptr, mlvc::Profiler* profiler = nullptr) {
    UploadEntry(&i_encoder_q_step_, stream, profiler);
    UploadEntry(&i_decoder_q_step_, stream, profiler);
    for (auto& [_, entry] : i_encoder_q_steps_) {
      UploadEntry(&entry, stream, profiler);
    }
    for (auto& [_, entry] : i_decoder_q_steps_) {
      UploadEntry(&entry, stream, profiler);
    }
    for (auto& [_, entry] : p_feature_q_steps_) {
      UploadEntry(&entry, stream, profiler);
    }
    for (auto& [_, entry] : p_encoder_q_steps_) {
      UploadEntry(&entry, stream, profiler);
    }
    for (auto& [_, entry] : p_decoder_q_steps_) {
      UploadEntry(&entry, stream, profiler);
    }
    for (auto& [_, entry] : p_reconstruction_q_steps_) {
      UploadEntry(&entry, stream, profiler);
    }
  }

 private:
  struct Entry {
    TensorData tensor;
    std::optional<mlvc::TensorHandle> handle;
  };

  Entry& Cached(std::map<int, Entry>* cache, const std::string& name, int qp,
                const std::vector<int64_t>& shape) {
    auto it = cache->find(qp);
    if (it == cache->end()) {
      it = cache->emplace(qp, Entry{MakeQScaleTensor(sidecar_, name, qp, shape), std::nullopt})
               .first;
    }
    return it->second;
  }

  static const mlvc::TensorHandle* HandleOrNull(const Entry& entry) {
    return entry.handle.has_value() && entry.handle->acl_valid() ? &*entry.handle : nullptr;
  }

  static void UploadEntry(Entry* entry, void* stream, mlvc::Profiler*) {
    mlvc::Check(entry != nullptr, "qscale cache entry is required");
    if (!entry->handle.has_value()) {
      entry->handle.emplace(entry->tensor.shape, entry->tensor.dtype);
      entry->handle->AttachCpuBuffer(entry->tensor.bytes.data(), entry->tensor.bytes.size(), true);
    }
    (void)entry->handle->EnsureAcl(stream);
  }

  int QpCount(const std::string& name) const {
    const mlvc::SidecarArray& array = sidecar_.Get(name);
    mlvc::Check(!array.shape.empty(), "q scale table has empty shape: " + name);
    return static_cast<int>(array.shape[0]);
  }

  const mlvc::RuntimeSidecar& sidecar_;
  Entry i_encoder_q_step_;
  Entry i_decoder_q_step_;
  std::map<int, Entry> i_encoder_q_steps_;
  std::map<int, Entry> i_decoder_q_steps_;
  std::map<int, Entry> p_feature_q_steps_;
  std::map<int, Entry> p_encoder_q_steps_;
  std::map<int, Entry> p_decoder_q_steps_;
  std::map<int, Entry> p_reconstruction_q_steps_;
};

}  // namespace mlvc::codec

#endif  // MLVC_CODEC_DETAIL_QSCALE_CACHE_H_
