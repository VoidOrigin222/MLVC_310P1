#ifndef MLVC_CODEC_MLVC_RATE_CONTROL_H_
#define MLVC_CODEC_MLVC_RATE_CONTROL_H_

#include <cstddef>
#include <cstdint>
#include <optional>

#include "mlvc/codec/mlvc_entropy.h"

namespace mlvc::codec {

struct MlvcRateControlOptions {
  int width = 0;
  int height = 0;
  double fps = 30.0;
  double target_bitrate_bps = 0.0;
  int default_q_index = 21;
  int min_q_index = 0;
  int max_q_index = 63;
};

class MlvcRateController {
 public:
  explicit MlvcRateController(const MlvcRateControlOptions& options);

  bool enabled() const { return enabled_; }
  int SolveQIndex(double presentation_time, MlvcFrameType frame_type);
  void Update(double presentation_time, MlvcFrameType frame_type, int q_index,
              std::size_t payload_bytes, std::size_t overhead_bytes = 0);

  struct RateModelParams {
    double initial_alpha = 0.0;
    double beta = 0.0;
    double alpha_tau = 2.0;
    double seed_alpha_tau = 2.0;
    std::optional<double> beta_ramp_target;
    std::optional<double> beta_ramp_duration;
    int min_q_index = 0;
    int max_q_index = 63;
  };

 private:
  struct LeakyBucket {
    double bitrate = 0.0;
    double fps = 30.0;
    double bucket_size = 1.0;
    double initial_level = 0.1;
    double capacity_bits = 0.0;
    double fill_bits = 0.0;
    std::optional<double> last_drain_timestamp;

    LeakyBucket() = default;
    LeakyBucket(double bitrate_in, double fps_in, double bucket_size_in = 1.0,
                double initial_level_in = 0.1);

    void Configure(double bitrate_in, double fps_in);
    double CalcDrainSecs(double presentation_time) const;
    double CalcFillBits(double presentation_time) const;
    void Update(double presentation_time, double frame_bits);
    double Level() const;

   private:
    double CalcDrainBits(double presentation_time) const;
  };

  struct RateAllocatorResult {
    int nominal_bits = 0;
    int allocated_bits = 0;
    double effective_target_level = 0.0;
    double estimated_level = 0.0;
  };

  struct RateAllocator {
    LeakyBucket bucket;
    double target_level = 0.1;
    double overshoot_tau = 0.25;
    double undershoot_tau = 1.0;
    double planned_excess_tau = 0.5;
    int accumulated_excess_bits = 0;

    RateAllocator() = default;
    RateAllocator(double bitrate, double fps, double target_level_in = 0.1);

    void Configure(double bitrate, double fps);
    RateAllocatorResult Allocate(double presentation_time, double frame_weight,
                                 std::optional<double> undershoot_tau_override = std::nullopt,
                                 std::optional<double> overshoot_tau_override = std::nullopt);
    void Update(double presentation_time, double frame_weight, int frame_bits);

   private:
    int CalcPlannedExcessBits(double presentation_time, double frame_weight) const;
  };

  class RateModel {
   public:
    explicit RateModel(const RateModelParams& params);

    void Reset();
    int SolveQIndex(double bpp) const;
    double PredictBpp(int q_index) const;
    void Update(int q_index, double bpp);

   private:
    RateModelParams params_;
    double seed_alpha_ = 0.0;
    double alpha_ = 0.0;
    double beta_ = 0.0;
    int num_updates_ = 0;
  };

  class RateImplementation {
   public:
    RateImplementation();

    int SolveQIndex(MlvcFrameType frame_type, double bpp) const;
    void Update(MlvcFrameType frame_type, int q_index, double bpp);

   private:
    RateModel& GetModel(MlvcFrameType frame_type);
    const RateModel& GetModel(MlvcFrameType frame_type) const;

    RateModel iframe_model_;
    RateModel ltr_recovery_model_;
    RateModel p_frame_after_idr_model_;
    RateModel p_frame_after_ltr_model_;
    MlvcFrameType last_recovery_type_ = MlvcFrameType::kIFrame;
  };

  static constexpr std::size_t kFrameOverheadBytes = 17;

  bool enabled_ = false;
  int width_ = 0;
  int height_ = 0;
  double fps_ = 30.0;
  double target_bitrate_bps_ = 0.0;
  int default_q_index_ = 21;
  int min_q_index_ = 0;
  int max_q_index_ = 63;
  mutable double frame_weight_ = 1.0;
  mutable double prev_frame_weight_ = 1.0;
  mutable int prev_res_q_index_ = -1;
  mutable MlvcFrameType frame_type_ = MlvcFrameType::kIFrame;
  mutable double presentation_time_ = 0.0;
  mutable RateAllocator alloc_;
  mutable RateImplementation impl_;
};

}  // namespace mlvc::codec

#endif  // MLVC_CODEC_MLVC_RATE_CONTROL_H_
