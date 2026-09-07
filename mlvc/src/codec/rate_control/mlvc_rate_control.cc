#include "mlvc/codec/mlvc_rate_control.h"

#include <algorithm>
#include <cmath>

#include "mlvc/core/status.h"

namespace mlvc::codec {
namespace {

constexpr double kMinBpp = 1.0e-9;
constexpr double kIFrameWeight = 10.0;
constexpr double kLtrRecoveryWeight = 6.0;
constexpr double kPFrameWeight = 1.0;
constexpr double kFrameWeightTau = 2.0;

MlvcRateController::RateModelParams MakeIFrameParams() {
  return {0.04969, -0.03626, 2.0, 2.0, std::nullopt, std::nullopt, 0, 63};
}

MlvcRateController::RateModelParams MakeLtrRecoveryParams() {
  return {0.01047, -0.05306, 2.0, 2.0, std::nullopt, std::nullopt, 0, 63};
}

MlvcRateController::RateModelParams MakePAfterIdrParams() {
  return {0.02156, -0.03173, 2.0, 2.0, -0.07654, 9.0, 0, 63};
}

MlvcRateController::RateModelParams MakePAfterLtrParams() {
  return {0.00315, -0.05925, 2.0, 2.0, -0.08307, 9.0, 0, 63};
}

int ClampQIndex(int q_index, int min_q_index, int max_q_index) {
  return std::clamp(q_index, min_q_index, max_q_index);
}

}  // namespace

MlvcRateController::LeakyBucket::LeakyBucket(double bitrate_in, double fps_in,
                                             double bucket_size_in, double initial_level_in)
    : bitrate(bitrate_in),
      fps(fps_in),
      bucket_size(bucket_size_in),
      initial_level(initial_level_in) {
  capacity_bits = static_cast<double>(bitrate) * bucket_size;
  fill_bits = initial_level * capacity_bits;
}

void MlvcRateController::LeakyBucket::Configure(double bitrate_in, double fps_in) {
  if (bitrate_in > 0.0) {
    const double excess_bits = fill_bits - initial_level * capacity_bits;
    const double new_capacity_bits = bitrate_in * bucket_size;
    const double new_fill_bits =
        std::clamp(initial_level * new_capacity_bits + excess_bits, 0.0, new_capacity_bits);
    bitrate = bitrate_in;
    capacity_bits = new_capacity_bits;
    fill_bits = new_fill_bits;
  }
  if (fps_in > 0.0) {
    fps = fps_in;
  }
}

double MlvcRateController::LeakyBucket::CalcDrainSecs(double presentation_time) const {
  if (!last_drain_timestamp.has_value()) {
    return 1.0 / fps;
  }
  return presentation_time - *last_drain_timestamp;
}

double MlvcRateController::LeakyBucket::CalcDrainBits(double presentation_time) const {
  return std::min(fill_bits, CalcDrainSecs(presentation_time) * bitrate);
}

double MlvcRateController::LeakyBucket::CalcFillBits(double presentation_time) const {
  return fill_bits - CalcDrainBits(presentation_time);
}

void MlvcRateController::LeakyBucket::Update(double presentation_time, double frame_bits) {
  fill_bits = CalcFillBits(presentation_time) + frame_bits;
  last_drain_timestamp = presentation_time;
}

double MlvcRateController::LeakyBucket::Level() const {
  return capacity_bits > 0.0 ? fill_bits / capacity_bits : 0.0;
}

MlvcRateController::RateAllocator::RateAllocator(double bitrate, double fps, double target_level_in)
    : bucket(bitrate, fps, 1.0, target_level_in), target_level(target_level_in) {}

void MlvcRateController::RateAllocator::Configure(double bitrate, double fps) {
  bucket.Configure(bitrate, fps);
}

int MlvcRateController::RateAllocator::CalcPlannedExcessBits(double presentation_time,
                                                             double frame_weight) const {
  const double drain_secs = bucket.CalcDrainSecs(presentation_time);
  const int decay_bits =
      static_cast<int>((drain_secs / planned_excess_tau) * accumulated_excess_bits);
  const int excess_bits = std::max(
      0, static_cast<int>((frame_weight - 1.0) * (bucket.bitrate / std::max(bucket.fps, 1.0))));
  const int max_excess_bits = static_cast<int>(0.5 * bucket.capacity_bits);
  return std::clamp(accumulated_excess_bits - decay_bits + excess_bits, 0, max_excess_bits);
}

MlvcRateController::RateAllocatorResult MlvcRateController::RateAllocator::Allocate(
    double presentation_time, double frame_weight, std::optional<double> undershoot_tau_override,
    std::optional<double> overshoot_tau_override) {
  const int nominal_bits =
      static_cast<int>(frame_weight * (bucket.bitrate / std::max(bucket.fps, 1.0)));
  const int planned_excess_bits = CalcPlannedExcessBits(presentation_time, frame_weight);
  const int target_fill_bits =
      static_cast<int>(target_level * bucket.capacity_bits) + planned_excess_bits;
  const int current_fill_bits = static_cast<int>(bucket.CalcFillBits(presentation_time));
  const int error_bits = target_fill_bits - (current_fill_bits + nominal_bits);
  const double correction_tau =
      error_bits >= 0
          ? (undershoot_tau_override.has_value() ? *undershoot_tau_override : undershoot_tau)
          : (overshoot_tau_override.has_value() ? *overshoot_tau_override : overshoot_tau);
  const int correction_bits = static_cast<int>(
      (1.0 / (correction_tau * std::max(bucket.fps, 1.0))) * static_cast<double>(error_bits));
  const int bucket_max_fill_bits = static_cast<int>(0.9 * bucket.capacity_bits);
  const int bucket_headroom_bits = bucket_max_fill_bits - current_fill_bits;
  int allocated_bits = std::min(nominal_bits + correction_bits, bucket_headroom_bits);
  allocated_bits = std::clamp(allocated_bits, static_cast<int>(0.33 * nominal_bits),
                              static_cast<int>(2.0 * nominal_bits));
  if (current_fill_bits + allocated_bits > bucket_max_fill_bits) {
    allocated_bits = 1;
  }
  return {nominal_bits, std::max(1, allocated_bits),
          static_cast<double>(target_fill_bits) / std::max(bucket.capacity_bits, 1.0),
          static_cast<double>(current_fill_bits + std::max(1, allocated_bits)) /
              std::max(bucket.capacity_bits, 1.0)};
}

void MlvcRateController::RateAllocator::Update(double presentation_time, double frame_weight,
                                               int frame_bits) {
  accumulated_excess_bits = CalcPlannedExcessBits(presentation_time, frame_weight);
  bucket.Update(presentation_time, static_cast<double>(frame_bits));
}

MlvcRateController::RateModel::RateModel(const RateModelParams& params) : params_(params) {
  seed_alpha_ = params_.initial_alpha;
  Reset();
}

void MlvcRateController::RateModel::Reset() {
  alpha_ = seed_alpha_;
  beta_ = params_.beta;
  num_updates_ = 0;
}

int MlvcRateController::RateModel::SolveQIndex(double bpp) const {
  const double q = -std::log(std::max(kMinBpp, bpp) / std::max(alpha_, kMinBpp)) / beta_;
  return ClampQIndex(static_cast<int>(std::round(q)), params_.min_q_index, params_.max_q_index);
}

double MlvcRateController::RateModel::PredictBpp(int q_index) const {
  return alpha_ * std::exp(-beta_ * static_cast<double>(q_index));
}

void MlvcRateController::RateModel::Update(int q_index, double bpp) {
  const double last_observed_alpha = bpp * std::exp(beta_ * static_cast<double>(q_index));
  alpha_ += (1.0 / params_.alpha_tau) * (last_observed_alpha - alpha_);
  if (num_updates_ == 0) {
    seed_alpha_ += (1.0 / params_.seed_alpha_tau) * (last_observed_alpha - seed_alpha_);
  }
  ++num_updates_;

  if (params_.beta_ramp_target.has_value() && params_.beta_ramp_duration.has_value()) {
    const double ramp_progress = std::min(
        1.0, static_cast<double>(num_updates_) / std::max(1.0, *params_.beta_ramp_duration));
    const double new_beta =
        params_.beta + ramp_progress * (*params_.beta_ramp_target - params_.beta);
    alpha_ *= std::exp((new_beta - beta_) * static_cast<double>(q_index));
    beta_ = new_beta;
  }
}

MlvcRateController::RateImplementation::RateImplementation()
    : iframe_model_(MakeIFrameParams()),
      ltr_recovery_model_(MakeLtrRecoveryParams()),
      p_frame_after_idr_model_(MakePAfterIdrParams()),
      p_frame_after_ltr_model_(MakePAfterLtrParams()) {}

int MlvcRateController::RateImplementation::SolveQIndex(MlvcFrameType frame_type,
                                                        double bpp) const {
  return GetModel(frame_type).SolveQIndex(bpp);
}

void MlvcRateController::RateImplementation::Update(MlvcFrameType frame_type, int q_index,
                                                    double bpp) {
  GetModel(frame_type).Update(q_index, bpp);
  if (frame_type == MlvcFrameType::kIFrame || frame_type == MlvcFrameType::kLtrRecovery) {
    p_frame_after_idr_model_.Reset();
    p_frame_after_ltr_model_.Reset();
    last_recovery_type_ = frame_type;
  }
}

MlvcRateController::RateModel& MlvcRateController::RateImplementation::GetModel(
    MlvcFrameType frame_type) {
  return const_cast<RateModel&>(static_cast<const RateImplementation*>(this)->GetModel(frame_type));
}

const MlvcRateController::RateModel& MlvcRateController::RateImplementation::GetModel(
    MlvcFrameType frame_type) const {
  if (frame_type == MlvcFrameType::kIFrame) {
    return iframe_model_;
  }
  if (frame_type == MlvcFrameType::kLtrRecovery) {
    return ltr_recovery_model_;
  }
  if (last_recovery_type_ == MlvcFrameType::kIFrame) {
    return p_frame_after_idr_model_;
  }
  if (last_recovery_type_ == MlvcFrameType::kLtrRecovery) {
    return p_frame_after_ltr_model_;
  }
  throw Error("unsupported frame type in rate implementation");
}

MlvcRateController::MlvcRateController(const MlvcRateControlOptions& options)
    : enabled_(options.target_bitrate_bps > 0.0),
      width_(options.width),
      height_(options.height),
      fps_(options.fps > 0.0 ? options.fps : 30.0),
      target_bitrate_bps_(options.target_bitrate_bps),
      default_q_index_(options.default_q_index),
      min_q_index_(options.min_q_index),
      max_q_index_(options.max_q_index),
      alloc_(options.target_bitrate_bps, fps_),
      impl_() {
  Check(width_ > 0 && height_ > 0, "rate control requires valid frame geometry");
  Check(min_q_index_ <= max_q_index_, "invalid q-index range");
  default_q_index_ = ClampQIndex(default_q_index_, min_q_index_, max_q_index_);
}

int MlvcRateController::SolveQIndex(double presentation_time, MlvcFrameType frame_type) {
  presentation_time_ = presentation_time;
  frame_type_ = frame_type;
  if (!enabled_) {
    return default_q_index_;
  }

  frame_weight_ = prev_frame_weight_ + (1.0 / kFrameWeightTau) * (1.0 - prev_frame_weight_);
  const double desired_weight =
      frame_type == MlvcFrameType::kIFrame
          ? kIFrameWeight
          : (frame_type == MlvcFrameType::kLtrRecovery ? kLtrRecoveryWeight : kPFrameWeight);
  frame_weight_ = std::max(frame_weight_, desired_weight);

  const std::optional<double> undershoot_tau =
      frame_type == MlvcFrameType::kIFrame || frame_type == MlvcFrameType::kLtrRecovery
          ? std::optional<double>(100.0)
          : std::nullopt;
  const RateAllocatorResult alloc_result =
      alloc_.Allocate(presentation_time_, frame_weight_, undershoot_tau, std::nullopt);

  const double target_bpp = std::max(
      0.0,
      static_cast<double>(alloc_result.allocated_bits - static_cast<int>(kFrameOverheadBytes * 8)) /
          static_cast<double>(std::max(width_ * height_, 1)));
  int model_q_index = impl_.SolveQIndex(frame_type, target_bpp);
  model_q_index = ClampQIndex(model_q_index, min_q_index_, max_q_index_);

  prev_res_q_index_ = model_q_index;
  return model_q_index;
}

void MlvcRateController::Update(double presentation_time, MlvcFrameType frame_type, int q_index,
                                std::size_t payload_bytes, std::size_t overhead_bytes) {
  if (!enabled_) {
    return;
  }
  const int actual_frame_bits = static_cast<int>((payload_bytes + overhead_bytes) * 8);
  alloc_.Update(presentation_time, frame_weight_, actual_frame_bits);
  const double actual_payload_bpp =
      static_cast<double>(payload_bytes * 8) / static_cast<double>(std::max(width_ * height_, 1));
  impl_.Update(frame_type, q_index, actual_payload_bpp);
  prev_frame_weight_ = frame_weight_;
  prev_res_q_index_ = q_index;
}

}  // namespace mlvc::codec
