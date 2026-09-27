#include "mlvc/codec/mlvc_rate_control.h"

#include <algorithm>
#include <cmath>
#include <limits>

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

int ToIntegerBudget(long double value, const char* label) {
  if (!std::isfinite(value) || value < 0.0L ||
      value > static_cast<long double>(std::numeric_limits<int>::max())) {
    throw Error(std::string("rate-control ") + label + " exceeds the integer budget");
  }
  return static_cast<int>(value);
}

int ToSignedIntegerBudget(long double value, const char* label) {
  if (!std::isfinite(value) ||
      value < static_cast<long double>(std::numeric_limits<int>::min()) ||
      value > static_cast<long double>(std::numeric_limits<int>::max())) {
    throw Error(std::string("rate-control ") + label + " exceeds the signed integer budget");
  }
  return static_cast<int>(value);
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
  Check(std::isfinite(bitrate_in) && bitrate_in >= 0.0,
        "rate-control bitrate must be finite and non-negative");
  Check(std::isfinite(fps_in) && fps_in > 0.0,
        "rate-control fps must be finite and positive");
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
  Check(std::isfinite(presentation_time) && presentation_time >= 0.0,
        "rate-control presentation time must be finite and non-negative");
  if (!last_drain_timestamp.has_value()) {
    return 1.0 / fps;
  }
  Check(presentation_time >= *last_drain_timestamp,
        "rate-control presentation time must not move backwards");
  return presentation_time - *last_drain_timestamp;
}

double MlvcRateController::LeakyBucket::CalcDrainBits(double presentation_time) const {
  return std::min(fill_bits, CalcDrainSecs(presentation_time) * bitrate);
}

double MlvcRateController::LeakyBucket::CalcFillBits(double presentation_time) const {
  return fill_bits - CalcDrainBits(presentation_time);
}

void MlvcRateController::LeakyBucket::Update(double presentation_time, double frame_bits) {
  Check(std::isfinite(frame_bits) && frame_bits >= 0.0,
        "rate-control frame bits must be finite and non-negative");
  const double next_fill = CalcFillBits(presentation_time) + frame_bits;
  Check(std::isfinite(next_fill) && next_fill >= 0.0,
        "rate-control bucket state became invalid");
  // A leaky bucket cannot retain more than its configured capacity.  Clamping
  // here also prevents a sequence of oversized but representable frames from
  // growing the floating-point state until a later int-budget conversion
  // fails or overflows.
  fill_bits = std::min(next_fill, capacity_bits);
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
  const int decay_bits = ToIntegerBudget(
      std::max(0.0, (drain_secs / planned_excess_tau) * accumulated_excess_bits),
      "decay budget");
  const int excess_bits = ToIntegerBudget(
      std::max(0.0, (frame_weight - 1.0) *
                         (bucket.bitrate / std::max(bucket.fps, 1.0))),
      "excess budget");
  const int max_excess_bits = ToIntegerBudget(0.5L * bucket.capacity_bits, "maximum excess budget");
  return std::clamp(accumulated_excess_bits - decay_bits + excess_bits, 0, max_excess_bits);
}

MlvcRateController::RateAllocatorResult MlvcRateController::RateAllocator::Allocate(
    double presentation_time, double frame_weight, std::optional<double> undershoot_tau_override,
    std::optional<double> overshoot_tau_override) {
  Check(std::isfinite(frame_weight) && frame_weight > 0.0,
        "rate-control frame weight must be finite and positive");
  const int nominal_bits = ToIntegerBudget(
      frame_weight * (bucket.bitrate / std::max(bucket.fps, 1.0)), "nominal frame budget");
  const int64_t planned_excess_bits = CalcPlannedExcessBits(presentation_time, frame_weight);
  const int64_t target_fill_bits = static_cast<int64_t>(ToIntegerBudget(
      target_level * bucket.capacity_bits, "target fill budget")) + planned_excess_bits;
  const int64_t current_fill_bits = ToIntegerBudget(
      std::max(0.0, bucket.CalcFillBits(presentation_time)), "current fill budget");
  const int64_t error_bits = target_fill_bits - (current_fill_bits + nominal_bits);
  const double correction_tau =
      error_bits >= 0
          ? (undershoot_tau_override.has_value() ? *undershoot_tau_override : undershoot_tau)
          : (overshoot_tau_override.has_value() ? *overshoot_tau_override : overshoot_tau);
  Check(std::isfinite(correction_tau) && correction_tau > 0.0,
        "rate-control correction time constant must be finite and positive");
  const int correction_bits = ToSignedIntegerBudget(
      (1.0 / (correction_tau * std::max(bucket.fps, 1.0))) *
          static_cast<double>(error_bits),
      "correction budget");
  const int64_t bucket_max_fill_bits = ToIntegerBudget(0.9L * bucket.capacity_bits,
                                                      "bucket fill budget");
  const int64_t bucket_headroom_bits = bucket_max_fill_bits - current_fill_bits;
  int64_t allocated_bits = std::min<int64_t>(static_cast<int64_t>(nominal_bits) + correction_bits,
                                             bucket_headroom_bits);
  allocated_bits = std::clamp<int64_t>(
      allocated_bits, static_cast<int64_t>(0.33 * nominal_bits),
      static_cast<int64_t>(2.0 * nominal_bits));
  if (current_fill_bits + allocated_bits > bucket_max_fill_bits) {
    allocated_bits = 1;
  }
  return {nominal_bits, static_cast<int>(std::max<int64_t>(1, allocated_bits)),
          static_cast<double>(target_fill_bits) / std::max(bucket.capacity_bits, 1.0),
          static_cast<double>(current_fill_bits + std::max<int64_t>(1, allocated_bits)) /
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
  Check(std::isfinite(bpp) && bpp >= 0.0,
        "rate-control bits per pixel must be finite and non-negative");
  const double q = -std::log(std::max(kMinBpp, bpp) / std::max(alpha_, kMinBpp)) / beta_;
  return ClampQIndex(static_cast<int>(std::round(q)), params_.min_q_index, params_.max_q_index);
}

double MlvcRateController::RateModel::PredictBpp(int q_index) const {
  return alpha_ * std::exp(-beta_ * static_cast<double>(q_index));
}

void MlvcRateController::RateModel::Update(int q_index, double bpp) {
  Check(q_index >= params_.min_q_index && q_index <= params_.max_q_index,
        "rate-control Q index is outside the model range");
  Check(std::isfinite(bpp) && bpp >= 0.0,
        "rate-control bits per pixel must be finite and non-negative");
  const double last_observed_alpha = bpp * std::exp(beta_ * static_cast<double>(q_index));
  Check(std::isfinite(last_observed_alpha), "rate-control model update is not finite");
  alpha_ += (1.0 / params_.alpha_tau) * (last_observed_alpha - alpha_);
  if (num_updates_ == 0) {
    seed_alpha_ += (1.0 / params_.seed_alpha_tau) * (last_observed_alpha - seed_alpha_);
  }
  ++num_updates_;

  if (params_.beta_ramp_target.has_value() && params_.beta_ramp_duration.has_value()) {
    Check(std::isfinite(*params_.beta_ramp_target) && *params_.beta_ramp_target != 0.0 &&
              std::isfinite(*params_.beta_ramp_duration) && *params_.beta_ramp_duration > 0.0,
          "rate-control beta ramp parameters must be finite and valid");
    const double ramp_progress = std::min(
        1.0, static_cast<double>(num_updates_) / std::max(1.0, *params_.beta_ramp_duration));
    const double new_beta =
        params_.beta + ramp_progress * (*params_.beta_ramp_target - params_.beta);
    alpha_ *= std::exp((new_beta - beta_) * static_cast<double>(q_index));
    beta_ = new_beta;
  }
  Check(std::isfinite(alpha_) && alpha_ > 0.0 && std::isfinite(beta_) && beta_ != 0.0,
        "rate-control model state became invalid");
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
  Check(width_ <= 16384 && height_ <= 16384,
        "rate control frame geometry exceeds the supported limit");
  Check(static_cast<int64_t>(width_) * static_cast<int64_t>(height_) <=
            std::numeric_limits<int>::max(),
        "rate control frame geometry overflows pixel count");
  Check(std::isfinite(options.fps) && options.fps >= 0.1 && options.fps <= 1000.0,
        "rate control requires finite fps in [0.1, 1000]");
  Check(std::isfinite(options.target_bitrate_bps) && options.target_bitrate_bps >= 0.0,
        "rate control requires finite non-negative target bitrate");
  // The allocator intentionally uses integer bit budgets.  Bound the input
  // before any floating-point-to-int conversion so a pathological config
  // cannot invoke undefined behaviour or wrap the leaky bucket state.
  // Every allocator intermediate is deliberately kept below INT_MAX before
  // conversion: the bucket capacity, the weighted nominal frame budget and
  // the correction path all use integer budgets for deterministic behavior.
  const double max_target_bitrate = std::min(
      static_cast<double>(std::numeric_limits<int>::max()) * 0.25,
      static_cast<double>(std::numeric_limits<int>::max()) * options.fps / 16.0);
  Check(options.target_bitrate_bps <= max_target_bitrate,
        "target bitrate is too large for the rate-control integer budget");
  Check(options.default_q_index >= 0 && options.default_q_index <= 63,
        "default q-index must be in [0, 63]");
  Check(options.min_q_index >= 0 && options.min_q_index <= 63 &&
            options.max_q_index >= 0 && options.max_q_index <= 63,
        "q-index range must be within [0, 63]");
  Check(min_q_index_ <= max_q_index_, "invalid q-index range");
  default_q_index_ = ClampQIndex(default_q_index_, min_q_index_, max_q_index_);
}

int MlvcRateController::SolveQIndex(double presentation_time, MlvcFrameType frame_type) {
  Check(std::isfinite(presentation_time) && presentation_time >= 0.0,
        "rate-control presentation time must be finite and non-negative");
  Check(frame_type == MlvcFrameType::kIFrame || frame_type == MlvcFrameType::kPFrame ||
            frame_type == MlvcFrameType::kLtrRecovery,
        "rate-control frame type is invalid");
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

  const int64_t pixels = static_cast<int64_t>(width_) * static_cast<int64_t>(height_);
  const double target_bpp = std::max(
      0.0, static_cast<double>(alloc_result.allocated_bits -
                                static_cast<int>(kFrameOverheadBytes * 8)) /
                   static_cast<double>(std::max<int64_t>(pixels, 1)));
  int model_q_index = impl_.SolveQIndex(frame_type, target_bpp);
  model_q_index = ClampQIndex(model_q_index, min_q_index_, max_q_index_);

  prev_res_q_index_ = model_q_index;
  return model_q_index;
}

void MlvcRateController::Update(double presentation_time, MlvcFrameType frame_type, int q_index,
                                std::size_t payload_bytes, std::size_t overhead_bytes) {
  Check(std::isfinite(presentation_time) && presentation_time >= 0.0,
        "rate-control presentation time must be finite and non-negative");
  Check(frame_type == MlvcFrameType::kIFrame || frame_type == MlvcFrameType::kPFrame ||
            frame_type == MlvcFrameType::kLtrRecovery,
        "rate-control frame type is invalid");
  Check(q_index >= min_q_index_ && q_index <= max_q_index_,
        "rate-control Q index is outside the configured range");
  if (!enabled_) {
    return;
  }
  Check(payload_bytes <= std::numeric_limits<std::size_t>::max() - overhead_bytes,
        "rate-control frame byte count overflows");
  const std::size_t frame_bytes = payload_bytes + overhead_bytes;
  Check(frame_bytes <= static_cast<std::size_t>(std::numeric_limits<int>::max() / 8),
        "rate-control frame byte count exceeds integer budget");
  const int actual_frame_bits = static_cast<int>(frame_bytes * 8);
  alloc_.Update(presentation_time, frame_weight_, actual_frame_bits);
  const double actual_payload_bpp =
      static_cast<double>(payload_bytes) * 8.0 /
      static_cast<double>(std::max<int64_t>(static_cast<int64_t>(width_) * height_, 1));
  impl_.Update(frame_type, q_index, actual_payload_bpp);
  prev_frame_weight_ = frame_weight_;
  prev_res_q_index_ = q_index;
}

}  // namespace mlvc::codec
