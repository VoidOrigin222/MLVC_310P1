#ifndef MLVC_APPLICATION_PIPELINE_ORDERED_FUTURE_WINDOW_H_
#define MLVC_APPLICATION_PIPELINE_ORDERED_FUTURE_WINDOW_H_

#include <mlvc/core/status.h>

#include <algorithm>
#include <cstddef>
#include <deque>
#include <future>
#include <utility>
#include <vector>

namespace mlvc::app {

template <typename T>
class OrderedFutureWindow {
 public:
  explicit OrderedFutureWindow(std::size_t capacity) : slots_in_use_(capacity, false) {
    mlvc::Check(capacity > 0, "ordered future window capacity must be positive");
  }

  ~OrderedFutureWindow() {
    for (Pending& pending : pending_) {
      if (!pending.future.valid()) continue;
      try {
        pending.future.wait();
      } catch (...) {
      }
    }
  }

  OrderedFutureWindow(const OrderedFutureWindow&) = delete;
  OrderedFutureWindow& operator=(const OrderedFutureWindow&) = delete;

  template <typename Submit>
  void SubmitNext(Submit&& submit) {
    mlvc::Check(!full(), "ordered future window is full");
    const auto slot_it = std::find(slots_in_use_.begin(), slots_in_use_.end(), false);
    mlvc::Check(slot_it != slots_in_use_.end(), "ordered future window has no free slot");
    const std::size_t slot = static_cast<std::size_t>(slot_it - slots_in_use_.begin());
    slots_in_use_[slot] = true;

    Pending pending;
    pending.slot = slot;
    try {
      pending.future = std::forward<Submit>(submit)(slot);
      mlvc::Check(pending.future.valid(), "ordered future window received an invalid future");
      pending_.push_back(std::move(pending));
      max_observed_size_ = std::max(max_observed_size_, pending_.size());
    } catch (...) {
      if (pending.future.valid()) {
        pending.future.wait();
      }
      slots_in_use_[slot] = false;
      throw;
    }
  }

  T PopFront() {
    mlvc::Check(!pending_.empty(), "cannot pop an empty ordered future window");
    Pending pending = std::move(pending_.front());
    pending_.pop_front();
    try {
      T value = pending.future.get();
      slots_in_use_[pending.slot] = false;
      return value;
    } catch (...) {
      slots_in_use_[pending.slot] = false;
      throw;
    }
  }

  bool empty() const { return pending_.empty(); }
  bool full() const { return pending_.size() == slots_in_use_.size(); }
  std::size_t size() const { return pending_.size(); }
  std::size_t capacity() const { return slots_in_use_.size(); }
  std::size_t max_observed_size() const { return max_observed_size_; }

 private:
  struct Pending {
    std::size_t slot = 0;
    std::future<T> future;
  };

  std::vector<bool> slots_in_use_;
  std::deque<Pending> pending_;
  std::size_t max_observed_size_ = 0;
};

}  // namespace mlvc::app

#endif  // MLVC_APPLICATION_PIPELINE_ORDERED_FUTURE_WINDOW_H_
