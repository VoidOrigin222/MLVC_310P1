#include <ulbvc/application/pipeline/ordered_future_window.h>
#include <ulbvc/application/stream/encode/encode_schedule.h>
#include <ulbvc/core/status.h>

#include <exception>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void TestRejectsZeroCapacity() {
  bool rejected = false;
  try {
    ulbvc::app::OrderedFutureWindow<int> window(0);
  } catch (const std::exception&) {
    rejected = true;
  }
  ulbvc::Check(rejected, "ordered future window must reject zero capacity");
}

void TestTwoSlotsRetireInOrderAndReuseReleasedSlot() {
  ulbvc::app::OrderedFutureWindow<int> window(2);
  std::promise<int> first;
  std::promise<int> second;
  std::vector<std::size_t> slots;

  window.SubmitNext([&](std::size_t slot) {
    slots.push_back(slot);
    return first.get_future();
  });
  window.SubmitNext([&](std::size_t slot) {
    slots.push_back(slot);
    return second.get_future();
  });

  ulbvc::Check(window.full() && window.size() == 2,
              "two-slot window must accept two in-flight tasks");
  ulbvc::Check(slots.size() == 2 && slots[0] != slots[1], "in-flight tasks must use distinct slots");
  ulbvc::Check(window.max_observed_size() == 2, "window must record maximum in-flight count");

  second.set_value(22);
  first.set_value(11);
  ulbvc::Check(window.PopFront() == 11,
              "results must retire in submission order, not completion order");

  std::promise<int> third;
  std::size_t third_slot = window.capacity();
  window.SubmitNext([&](std::size_t slot) {
    third_slot = slot;
    return third.get_future();
  });
  ulbvc::Check(third_slot == slots[0], "only the retired task slot may be reused");
  third.set_value(33);
  ulbvc::Check(window.PopFront() == 22, "second submitted result must retire next");
  ulbvc::Check(window.PopFront() == 33, "reused-slot result must retire last");
  ulbvc::Check(window.empty(), "window must be empty after all results retire");
}

void TestOneSlotBehavior() {
  ulbvc::app::OrderedFutureWindow<int> window(1);
  std::promise<int> first;
  window.SubmitNext([&](std::size_t slot) {
    ulbvc::Check(slot == 0, "single-slot window must use slot zero");
    return first.get_future();
  });
  ulbvc::Check(window.full(), "single-slot window must be full after one submit");
  first.set_value(7);
  ulbvc::Check(window.PopFront() == 7, "single-slot result mismatch");

  std::promise<int> second;
  window.SubmitNext([&](std::size_t slot) {
    ulbvc::Check(slot == 0, "single slot must be reusable after retirement");
    return second.get_future();
  });
  second.set_value(8);
  ulbvc::Check(window.PopFront() == 8, "reused single-slot result mismatch");
}

void TestExceptionPropagatesWithoutLosingLaterResult() {
  ulbvc::app::OrderedFutureWindow<int> window(2);
  std::promise<int> failed;
  std::promise<int> later;
  window.SubmitNext([&](std::size_t) { return failed.get_future(); });
  window.SubmitNext([&](std::size_t) { return later.get_future(); });
  failed.set_exception(std::make_exception_ptr(std::runtime_error("expected failure")));
  later.set_value(19);

  bool propagated = false;
  try {
    (void)window.PopFront();
  } catch (const std::runtime_error& error) {
    propagated = std::string(error.what()) == "expected failure";
  }
  ulbvc::Check(propagated, "oldest task exception must propagate");
  ulbvc::Check(window.PopFront() == 19, "later result must remain ordered after an exception");
}

void TestEncodePendingWindowRetiresOnlyWhenCapacityIsExceeded() {
  constexpr std::size_t capacity = 2;
  ulbvc::Check(!ulbvc::codec::ShouldRetirePendingEntropy(1, capacity),
              "one pending entropy task must remain in flight");
  ulbvc::Check(!ulbvc::codec::ShouldRetirePendingEntropy(2, capacity),
              "a full entropy window must remain in flight");
  ulbvc::Check(ulbvc::codec::ShouldRetirePendingEntropy(3, capacity),
              "the oldest entropy task must retire after capacity is exceeded");

  bool rejected = false;
  try {
    (void)ulbvc::codec::ShouldRetirePendingEntropy(1, 0);
  } catch (const std::exception&) {
    rejected = true;
  }
  ulbvc::Check(rejected, "encode pending entropy capacity must be positive");
}

}  // namespace

int main() {
  try {
    TestRejectsZeroCapacity();
    TestTwoSlotsRetireInOrderAndReuseReleasedSlot();
    TestOneSlotBehavior();
    TestExceptionPropagatesWithoutLosingLaterResult();
    TestEncodePendingWindowRetiresOnlyWhenCapacityIsExceeded();
    std::cout << "ordered future window test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_ordered_future_window failed: " << error.what() << "\n";
    return 1;
  }
}
