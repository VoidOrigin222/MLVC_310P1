#include <mlvc/application/stream/encode/encode_schedule.h>
#include <mlvc/core/status.h>

#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>

int main() {
  try {
    using mlvc::codec::ShouldRetirePendingEntropy;
    using mlvc::codec::ShouldRetireReadyEntropy;
    std::promise<int> promise;
    auto front = promise.get_future();
    auto status = front.wait_for(std::chrono::milliseconds(0));
    mlvc::Check(status == std::future_status::timeout &&
                    !ShouldRetireReadyEntropy(2, false, status),
                "unfinished front must remain queued without a blocking get");
    // A later completed frame must never be used to bypass the unfinished front.
    std::promise<int> later_promise;
    auto later = later_promise.get_future();
    later_promise.set_value(2);
    mlvc::Check(later.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready &&
                    !ShouldRetireReadyEntropy(2, false, status),
                "later readiness must not bypass the queue front");
    promise.set_value(1);
    status = front.wait_for(std::chrono::milliseconds(0));
    mlvc::Check(ShouldRetireReadyEntropy(1, false, status) &&
                    ShouldRetireReadyEntropy(2, false, status),
                "fixed-QP prefetch must retire completed fronts");
    mlvc::Check(!ShouldRetireReadyEntropy(0, false, status) &&
                    !ShouldRetireReadyEntropy(2, true, status),
                "serial/adaptive-QP retirement timing must remain unchanged");
    mlvc::Check(front.valid() && front.get() == 1 && later.get() == 2,
                "readiness checks must not consume the future payload");
    auto deferred = std::async(std::launch::deferred, [] { return 3; });
    mlvc::Check(!ShouldRetireReadyEntropy(2, false,
                    deferred.wait_for(std::chrono::milliseconds(0))),
                "deferred work is not a completed payload");
    std::promise<int> failed_promise;
    auto failed = failed_promise.get_future();
    failed_promise.set_exception(std::make_exception_ptr(std::runtime_error("entropy failed")));
    mlvc::Check(ShouldRetireReadyEntropy(2, false,
                    failed.wait_for(std::chrono::milliseconds(0))),
                "completed entropy errors must reach the output get path");
    bool error_preserved = false;
    try { (void)failed.get(); }
    catch (const std::runtime_error& e) { error_preserved = std::string(e.what()) == "entropy failed"; }
    mlvc::Check(error_preserved, "entropy error was swallowed by readiness polling");
    mlvc::Check(!ShouldRetirePendingEntropy(2, 2) && ShouldRetirePendingEntropy(3, 2),
                "bounded blocking fallback must retain the original capacity");
    std::cout << "encode entropy retirement test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_encode_entropy_retirement failed: " << e.what() << "\n";
    return 1;
  }
}
