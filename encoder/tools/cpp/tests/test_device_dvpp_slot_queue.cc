#include <mlvc/core/status.h>
#include <mlvc/io/udp_frame_transport.h>

#include <chrono>
#include <future>
#include <iostream>
#include <string>

int main() {
  try {
    using Queue = mlvc::io::detail::DeviceDvppSlotQueue;
    using Pending = mlvc::io::detail::DeviceDvppPendingSlot;

    Queue capacity_queue(1);
    const std::size_t first = capacity_queue.Acquire();
    std::future<std::size_t> blocked = std::async(std::launch::async, [&] {
      return capacity_queue.Acquire();
    });
    mlvc::Check(blocked.wait_for(std::chrono::milliseconds(20)) ==
                    std::future_status::timeout,
                "slot acquire must block while capacity is exhausted");
    capacity_queue.Release(first);
    const std::size_t reused = blocked.get();
    mlvc::Check(reused == first, "released slot was not reused");
    capacity_queue.Release(reused);

    Queue fifo_queue(2);
    const std::size_t slot0 = fifo_queue.Acquire();
    const std::size_t slot1 = fifo_queue.Acquire();
    fifo_queue.Enqueue(slot0, 17);
    fifo_queue.Enqueue(slot1, 18);
    Pending pending;
    mlvc::Check(fifo_queue.Pop(&pending) && pending.slot == slot0 &&
                    pending.frame_index == 17,
                "slot queue did not preserve FIFO order for the first frame");
    fifo_queue.Release(pending.slot);
    mlvc::Check(fifo_queue.Pop(&pending) && pending.slot == slot1 &&
                    pending.frame_index == 18,
                "slot queue did not preserve FIFO order for the second frame");
    fifo_queue.Release(pending.slot);
    fifo_queue.Shutdown();
    mlvc::Check(!fifo_queue.Pop(&pending),
                "shutdown queue must finish after pending slots are drained");

    Queue drain_queue(1);
    const std::size_t drain_slot = drain_queue.Acquire();
    drain_queue.Enqueue(drain_slot, 42);
    drain_queue.Shutdown();
    mlvc::Check(drain_queue.Pop(&pending) && pending.frame_index == 42,
                "shutdown must not discard an already queued slot");
    drain_queue.Release(pending.slot);
    mlvc::Check(!drain_queue.Pop(&pending),
                "drained shutdown queue must report completion");

    Queue failed_queue(1);
    const std::size_t held = failed_queue.Acquire();
    std::future<std::string> failed_waiter = std::async(std::launch::async, [&] {
      try {
        (void)failed_queue.Acquire();
      } catch (const std::exception& error) {
        return std::string(error.what());
      }
      return std::string();
    });
    mlvc::Check(failed_waiter.wait_for(std::chrono::milliseconds(20)) ==
                    std::future_status::timeout,
                "failure waiter must start in a blocked state");
    failed_queue.Fail(std::make_exception_ptr(mlvc::Error("expected slot failure")));
    mlvc::Check(failed_waiter.get() == "expected slot failure",
                "slot failure did not wake and rethrow to a blocked producer");
    (void)held;

    std::cout << "device DVPP slot queue test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_device_dvpp_slot_queue failed: " << error.what() << "\n";
    return 1;
  }
}
