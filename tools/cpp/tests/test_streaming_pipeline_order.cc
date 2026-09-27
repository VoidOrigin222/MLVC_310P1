#include <mlvc/core/status.h>
#include <mlvc/framework/streaming_pipeline.h>

#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

namespace {
struct Item : mlvc::DataObject {
  explicit Item(int value) : value(value) {}
  int value;
};
}  // namespace

int main() {
  try {
    mlvc::StreamingPipeline pipeline(4, 8);
    pipeline.SetProcessor([](const std::shared_ptr<mlvc::DataObject>& input) {
      const auto item = std::dynamic_pointer_cast<Item>(input);
      mlvc::Check(item != nullptr, "unexpected pipeline item type");
      std::this_thread::sleep_for(std::chrono::milliseconds(20 - item->value));
      return std::static_pointer_cast<mlvc::DataObject>(std::make_shared<Item>(item->value));
    });
    pipeline.Start();
    for (int i = 0; i < 8; ++i) {
      mlvc::Check(pipeline.AddInput(std::make_shared<Item>(i)), "failed to add pipeline item");
    }
    pipeline.CloseInput();
    std::vector<int> values;
    std::shared_ptr<mlvc::DataObject> output;
    while (pipeline.GetOutput(&output)) {
      const auto item = output->As<Item>();
      mlvc::Check(item != nullptr, "pipeline returned an unexpected output type");
      values.push_back(item->value);
    }
    pipeline.Stop();
    mlvc::Check(values.size() == 8, "pipeline output count mismatch");
    for (int i = 0; i < 8; ++i) {
      mlvc::Check(values[static_cast<std::size_t>(i)] == i,
                  "multi-worker pipeline output order mismatch at " + std::to_string(i) + ": got " +
                      std::to_string(values[static_cast<std::size_t>(i)]));
    }

    std::promise<void> first_started;
    std::promise<void> later_results_ready;
    std::promise<void> release_first;
    std::shared_future<void> release_first_future = release_first.get_future().share();
    mlvc::StreamingPipeline bounded_pipeline(2, 4);
    bounded_pipeline.SetProcessor(
        [&](const std::shared_ptr<mlvc::DataObject>& input) {
          const auto item = std::dynamic_pointer_cast<Item>(input);
          mlvc::Check(item != nullptr, "unexpected bounded pipeline item type");
          if (item->value == 0) {
            first_started.set_value();
            release_first_future.wait();
          } else if (item->value == 3) {
            later_results_ready.set_value();
          }
          return std::static_pointer_cast<mlvc::DataObject>(
              std::make_shared<Item>(item->value));
        });
    bounded_pipeline.Start();
    mlvc::Check(bounded_pipeline.AddInput(std::make_shared<Item>(0)),
                "failed to add first bounded pipeline item");
    first_started.get_future().wait();
    for (int value = 1; value < 4; ++value) {
      mlvc::Check(bounded_pipeline.AddInput(std::make_shared<Item>(value)),
                  "pipeline rejected input within total in-flight capacity");
    }
    later_results_ready.get_future().wait();
    auto fifth_add = std::async(std::launch::async, [&] {
      return bounded_pipeline.AddInput(std::make_shared<Item>(4));
    });
    const bool fifth_was_blocked =
        fifth_add.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout;
    release_first.set_value();
    mlvc::Check(bounded_pipeline.GetOutput(&output),
                "bounded pipeline did not publish its first output");
    const auto first_value = output->As<Item>()->value;
    const bool fifth_was_accepted = fifth_add.get();
    bounded_pipeline.CloseInput();
    std::vector<int> bounded_values{first_value};
    for (int index = 1; index < 5; ++index) {
      mlvc::Check(bounded_pipeline.GetOutput(&output),
                  "bounded pipeline did not publish every accepted output");
      bounded_values.push_back(output->As<Item>()->value);
    }
    bounded_pipeline.Stop();
    mlvc::Check(fifth_was_blocked,
                "reorder backlog bypassed total in-flight pipeline capacity");
    mlvc::Check(fifth_was_accepted, "pipeline rejected input after output capacity was released");
    mlvc::Check(bounded_values == std::vector<int>({0, 1, 2, 3, 4}),
                "bounded pipeline output order mismatch");

    mlvc::StreamingPipeline failing_pipeline(3, 4);
    failing_pipeline.SetProcessor([](const std::shared_ptr<mlvc::DataObject>& input) {
      const auto item = std::dynamic_pointer_cast<Item>(input);
      mlvc::Check(item != nullptr, "unexpected failing pipeline item type");
      mlvc::Check(item->value != 2, "expected worker failure");
      return std::static_pointer_cast<mlvc::DataObject>(std::make_shared<Item>(item->value));
    });
    failing_pipeline.Start();
    for (int i = 0; i < 4; ++i) {
      mlvc::Check(failing_pipeline.AddInput(std::make_shared<Item>(i)),
                  "failed to add failing pipeline item");
    }
    failing_pipeline.CloseInput();
    while (failing_pipeline.GetOutput(&output)) {
    }
    failing_pipeline.Stop();
    bool propagated = false;
    try {
      failing_pipeline.RethrowIfFailed();
    } catch (const std::exception&) {
      propagated = true;
    }
    mlvc::Check(propagated, "worker exception was not propagated");
    std::cout << "streaming pipeline order test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_streaming_pipeline_order failed: " << error.what() << "\n";
    return 1;
  }
}
