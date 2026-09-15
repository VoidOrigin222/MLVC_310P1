#include <mlvc/core/status.h>
#include <mlvc/framework/streaming_pipeline.h>

#include <chrono>
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
    std::cout << "streaming pipeline order test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_streaming_pipeline_order failed: " << error.what() << "\n";
    return 1;
  }
}
