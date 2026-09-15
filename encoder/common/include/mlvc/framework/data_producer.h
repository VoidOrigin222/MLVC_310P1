#ifndef MLVC_FRAMEWORK_DATA_PRODUCER_H_
#define MLVC_FRAMEWORK_DATA_PRODUCER_H_

#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>

#include "mlvc/framework/data_object.h"

namespace mlvc {

class StreamingPipeline;

class DataProducer {
 public:
  DataProducer(StreamingPipeline& pipeline, std::atomic<bool>& running);
  virtual ~DataProducer();

  DataProducer(const DataProducer&) = delete;
  DataProducer& operator=(const DataProducer&) = delete;

  void Start();
  void Stop();
  void Join();
  void RethrowIfFailed() const;

 protected:
  virtual void Produce() = 0;
  bool AddData(std::shared_ptr<DataObject> data);
  bool ShouldContinue() const { return running_.load(); }

  StreamingPipeline& pipeline_;
  std::atomic<bool>& running_;

 private:
  void RunThread();
  std::thread thread_;
  mutable std::mutex error_mutex_;
  std::exception_ptr error_;
};

}  // namespace mlvc

#endif  // MLVC_FRAMEWORK_DATA_PRODUCER_H_
