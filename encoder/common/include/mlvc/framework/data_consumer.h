#ifndef MLVC_FRAMEWORK_DATA_CONSUMER_H_
#define MLVC_FRAMEWORK_DATA_CONSUMER_H_

#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>

#include "mlvc/framework/data_object.h"

namespace mlvc {

class StreamingPipeline;

class DataConsumer {
 public:
  DataConsumer(StreamingPipeline& pipeline, std::atomic<bool>& running);
  virtual ~DataConsumer();

  DataConsumer(const DataConsumer&) = delete;
  DataConsumer& operator=(const DataConsumer&) = delete;

  void Start();
  void Stop();
  void Join();
  void RethrowIfFailed() const;

 protected:
  virtual void Consume() = 0;
  bool GetData(std::shared_ptr<DataObject>* data);
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

#endif  // MLVC_FRAMEWORK_DATA_CONSUMER_H_
