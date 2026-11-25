#ifndef ULBVC_DETECTION_YOLO_DETECTOR_H_
#define ULBVC_DETECTION_YOLO_DETECTOR_H_

#include <acl/acl.h>

#include <filesystem>
#include <future>
#include <memory>
#include <vector>

#include "ulbvc/framework/profiler.h"
#include "ulbvc/transport/detection_metadata.h"

namespace ulbvc::detection {

struct Options {
  bool enabled = false;
  std::filesystem::path model;
  std::filesystem::path labels;
  std::filesystem::path report;
  float confidence = 0.25f;
  float nms_iou = 0.45f;
  int queue_capacity = 2;
};

struct Stats {
  uint64_t frames = 0, boxes = 0;
  double preprocess_ms = 0, inference_ms = 0, postprocess_ms = 0;
  double color_convert_ms = 0, resize_ms = 0, pack_ms = 0;
  double submit_wait_ms = 0, submit_copy_ms = 0;
};

class YoloWorker {
 public:
  YoloWorker(aclrtContext context, Options options, int source_width, int source_height,
             Profiler* profiler = nullptr);
  ~YoloWorker();
  YoloWorker(const YoloWorker&) = delete;
  YoloWorker& operator=(const YoloWorker&) = delete;

  std::shared_future<transport::DetectionFrame> Submit(int frame_id,
                                                       const std::vector<uint8_t>& nv12);
  Stats stats() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ulbvc::detection
#endif  // ULBVC_DETECTION_YOLO_DETECTOR_H_
