#include <acl/ops/acl_dvpp.h>
#include <ulbvc/codec/tensor_data.h>
#include <ulbvc/core/status.h>
#include <ulbvc/detection/yolo_detector.h>
#include <ulbvc/io/dvpp_jpeg_decoder.h>
#include <ulbvc/runtime/acl_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point first, Clock::time_point last) {
  return std::chrono::duration<double, std::milli>(last - first).count();
}
struct Allocation {
  void* data = nullptr;
  explicit Allocation(std::size_t bytes) {
    ulbvc::CheckAcl(acldvppMalloc(&data, bytes), "allocate JPEGD benchmark output");
  }
  ~Allocation() {
    if (data) acldvppFree(data);
  }
};
struct Slot {
  std::unique_ptr<Allocation> memory;
  acldvppPicDesc* picture = nullptr;
  ~Slot() {
    if (picture) acldvppDestroyPicDesc(picture);
  }
};
struct Session {
  acldvppChannelDesc* channel = nullptr;
  aclrtStream stream = nullptr;
  bool created = false;
  std::vector<std::unique_ptr<Slot>> slots;
  ~Session() {
    // Outstanding work must finish before any descriptor or output buffer is released.
    if (stream) aclrtSynchronizeStream(stream);
    slots.clear();
    if (stream) aclrtDestroyStream(stream);
    if (created) acldvppDestroyChannel(channel);
    if (channel) acldvppDestroyChannelDesc(channel);
  }
};

// Isolated encoder-board load: same context, independent stream, at most 30 calls/s.
class BackgroundYolo {
 public:
  BackgroundYolo(aclrtContext context, const char* model)
      : runtime_(context), stage_(&runtime_, ulbvc::ModelRecord{}, model, true) {
    ulbvc::Check(stage_.record().inputs.size() == 1 && stage_.record().outputs.size() == 1,
                 "background model must have one input/output");
    input_ = ulbvc::codec::MakeTensorLike(stage_.record().inputs[0]);
    output_ = ulbvc::codec::MakeTensorLike(stage_.record().outputs[0]);
    ulbvc::Check(input_.dtype == ulbvc::DataType::kFloat16, "background input must be FP16");
    ulbvc::codec::FillFp16Tensor(114.0f / 255, &input_);
    started_ = Clock::now();
    worker_ = std::thread([this] {
      try {
        runtime_.MakeCurrent();
        auto next = Clock::now();
        while (!stop_.load()) {
          stage_.Run({{stage_.record().inputs[0].name.c_str(), input_.View()}},
                     {{stage_.record().outputs[0].name.c_str(), output_.View()}});
          ++calls_;
          next += std::chrono::microseconds(33333);
          std::this_thread::sleep_until(next);
        }
      } catch (...) {
        error_ = std::current_exception();
      }
    });
  }
  ~BackgroundYolo() { Join(); }
  void Finish() {
    Join();
    if (error_) std::rethrow_exception(error_);
    std::cout << "{\"background\":\"yolo_same_context\",\"calls\":" << calls_
              << ",\"rate_limit_fps\":30,\"elapsed_ms\":" << Ms(started_, Clock::now()) << "}\n";
  }

 private:
  void Join() {
    stop_ = true;
    if (worker_.joinable()) worker_.join();
  }
  ulbvc::AclRuntime runtime_;
  ulbvc::AclStage stage_;
  ulbvc::codec::TensorData input_, output_;
  std::atomic<bool> stop_{false};
  std::thread worker_;
  std::exception_ptr error_;
  uint64_t calls_ = 0;
  Clock::time_point started_;
};
bool SameVisible(const std::vector<uint8_t>& first, const std::vector<uint8_t>& second, int width,
                 int height, int width_stride, int height_stride) {
  for (int y = 0; y < height; ++y)
    if (std::memcmp(first.data() + y * width_stride, second.data() + y * width_stride, width))
      return false;
  const auto uv = static_cast<std::size_t>(width_stride) * height_stride;
  for (int y = 0; y < height / 2; ++y)
    if (std::memcmp(first.data() + uv + y * width_stride, second.data() + uv + y * width_stride,
                    width))
      return false;
  return true;
}

void Run(const std::vector<std::vector<uint8_t>>& images,
         const std::vector<std::vector<uint8_t>>& reference, int width, int height, uint32_t flags,
         int depth, int frames, bool device_input) {
  const int width_stride = (width + 15) / 16 * 16, height_stride = (height + 15) / 16 * 16;
  const std::size_t visible_storage =
      static_cast<std::size_t>(width_stride) * height_stride * 3 / 2;
  std::size_t allocation = visible_storage * 2;
  for (const auto& image : images) {
    uint32_t predicted = 0;
    ulbvc::CheckAcl(acldvppJpegPredictDecSize(image.data(), image.size(),
                                              PIXEL_FORMAT_YUV_SEMIPLANAR_420, &predicted),
                    "predict JPEGD benchmark output");
    allocation = std::max(allocation, static_cast<std::size_t>(predicted));
  }
  std::vector<std::unique_ptr<Allocation>> encoded;
  std::vector<const void*> input_pointers;
  for (const auto& image : images) {
    if (device_input) {
      auto buffer = std::make_unique<Allocation>(image.size());
      ulbvc::CheckAcl(aclrtMemcpy(buffer->data, image.size(), image.data(), image.size(),
                                  ACL_MEMCPY_HOST_TO_DEVICE),
                      "stage JPEG benchmark input");
      input_pointers.push_back(buffer->data);
      encoded.push_back(std::move(buffer));
    } else
      input_pointers.push_back(image.data());
  }
  Session session;
  session.channel = acldvppCreateChannelDesc();
  ulbvc::Check(session.channel, "create JPEGD benchmark channel descriptor");
  ulbvc::CheckAcl(acldvppCreateChannel(session.channel), "create JPEGD benchmark channel");
  session.created = true;
  if (flags)
    ulbvc::CheckAcl(aclrtCreateStreamWithConfig(&session.stream, 0, flags),
                    "create fast JPEGD stream");
  else
    ulbvc::CheckAcl(aclrtCreateStream(&session.stream), "create default JPEGD stream");
  for (int index = 0; index < depth; ++index) {
    auto slot = std::make_unique<Slot>();
    slot->memory = std::make_unique<Allocation>(allocation);
    slot->picture = acldvppCreatePicDesc();
    ulbvc::Check(slot->picture, "create JPEGD benchmark picture descriptor");
    ulbvc::CheckAcl(acldvppSetPicDescData(slot->picture, slot->memory->data), "set JPEGD data");
    ulbvc::CheckAcl(acldvppSetPicDescSize(slot->picture, allocation), "set JPEGD bytes");
    ulbvc::CheckAcl(acldvppSetPicDescFormat(slot->picture, PIXEL_FORMAT_YUV_SEMIPLANAR_420),
                    "set JPEGD format");
    ulbvc::CheckAcl(acldvppSetPicDescWidth(slot->picture, width), "set JPEGD width");
    ulbvc::CheckAcl(acldvppSetPicDescHeight(slot->picture, height), "set JPEGD height");
    ulbvc::CheckAcl(acldvppSetPicDescWidthStride(slot->picture, width_stride),
                    "set JPEGD width stride");
    ulbvc::CheckAcl(acldvppSetPicDescHeightStride(slot->picture, height_stride),
                    "set JPEGD height stride");
    session.slots.push_back(std::move(slot));
  }
  const auto batch = [&](int index, double* submit, double* wait, bool verify) {
    const auto begin = Clock::now();
    for (int slot = 0; slot < depth; ++slot) {
      const auto source = (index + slot) % images.size();
      const auto& image = images[source];
      ulbvc::CheckAcl(acldvppJpegDecodeAsync(session.channel, input_pointers[source], image.size(),
                                             session.slots[slot]->picture, session.stream),
                      "submit JPEGD benchmark");
    }
    const auto submitted = Clock::now();
    ulbvc::CheckAcl(aclrtSynchronizeStream(session.stream), "wait JPEGD benchmark");
    const auto done = Clock::now();
    *submit += Ms(begin, submitted);
    *wait += Ms(submitted, done);
    for (const auto& slot : session.slots)
      ulbvc::Check(acldvppGetPicDescRetCode(slot->picture) == 0, "JPEGD reported a decode error");
    if (verify) {
      for (int slot = 0; slot < depth; ++slot) {
        std::vector<uint8_t> result(visible_storage);
        ulbvc::CheckAcl(aclrtMemcpy(result.data(), result.size(), session.slots[slot]->memory->data,
                                    result.size(), ACL_MEMCPY_DEVICE_TO_HOST),
                        "copy JPEGD verification output");
        ulbvc::Check(SameVisible(result, reference[(index + slot) % images.size()], width, height,
                                 width_stride, height_stride),
                     "JPEGD output/frame order mismatch");
      }
    }
  };
  double ignored_submit = 0, ignored_wait = 0;
  for (int index = 0; index < 4; index += depth) batch(index, &ignored_submit, &ignored_wait, true);
  double submit = 0, wait = 0;
  const auto begin = Clock::now();
  for (int index = 0; index < frames; index += depth) batch(index, &submit, &wait, false);
  const double total = Ms(begin, Clock::now());
  // Verify recycled buffers after timing, including alternating source images.
  for (int index = 0; index < 8; index += depth) batch(index, &ignored_submit, &ignored_wait, true);
  std::cout << "{\"input_memory\":\"" << (device_input ? "dvpp" : "direct")
            << "\",\"stream_flags\":" << flags << ",\"depth\":" << depth << ",\"frames\":" << frames
            << ",\"submit_avg_ms\":" << submit / frames << ",\"wait_avg_ms\":" << wait / frames
            << ",\"frame_avg_ms\":" << total / frames << ",\"fps\":" << frames * 1000 / total
            << ",\"visible_output_verification\":\"passed\"}\n";
}
void RunWrapper(aclrtContext context, const std::vector<std::vector<uint8_t>>& images,
                const std::vector<std::vector<uint8_t>>& reference, int frames) {
  ulbvc::io::DvppJpegDecoder decoder(context, {1920, 1080, 2, true});
  uint64_t differing = 0, samples = 0, absolute_error = 0;
  unsigned maximum_error = 0;
  const auto verify = [&] {
    for (std::size_t i = 0; i < images.size(); ++i) {
      const auto actual = decoder.Decode(images[i]);
      // Visible rows only: height stride is 1088; padding is unspecified.
      for (int plane = 0; plane < 2; ++plane) {
        const std::size_t offset = plane ? 1920 * 1088 : 0;
        const int rows = plane ? 540 : 1080;
        for (int j = 0; j < rows * 1920; ++j) {
          const auto index = offset + j;
          const unsigned error = std::abs(int(actual[index]) - int(reference[i][index]));
          maximum_error = std::max(maximum_error, error);
          differing += error != 0;
          absolute_error += error;
          ++samples;
        }
      }
    }
  };
  verify();
  const auto begin = Clock::now();
  for (int i = 0; i < frames; ++i) decoder.DecodeDevice(images[i % images.size()]);
  const double elapsed = Ms(begin, Clock::now());
  verify();
  const auto& stats = decoder.stats();
  std::cout << "{\"path\":\"production_wrapper\",\"frames\":" << frames
            << ",\"frame_avg_ms\":" << elapsed / frames
            << ",\"normalization_avg_ms\":" << stats.normalization_ms / stats.frames
            << ",\"normalized_calls\":" << stats.normalized_frames
            << ",\"invalid_calls\":" << stats.invalid_frames
            << ",\"unsupported_calls\":" << stats.unsupported_frames
            << ",\"raw_vs_normalized_visible_max_abs\":" << maximum_error
            << ",\"raw_vs_normalized_visible_mae\":" << double(absolute_error) / samples
            << ",\"raw_vs_normalized_differing_samples\":" << differing << "}\n";
}

void CheckDetections(aclrtContext context, const std::vector<std::vector<uint8_t>>& images,
                     const char* model) {
  ulbvc::detection::Options options;
  options.model = model;
  options.labels = options.model.parent_path() / "labels.txt";
  ulbvc::detection::YoloWorker detector(context, options, 1920, 1080);
  ulbvc::io::DvppJpegDecoder original(context, {1920, 1080, 2, false});
  ulbvc::io::DvppJpegDecoder corrected(context, {1920, 1080, 2, true});
  const auto tight = [](const std::vector<uint8_t>& padded) {
    std::vector<uint8_t> result(1920 * 1080 * 3 / 2);
    std::memcpy(result.data(), padded.data(), 1920 * 1080);
    std::memcpy(result.data() + 1920 * 1080, padded.data() + 1920 * 1088, 1920 * 540);
    return result;
  };
  float min_iou = 1, max_confidence_delta = 0;
  std::size_t boxes = 0;
  for (std::size_t i = 0; i < images.size(); ++i) {
    const auto raw = tight(original.Decode(images[i]));
    const auto fixed = tight(corrected.Decode(images[i]));
    const auto before = detector.Submit(int(i * 2), raw).get();
    const auto after = detector.Submit(int(i * 2 + 1), fixed).get();
    ulbvc::Check(before.boxes.size() == after.boxes.size(), "JPEG correction changed box count");
    std::vector<bool> matched(after.boxes.size());
    for (const auto& a : before.boxes) {
      float best = -1;
      std::size_t selected = after.boxes.size();
      for (std::size_t j = 0; j < after.boxes.size(); ++j) {
        const auto& b = after.boxes[j];
        if (matched[j] || a.class_id != b.class_id || a.label != b.label) continue;
        const float w = std::max(0.0f, std::min(a.cx + a.width / 2, b.cx + b.width / 2) -
                                           std::max(a.cx - a.width / 2, b.cx - b.width / 2));
        const float h = std::max(0.0f, std::min(a.cy + a.height / 2, b.cy + b.height / 2) -
                                           std::max(a.cy - a.height / 2, b.cy - b.height / 2));
        const float intersection = w * h;
        const float iou =
            intersection / std::max(1e-6f, a.width * a.height + b.width * b.height - intersection);
        if (iou > best) {
          best = iou;
          selected = j;
        }
      }
      ulbvc::Check(selected < after.boxes.size() && best >= 0.98f,
                   "JPEG correction changed detection class/geometry");
      matched[selected] = true;
      min_iou = std::min(min_iou, best);
      max_confidence_delta =
          std::max(max_confidence_delta, std::abs(a.confidence - after.boxes[selected].confidence));
    }
    boxes += before.boxes.size();
    std::cout << "{\"fixture\":" << i << ",\"before_boxes\":" << before.boxes.size()
              << ",\"after_boxes\":" << after.boxes.size() << "}\n";
  }
  ulbvc::Check(boxes >= 5 && max_confidence_delta <= 0.01f,
               "detection control must include objects and retain confidence");
  ulbvc::Check(corrected.stats().normalized_frames > 0, "fixtures did not exercise correction");
  std::cout << "{\"detection_comparison\":\"passed\",\"boxes\":" << boxes
            << ",\"min_iou\":" << min_iou << ",\"max_confidence_delta\":" << max_confidence_delta
            << "}\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    ulbvc::Check(argc >= 4 && argc <= 6,
                 "usage: benchmark_dvpp_jpeg_decode IMAGE_A IMAGE_B FRAMES "
                 "[direct|dvpp|wrapper|detect] [YOLO_OM]");
    const std::string memory = argc >= 5 ? argv[4] : "direct";
    ulbvc::Check(
        memory == "direct" || memory == "dvpp" || memory == "wrapper" || memory == "detect",
        "invalid path");
    const bool device_input = memory == "dvpp";
    const int frames = std::stoi(argv[3]);
    ulbvc::Check(frames >= 2 && frames <= 120 && frames % 2 == 0,
                 "frames must be even, between 2 and 120");
    std::vector<std::vector<uint8_t>> images;
    uint32_t width = 0, height = 0;
    for (int index = 1; index <= 2; ++index) {
      std::ifstream input(argv[index], std::ios::binary);
      ulbvc::Check(input.good(), "cannot read JPEG benchmark fixture");
      images.emplace_back(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    }
    ulbvc::AclRuntime runtime(0);
    if (memory == "detect") {
      ulbvc::Check(argc == 6, "detect requires a YOLO model path");
      CheckDetections(runtime.context(), images, argv[5]);
      return 0;
    }
    std::vector<std::vector<uint8_t>> reference;
    for (const auto& image : images) {
      uint32_t current_width = 0, current_height = 0;
      int32_t components = 0;
      ulbvc::CheckAcl(acldvppJpegGetImageInfo(image.data(), image.size(), &current_width,
                                              &current_height, &components),
                      "read JPEG benchmark dimensions");
      if (!width) {
        width = current_width;
        height = current_height;
      }
      ulbvc::Check(current_width == width && current_height == height && width == 1920 &&
                       height == 1080 && components == 3,
                   "benchmark fixtures must both be RGB 1920x1080 JPEGs");
    }
    {
      ulbvc::io::DvppJpegDecoder decoder(
          runtime.context(), {static_cast<int>(width), static_cast<int>(height), 2, false});
      for (const auto& image : images) reference.push_back(decoder.Decode(image));
    }
    std::unique_ptr<BackgroundYolo> background;
    if (argc == 6) background = std::make_unique<BackgroundYolo>(runtime.context(), argv[5]);
    if (memory == "wrapper") {
      RunWrapper(runtime.context(), images, reference, frames);
      if (background) background->Finish();
      return 0;
    }
    Run(images, reference, width, height, 0, 1, frames, device_input);
    Run(images, reference, width, height, ACL_STREAM_FAST_SYNC, 1, frames, device_input);
    Run(images, reference, width, height, 0, 2, frames, device_input);
    Run(images, reference, width, height, ACL_STREAM_FAST_SYNC, 2, frames, device_input);
    if (background) background->Finish();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "JPEGD benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
