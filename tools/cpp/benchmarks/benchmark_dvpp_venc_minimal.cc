#include <acl/acl.h>
#include <acl/ops/acl_dvpp.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <condition_variable>
#include <exception>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <string>
#include <stdexcept>
#include <thread>

namespace {

struct Options {
  int device = 0;
  int width = 1920;
  int height = 1088;
  int level = 0;
};

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    if (index + 1 >= argc) throw std::runtime_error("missing value for " + arg);
    const std::string value = argv[++index];
    if (arg == "--device") options.device = std::stoi(value);
    else if (arg == "--width") options.width = std::stoi(value);
    else if (arg == "--height") options.height = std::stoi(value);
    else if (arg == "--level") options.level = std::stoi(value);
    else throw std::runtime_error("unknown option: " + arg);
  }
  if (options.width <= 0 || options.height <= 0 || options.level < 0 || options.level > 2) {
    throw std::runtime_error("invalid dimensions or level (level must be 0, 1, or 2)");
  }
  return options;
}

void Check(aclError status, const std::string& operation) {
  if (status == ACL_ERROR_NONE) return;
  std::cerr << operation << " failed: ret=" << status;
  const char* detail = aclGetRecentErrMsg();
  if (detail != nullptr && detail[0] != '\0') std::cerr << ", detail=" << detail;
  std::cerr << "\n";
  throw std::runtime_error(operation + " failed");
}

void NoopCallback(acldvppPicDesc*, acldvppStreamDesc*, void*) {}

class ReportThread {
 public:
  explicit ReportThread(aclrtContext context) : context_(context) {
    thread_ = std::thread([this] {
      try {
        Check(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext report thread");
        {
          std::lock_guard<std::mutex> lock(mutex_);
          ready_ = true;
        }
        condition_.notify_one();
        while (running_.load(std::memory_order_acquire)) {
          const aclError status = aclrtProcessReport(100);
          if (status != ACL_ERROR_NONE && status != ACL_ERROR_RT_REPORT_TIMEOUT) {
            std::lock_guard<std::mutex> lock(mutex_);
            error_ = std::make_exception_ptr(std::runtime_error("aclrtProcessReport failed"));
            running_.store(false, std::memory_order_release);
            break;
          }
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        error_ = std::current_exception();
        ready_ = true;
        running_.store(false, std::memory_order_release);
      }
      condition_.notify_one();
    });
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return ready_; });
    if (error_ != nullptr) std::rethrow_exception(error_);
  }

  ~ReportThread() {
    running_.store(false, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
  }

  uint64_t thread_handle() {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(thread_.native_handle()));
  }

 private:
  aclrtContext context_ = nullptr;
  std::thread thread_;
  std::atomic<bool> running_{true};
  std::mutex mutex_;
  std::condition_variable condition_;
  bool ready_ = false;
  std::exception_ptr error_;
};

}  // namespace

int main(int argc, char** argv) {
  aclvencChannelDesc* desc = nullptr;
  bool channel_created = false;
  aclrtContext context = nullptr;
  try {
    const Options options = ParseOptions(argc, argv);
    Check(aclInit(nullptr), "aclInit");
    Check(aclrtSetDevice(options.device), "aclrtSetDevice");
    Check(aclrtCreateContext(&context, options.device), "aclrtCreateContext");
    Check(aclrtSetCurrentContext(context), "aclrtSetCurrentContext");
    ReportThread report(context);

    desc = aclvencCreateChannelDesc();
    if (desc == nullptr) throw std::runtime_error("aclvencCreateChannelDesc returned null");
    Check(aclvencSetChannelDescThreadId(desc, report.thread_handle()), "set pthread handle");
    Check(aclvencSetChannelDescCallback(desc, &NoopCallback), "set callback");
    Check(aclvencSetChannelDescEnType(desc, H264_MAIN_LEVEL), "set H.264 type");
    Check(aclvencSetChannelDescPicFormat(desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420),
          "set NV12 format");
    Check(aclvencSetChannelDescPicWidth(desc, static_cast<uint32_t>(options.width)),
          "set width");
    Check(aclvencSetChannelDescPicHeight(desc, static_cast<uint32_t>(options.height)),
          "set height");
    if (options.level >= 1) {
      Check(aclvencSetChannelDescKeyFrameInterval(desc, options.level == 2 ? 1 : 96),
            "set GOP");
    }
    if (options.level >= 2) {
      // Match the official CANN VENC sample: CBR, GOP=1, 10 Mbps.
      Check(aclvencSetChannelDescRcMode(desc, 2), "set CBR mode");
      Check(aclvencSetChannelDescMaxBitRate(desc, 10000), "set max bitrate");
    }
    std::cout << "attempt level=" << options.level << " dimensions=" << options.width << "x"
              << options.height << "\n";
    Check(aclvencCreateChannel(desc), "aclvencCreateChannel");
    channel_created = true;
    std::cout << "venc_channel_created=1\n";
    Check(aclvencDestroyChannel(desc), "aclvencDestroyChannel");
    channel_created = false;
    Check(aclvencDestroyChannelDesc(desc), "aclvencDestroyChannelDesc");
    desc = nullptr;
    Check(aclrtDestroyContext(context), "aclrtDestroyContext");
    context = nullptr;
    Check(aclrtResetDevice(options.device), "aclrtResetDevice");
    Check(aclFinalize(), "aclFinalize");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "benchmark_dvpp_venc_minimal failed: " << error.what() << "\n";
    if (channel_created && desc != nullptr) (void)aclvencDestroyChannel(desc);
    if (desc != nullptr) (void)aclvencDestroyChannelDesc(desc);
    if (context != nullptr) (void)aclrtDestroyContext(context);
    return 1;
  }
}
