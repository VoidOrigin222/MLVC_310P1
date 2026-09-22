#include "mlvc/transport/udp_message_transport.h"

#include <cassert>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>

int main() {
  int send_count = 0;
  std::mutex hook_mutex;
  std::condition_variable hook_cv;
  bool first_send_started = false;
  bool release_first_send = false;
  mlvc::transport::UdpSendOptions options;
  options.before_send = [&] {
    std::unique_lock<std::mutex> lock(hook_mutex);
    ++send_count;
    if (send_count == 1) {
      first_send_started = true;
      hook_cv.notify_one();
      hook_cv.wait(lock, [&] { return release_first_send; });
    } else if (send_count == 2) {
      throw std::runtime_error("injected final UDP send failure");
    }
  };

  bool flush_failed = false;
  {
    mlvc::transport::UdpMessageSender sender("127.0.0.1", 9, "flush error test", options);
    sender.Send({1});
    {
      std::unique_lock<std::mutex> lock(hook_mutex);
      hook_cv.wait(lock, [&] { return first_send_started; });
    }
    sender.Send({2});
    {
      std::lock_guard<std::mutex> lock(hook_mutex);
      release_first_send = true;
    }
    hook_cv.notify_one();
    try {
      sender.Flush();
    } catch (const std::exception& error) {
      flush_failed = std::string(error.what()).find("injected final UDP send failure") !=
                     std::string::npos;
    }
  }
  assert(flush_failed);
  return 0;
}
