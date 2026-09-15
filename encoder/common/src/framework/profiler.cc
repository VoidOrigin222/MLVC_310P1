#include "mlvc/framework/profiler.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>

#include "mlvc/core/status.h"

namespace mlvc {
namespace {

std::string EscapeJson(const std::string& value) {
  std::ostringstream output;
  for (char c : value) {
    switch (c) {
      case '\\':
        output << "\\\\";
        break;
      case '"':
        output << "\\\"";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20U) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<int>(static_cast<unsigned char>(c));
        } else {
          output << c;
        }
        break;
    }
  }
  return output.str();
}

int ThreadIdFor(const std::string& thread) {
  if (thread == "acl") {
    return 1;
  }
  if (thread == "copy") {
    return 2;
  }
  if (thread == "entropy") {
    return 3;
  }
  if (thread == "video_io") {
    return 4;
  }
  if (thread == "bitstream_io") {
    return 5;
  }
  if (thread == "allocation") {
    return 6;
  }
  if (thread == "sync") {
    return 7;
  }
  if (thread == "pipeline") {
    return 8;
  }
  return 0;
}

}  // namespace

Profiler::Profiler() { Reset(); }

void Profiler::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  origin_ = std::chrono::steady_clock::now();
  events_.clear();
}

void Profiler::ReserveEvents(std::size_t capacity) {
  std::lock_guard<std::mutex> lock(mutex_);
  events_.reserve(capacity);
}

void Profiler::AddEvent(const std::string& name, double start_ms, double duration_ms) {
  AddEventWithArgs(name, InferCategory(name), InferThread(name), start_ms, duration_ms, {});
}

void Profiler::AddEvent(const char* name, double start_ms, double duration_ms) {
  AddEvent(std::string(name), start_ms, duration_ms);
}

void Profiler::AddEventWithArgs(std::string name, std::string category, std::string thread,
                                double start_ms, double duration_ms,
                                std::vector<ProfileArgument> args) {
  std::lock_guard<std::mutex> lock(mutex_);
  events_.push_back(ProfileEvent{std::move(name), std::move(category), std::move(thread), start_ms,
                                 duration_ms, false, std::move(args)});
}

void Profiler::AddCounter(std::string name, std::string category, std::string thread,
                          double start_ms, std::vector<ProfileArgument> args) {
  std::lock_guard<std::mutex> lock(mutex_);
  events_.push_back(ProfileEvent{std::move(name), std::move(category), std::move(thread), start_ms,
                                 0.0, true, std::move(args)});
}

void Profiler::AddDurationNow(const std::string& name, std::chrono::steady_clock::time_point begin,
                              std::chrono::steady_clock::time_point end) {
  AddEvent(name, StartMs(begin), DurationMs(begin, end));
}

void Profiler::AddDurationNow(const char* name, std::chrono::steady_clock::time_point begin,
                              std::chrono::steady_clock::time_point end) {
  AddEvent(name, StartMs(begin), DurationMs(begin, end));
}

double Profiler::StartMs(std::chrono::steady_clock::time_point time) const {
  return std::chrono::duration<double, std::milli>(time - origin_).count();
}

double Profiler::DurationMs(std::chrono::steady_clock::time_point begin,
                            std::chrono::steady_clock::time_point end) const {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::map<std::string, double> Profiler::TotalsByName() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::map<std::string, double> totals;
  for (const ProfileEvent& event : events_) {
    if (!event.counter) {
      totals[event.name] += event.duration_ms;
    }
  }
  return totals;
}

void Profiler::WriteChromeTrace(const std::filesystem::path& path) const {
  std::vector<ProfileEvent> events;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    events = events_;
  }
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path);
  Check(output.good(), "failed to open profile trace output: " + path.string());
  output << "{\n  \"traceEvents\": [\n";
  bool first = true;
  auto write_metadata = [&](const std::string& name, int tid) {
    if (!first) {
      output << ",\n";
    }
    first = false;
    output << "    {\"ph\":\"M\",\"pid\":1,\"tid\":" << tid
           << ",\"name\":\"thread_name\",\"args\":{\"name\":\"" << EscapeJson(name) << "\"}}";
  };
  write_metadata("main", 0);
  write_metadata("acl", 1);
  write_metadata("copy", 2);
  write_metadata("entropy", 3);
  write_metadata("video_io", 4);
  write_metadata("bitstream_io", 5);
  write_metadata("allocation", 6);
  write_metadata("sync", 7);
  write_metadata("pipeline", 8);
  output << std::fixed << std::setprecision(3);
  for (const ProfileEvent& event : events) {
    output << ",\n    {\"ph\":\"" << (event.counter ? "C" : "X")
           << "\",\"pid\":1,\"tid\":" << ThreadIdFor(event.thread) << ",\"cat\":\""
           << EscapeJson(event.category) << "\",\"name\":\"" << EscapeJson(event.name)
           << "\",\"ts\":" << event.start_ms * 1000.0;
    if (!event.counter) {
      output << ",\"dur\":" << event.duration_ms * 1000.0;
    }
    output << ",\"args\":{";
    for (std::size_t i = 0; i < event.args.size(); ++i) {
      const ProfileArgument& arg = event.args[i];
      if (i > 0) {
        output << ",";
      }
      output << "\"" << EscapeJson(arg.key) << "\":";
      if (arg.type == ProfileArgType::kString) {
        output << "\"" << EscapeJson(arg.value) << "\"";
      } else if (arg.type == ProfileArgType::kBool) {
        output << arg.value;
      } else {
        output << arg.value;
      }
    }
    output << "}}";
  }
  output << "\n  ],\n  \"displayTimeUnit\": \"ms\"\n}\n";
  Check(output.good(), "failed to write profile trace output: " + path.string());
}

ProfileArgument Profiler::Arg(std::string key, std::string value) {
  return ProfileArgument{std::move(key), std::move(value), ProfileArgType::kString};
}

ProfileArgument Profiler::Arg(std::string key, const char* value) {
  return Arg(std::move(key), std::string(value == nullptr ? "" : value));
}

ProfileArgument Profiler::Arg(std::string key, uint64_t value) {
  return ProfileArgument{std::move(key), std::to_string(value), ProfileArgType::kNumber};
}

ProfileArgument Profiler::Arg(std::string key, int64_t value) {
  return ProfileArgument{std::move(key), std::to_string(value), ProfileArgType::kNumber};
}

ProfileArgument Profiler::Arg(std::string key, int value) {
  return ProfileArgument{std::move(key), std::to_string(value), ProfileArgType::kNumber};
}

ProfileArgument Profiler::Arg(std::string key, double value) {
  std::ostringstream output;
  output << std::setprecision(12) << value;
  return ProfileArgument{std::move(key), output.str(), ProfileArgType::kNumber};
}

ProfileArgument Profiler::BoolArg(std::string key, bool value) {
  return ProfileArgument{std::move(key), value ? "true" : "false", ProfileArgType::kBool};
}

std::string Profiler::InferCategory(const std::string& name) {
  if (name.rfind("acl.", 0) == 0) {
    return "acl";
  }
  if (name.rfind("copy.", 0) == 0) {
    return "copy";
  }
  if (name.rfind("entropy.", 0) == 0) {
    return "entropy";
  }
  if (name.rfind("video.", 0) == 0) {
    return "video_io";
  }
  if (name.rfind("bitstream.", 0) == 0) {
    return "bitstream_io";
  }
  if (name.rfind("allocation.", 0) == 0) {
    return "allocation";
  }
  if (name.rfind("sync.", 0) == 0) {
    return "sync";
  }
  if (name.rfind("pipeline.", 0) == 0) {
    return "pipeline";
  }
  return "cpu";
}

std::string Profiler::InferThread(const std::string& name) {
  const std::string category = InferCategory(name);
  if (category == "cpu") {
    return "main";
  }
  return category;
}

ScopedCpuTimer::ScopedCpuTimer(Profiler* profiler, std::string name)
    : profiler_(profiler), name_(std::move(name)), begin_(std::chrono::steady_clock::now()) {}

ScopedCpuTimer::~ScopedCpuTimer() {
  if (profiler_ == nullptr) {
    return;
  }
  profiler_->AddDurationNow(name_, begin_, std::chrono::steady_clock::now());
}

}  // namespace mlvc
