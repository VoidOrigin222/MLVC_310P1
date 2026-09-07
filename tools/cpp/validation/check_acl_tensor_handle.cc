#include <acl/acl.h>

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <thread>
#include <vector>

#include "mlvc/core/status.h"
#include "mlvc/core/tensor_handle.h"
#include "mlvc/runtime/acl_runtime.h"

int main(int argc, char** argv) {
  int device = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--device" && i + 1 < argc) {
      device = std::stoi(argv[++i]);
    } else {
      std::cerr << "usage: " << argv[0] << " [--device 0]\n";
      return 2;
    }
  }

  try {
    mlvc::AclRuntime runtime(device);
    mlvc::TensorCopyResult upload;
    mlvc::TensorCopyResult download;
    std::string residency;
    std::exception_ptr worker_error;
    std::thread worker([&] {
      try {
        runtime.MakeCurrent();
        mlvc::TensorHandle handle(mlvc::TensorShape(std::vector<int64_t>{1, 1, 1, 16}),
                                  mlvc::DataType::kUInt8);
        std::vector<uint8_t> input(handle.bytes());
        for (std::size_t i = 0; i < input.size(); ++i) {
          input[i] = static_cast<uint8_t>(i * 7 + 3);
        }
        handle.AttachCpuBuffer(input.data(), input.size(), true);
        upload = handle.EnsureAcl(runtime.stream());
        handle.MarkAclModified();
        download = handle.MaterializeToCpu("validation", runtime.stream());
        if (std::memcmp(handle.CpuView().data(), input.data(), input.size()) != 0) {
          throw mlvc::Error("TensorHandle ACL round trip mismatch");
        }
        residency = handle.ResidencyString();
      } catch (...) {
        worker_error = std::current_exception();
      }
    });
    worker.join();
    if (worker_error != nullptr) {
      std::rethrow_exception(worker_error);
    }
    std::cout << "upload_bytes=" << upload.bytes << "\n";
    std::cout << "download_bytes=" << download.bytes << "\n";
    std::cout << "residency=" << residency << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "check_acl_tensor_handle failed: " << error.what() << "\n";
    return 1;
  }
}
