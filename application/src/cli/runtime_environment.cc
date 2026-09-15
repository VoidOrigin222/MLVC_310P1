#include <mlvc/application/cli/runtime_environment.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

namespace mlvc {
namespace {

constexpr char kRuntimeEnvBootstrapped[] = "MLVC_CODEC_RUNTIME_ENV_BOOTSTRAPPED";

void AppendIfDirectory(const std::filesystem::path& path,
                       std::vector<std::filesystem::path>* paths) {
  std::error_code error;
  if (std::filesystem::is_directory(path, error)) {
    paths->push_back(path);
  }
}

std::filesystem::path ExecutablePath(const char* argv0) {
  std::error_code error;
  std::filesystem::path path = std::filesystem::canonical("/proc/self/exe", error);
  if (!error) {
    return path;
  }
  return std::filesystem::absolute(argv0, error);
}

std::string BuildLibraryPath(const std::vector<std::filesystem::path>& paths) {
  std::string library_path;
  for (const std::filesystem::path& path : paths) {
    if (!library_path.empty()) {
      library_path += ":";
    }
    library_path += path.string();
  }
  const char* existing = std::getenv("LD_LIBRARY_PATH");
  if (existing != nullptr && existing[0] != '\0') {
    if (!library_path.empty()) {
      library_path += ":";
    }
    library_path += existing;
  }
  return library_path;
}

void SetPriorOpapiLibraryIfPresent(const std::filesystem::path& repo_root) {
  constexpr char kPriorOpapiEnv[] = "MLVC_PRIOR_OPAPI_LIB";
  if (std::getenv(kPriorOpapiEnv) != nullptr) {
    return;
  }
  const std::filesystem::path library = repo_root / "output" / "custom_opp" / "mlvc_prior_ops" /
                                        "vendors" / "ulbvc" / "op_api" / "lib" / "libcust_opapi.so";
  std::error_code error;
  if (std::filesystem::is_regular_file(library, error)) {
    setenv(kPriorOpapiEnv, library.c_str(), 1);
  }
}

}  // namespace

void PrepareAscendRuntimeEnvironment(char** argv) {
  if (std::getenv(kRuntimeEnvBootstrapped) != nullptr) {
    return;
  }

  const std::filesystem::path executable = ExecutablePath(argv[0]);
  const std::filesystem::path repo_root = executable.parent_path().parent_path();
  const std::filesystem::path cann_home = std::getenv("CANN_HOME") != nullptr
                                              ? std::filesystem::path(std::getenv("CANN_HOME"))
                                              : std::filesystem::path("/usr/local/Ascend/cann");

  std::vector<std::filesystem::path> library_dirs;
  AppendIfDirectory(cann_home / "lib64", &library_dirs);
  AppendIfDirectory(cann_home / "runtime" / "lib64", &library_dirs);
  AppendIfDirectory(repo_root / "output" / "custom_opp" / "wsiluchunkadd" / "vendors" / "ulbvc" /
                        "op_api" / "lib",
                    &library_dirs);
  AppendIfDirectory(repo_root / "output" / "custom_opp" / "mlvc_prior_ops" / "vendors" / "ulbvc" /
                        "op_api" / "lib",
                    &library_dirs);
  SetPriorOpapiLibraryIfPresent(repo_root);

  const std::string library_path = BuildLibraryPath(library_dirs);
  if (library_path.empty()) {
    return;
  }
  setenv(kRuntimeEnvBootstrapped, "1", 1);
  setenv("LD_LIBRARY_PATH", library_path.c_str(), 1);
  execv(executable.c_str(), argv);
  std::cerr << "warning: failed to restart with ACL runtime library path; continuing\n";
}

}  // namespace mlvc
