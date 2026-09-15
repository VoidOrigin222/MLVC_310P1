#ifndef MLVC_APPLICATION_CLI_RUNTIME_ENVIRONMENT_H_
#define MLVC_APPLICATION_CLI_RUNTIME_ENVIRONMENT_H_

namespace mlvc {

// Prepares CANN/custom-op libraries for the current executable. The function
// may re-exec the process once after updating LD_LIBRARY_PATH.
void PrepareAscendRuntimeEnvironment(char** argv);

}  // namespace mlvc

#endif  // MLVC_APPLICATION_CLI_RUNTIME_ENVIRONMENT_H_
