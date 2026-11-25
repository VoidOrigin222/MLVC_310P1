#ifndef ULBVC_APPLICATION_CLI_RUNTIME_ENVIRONMENT_H_
#define ULBVC_APPLICATION_CLI_RUNTIME_ENVIRONMENT_H_

namespace ulbvc {

// Prepares CANN/custom-op libraries for the current executable. The function
// may re-exec the process once after updating LD_LIBRARY_PATH.
void PrepareAscendRuntimeEnvironment(char** argv);

}  // namespace ulbvc

#endif  // ULBVC_APPLICATION_CLI_RUNTIME_ENVIRONMENT_H_
