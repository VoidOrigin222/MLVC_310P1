cmake_minimum_required(VERSION 3.22)

project(mlvc_acl_cpp LANGUAGES CXX)

option(MLVC_BUILD_APPS "Build MLVC command-line applications" ON)
set(MLVC_APP_SUFFIX "" CACHE STRING "Suffix for isolated deployment executables")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann" CACHE PATH "CANN package path")
set(ASCEND_CANN_LIB_DIR "${ASCEND_CANN_PACKAGE_PATH}/lib64")
set(ASCEND_CANN_ARCH_LIB_DIR "${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/lib64")
set(MLVC_EXTERNAL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH
    "Directory containing third_party and mlvc-main")
set(TOMLPLUSPLUS_INCLUDE_DIR
    "${MLVC_EXTERNAL_ROOT}/third_party/tomlplusplus/include"
    CACHE PATH "toml++ include directory")

include(CTest)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

set(MLVC_MSRTC_RANS_DIR
    "${MLVC_EXTERNAL_ROOT}/mlvc-main/mlvc-main/packages/msrtc_rans"
    CACHE PATH "Official MLVC rANS source directory")
add_library(mlvc_msrtc_rans STATIC
  "${MLVC_MSRTC_RANS_DIR}/src/EntropyCoder.cpp"
)
target_include_directories(mlvc_msrtc_rans
  PUBLIC "${MLVC_MSRTC_RANS_DIR}/include"
  PRIVATE "${MLVC_MSRTC_RANS_DIR}/private/include"
)

add_library(mlvc_common
  common/src/core/buffer.cc
  common/src/core/status.cc
  common/src/core/tensor.cc
  common/src/core/tensor_handle.cc
  common/src/framework/async_plan.cc
  common/src/framework/codec_graph_executor.cc
  common/src/framework/data_consumer.cc
  common/src/framework/data_producer.cc
  common/src/framework/entropy_worker.cc
  common/src/framework/streaming_pipeline.cc
  common/src/framework/thread_pool.cc
  common/src/framework/profiler.cc
  common/src/runtime/acl_runtime.cc
  common/src/runtime/model_manifest.cc
  common/src/runtime/stage_runtime.cc
  common/src/entropy/rans_core.cc
  common/src/transport/udp_message_transport.cc
  common/src/transport/udp_pacer.cc
  common/src/transport/rtp_mlvc.cc
  common/src/transport/rtp_message_receiver.cc
)
if(BUILD_TESTING)
  add_executable(test_rtp_mlvc tools/cpp/tests/test_rtp_mlvc.cc)
  target_link_libraries(test_rtp_mlvc PRIVATE mlvc_common)
  target_include_directories(test_rtp_mlvc PRIVATE common/include)
  add_test(NAME rtp_mlvc COMMAND test_rtp_mlvc)
  add_executable(test_rtp_reassembler tools/cpp/tests/test_rtp_reassembler.cc)
  target_link_libraries(test_rtp_reassembler PRIVATE mlvc_common)
  add_test(NAME rtp_reassembler COMMAND test_rtp_reassembler)
  add_executable(test_rtp_message_receiver_sequence
    tools/cpp/tests/test_rtp_message_receiver_sequence.cc
  )
  target_link_libraries(test_rtp_message_receiver_sequence PRIVATE mlvc_common)
  target_compile_options(test_rtp_message_receiver_sequence PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
  )
  add_test(NAME rtp_message_receiver_sequence
    COMMAND test_rtp_message_receiver_sequence
  )
  add_executable(test_rtp_sender tools/cpp/tests/test_rtp_sender.cc)
  target_link_libraries(test_rtp_sender PRIVATE mlvc_common)
  target_include_directories(test_rtp_sender PRIVATE common/include)
  add_test(NAME rtp_sender COMMAND test_rtp_sender)
endif()

target_include_directories(mlvc_common
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/common/include"
    "${ASCEND_CANN_PACKAGE_PATH}/include"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/common/src"
)

target_link_directories(mlvc_common PUBLIC
  "${ASCEND_CANN_LIB_DIR}"
  "${ASCEND_CANN_ARCH_LIB_DIR}"
)

target_link_libraries(mlvc_common PUBLIC
  ascendcl
  opapi
  nnopbase
  Threads::Threads
  ${CMAKE_DL_LIBS}
)

target_link_options(mlvc_common PUBLIC "-Wl,--disable-new-dtags")
set_target_properties(mlvc_common PROPERTIES
  BUILD_RPATH "${ASCEND_CANN_LIB_DIR};${ASCEND_CANN_ARCH_LIB_DIR}"
  INSTALL_RPATH "${ASCEND_CANN_LIB_DIR};${ASCEND_CANN_ARCH_LIB_DIR}"
)

target_compile_options(mlvc_common PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)

add_library(mlvc_codec
  mlvc/src/bitstream/bitstream.cc
  mlvc/src/codec/frame/decoder.cc
  mlvc/src/codec/frame/encoder.cc
  mlvc/src/codec/frame/reference.cc
  mlvc/src/codec/stage/stage_output_policy.cc
  mlvc/src/codec/stage/stage_runner.cc
  mlvc/src/codec/stage/stage_runtime_state.cc
  mlvc/src/codec/stage/stage_workspace.cc
  mlvc/src/codec/entropy/entropy_buffers.cc
  mlvc/src/codec/entropy/mlvc_entropy.cc
  mlvc/src/codec/rate_control/mlvc_rate_control.cc
  mlvc/src/codec/prior/prior_layout.cc
  mlvc/src/codec/tensor/tensor_utils.cc
  mlvc/src/codec/profile/allocation_profile.cc
  mlvc/src/codec/profile/allocation_tracking.cc
  mlvc/src/codec/profile/codec_profile.cc
  mlvc/src/codec/memory/workspace.cc
  mlvc/src/entropy/entropy_codec.cc
  mlvc/src/entropy/mlvc_official_entropy.cc
  mlvc/src/entropy/sidecar.cc
  mlvc/src/io/bitstream_io.cc
  mlvc/src/io/frame_source.cc
  mlvc/src/io/fp16_yuv444_to_nv12.cc
  mlvc/src/io/mlvc_bitstream.cc
  mlvc/src/io/udp_frame_transport.cc
  mlvc/src/io/video_io.cc
  mlvc/src/runtime/decode_prior_acl.cc
  mlvc/src/runtime/encode_index_acl.cc
  mlvc/src/runtime/fp16_yuv444_to_nv12_acl.cc
)

target_include_directories(mlvc_codec
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/mlvc/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/include"
    "${ASCEND_CANN_PACKAGE_PATH}/include"
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(mlvc_codec PUBLIC
  mlvc_common
  mlvc_msrtc_rans
  acl_dvpp
  ${OpenCV_LIBS}
)

target_compile_options(mlvc_codec PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set_source_files_properties(mlvc/src/io/fp16_yuv444_to_nv12.cc PROPERTIES
    COMPILE_OPTIONS "-march=armv8.2-a+fp16"
  )
endif()

add_library(mlvc_application STATIC
  application/src/pipeline/codec_frame_pipeline.cc
  application/src/input/frame_input_queue.cc
  application/src/progress.cc
  application/src/runtime/mlvc_codec_runtime.cc
  application/src/stream/mlvc/common.cc
  application/src/stream/mlvc/encode.cc
  application/src/stream/mlvc/encode/encode_setup.cc
  application/src/stream/mlvc/encode/encode_state.cc
  application/src/stream/mlvc/encode/encode_frame.cc
  application/src/stream/mlvc/encode/encode_output.cc
  application/src/stream/mlvc/entropy_decode.cc
  application/src/stream/mlvc/decode.cc
  application/src/cli/decode_config.cc
  application/src/cli/encode_config.cc
  application/src/cli/command_line.cc
  application/src/cli/runtime_environment.cc
)

target_include_directories(mlvc_application
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/application/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/mlvc/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/include"
    "${TOMLPLUSPLUS_INCLUDE_DIR}"
)

target_link_libraries(mlvc_application PUBLIC mlvc_codec)
target_compile_options(mlvc_application PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)

if(MLVC_BUILD_APPS)
  add_executable(benchmark_acl_stage tools/cpp/benchmarks/benchmark_acl_stage.cc)
  target_link_libraries(benchmark_acl_stage PRIVATE mlvc_common)
  target_compile_options(benchmark_acl_stage PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
  )

  add_executable(benchmark_acl_video tools/cpp/benchmarks/benchmark_acl_video.cc)
  target_link_libraries(benchmark_acl_video PRIVATE mlvc_application)
  target_compile_options(benchmark_acl_video PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
  )

  add_executable(benchmark_mlvc_official_loop tools/cpp/benchmarks/benchmark_mlvc_official_loop.cc)
  target_link_libraries(benchmark_mlvc_official_loop PRIVATE mlvc_application)
  target_compile_options(benchmark_mlvc_official_loop PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
  )

  add_executable(benchmark_acl_encoder tools/cpp/benchmarks/benchmark_acl_encoder.cc)
  target_link_libraries(benchmark_acl_encoder PRIVATE mlvc_application)
  target_compile_options(benchmark_acl_encoder PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
  )

  add_executable(benchmark_acl_decoder tools/cpp/benchmarks/benchmark_acl_decoder.cc)
  target_link_libraries(benchmark_acl_decoder PRIVATE mlvc_codec)
  target_compile_options(benchmark_acl_decoder PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
  )

  add_executable(extract_acl_video_frames tools/data/extract_acl_video_frames.cc)
  target_link_libraries(extract_acl_video_frames PRIVATE mlvc_codec)
  target_compile_options(extract_acl_video_frames PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
  )

  add_executable(mlvc_decode apps/decode_main.cc)
  set_target_properties(mlvc_decode PROPERTIES OUTPUT_NAME "mlvc_decode${MLVC_APP_SUFFIX}")
  target_link_libraries(mlvc_decode PRIVATE mlvc_application)
  target_compile_options(mlvc_decode PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
  )

  add_executable(mlvc_decode_entropy2_candidate apps/decode_main.cc)
  target_link_libraries(mlvc_decode_entropy2_candidate PRIVATE mlvc_application)
  target_compile_options(mlvc_decode_entropy2_candidate PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
  )


endif()
