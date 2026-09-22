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
set(MLVC_VENDOR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
set(TOMLPLUSPLUS_INCLUDE_DIR "${MLVC_VENDOR_ROOT}/tomlplusplus/include")

include(CTest)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

set(MLVC_MSRTC_RANS_DIR "${MLVC_VENDOR_ROOT}/msrtc_rans")
if(NOT EXISTS "${TOMLPLUSPLUS_INCLUDE_DIR}/toml++/toml.h")
  message(FATAL_ERROR "Missing vendored toml++ headers under ${TOMLPLUSPLUS_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${MLVC_MSRTC_RANS_DIR}/src/EntropyCoder.cpp")
  message(FATAL_ERROR "Missing vendored MLVC rANS source under ${MLVC_MSRTC_RANS_DIR}")
endif()
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
  add_executable(test_udp_sender_flush_error tools/cpp/tests/test_udp_sender_flush_error.cc)
  target_link_libraries(test_udp_sender_flush_error PRIVATE mlvc_common)
  target_include_directories(test_udp_sender_flush_error PRIVATE common/include)
  add_test(NAME udp_sender_flush_error COMMAND test_udp_sender_flush_error)
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
  mlvc/src/io/dvpp_jpeg_encoder.cc
  mlvc/src/io/dvpp_h264_encoder.cc
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

target_compile_definitions(mlvc_codec PRIVATE ENABLE_DVPP_INTERFACE)

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

  add_executable(benchmark_dvpp_jpeg tools/cpp/benchmarks/benchmark_dvpp_jpeg.cc)
  target_link_libraries(benchmark_dvpp_jpeg PRIVATE mlvc_codec)
  target_compile_definitions(benchmark_dvpp_jpeg PRIVATE ENABLE_DVPP_INTERFACE)
  target_compile_options(benchmark_dvpp_jpeg PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
  )

  add_executable(benchmark_dvpp_h264 tools/cpp/benchmarks/benchmark_dvpp_h264.cc)
  target_link_libraries(benchmark_dvpp_h264 PRIVATE mlvc_codec)
  target_compile_definitions(benchmark_dvpp_h264 PRIVATE ENABLE_DVPP_INTERFACE)
  target_compile_options(benchmark_dvpp_h264 PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
  )

  add_executable(benchmark_dvpp_venc_minimal
    tools/cpp/benchmarks/benchmark_dvpp_venc_minimal.cc
  )
  target_link_libraries(benchmark_dvpp_venc_minimal PRIVATE ascendcl acl_dvpp Threads::Threads)
  target_compile_definitions(benchmark_dvpp_venc_minimal PRIVATE ENABLE_DVPP_INTERFACE)
  target_include_directories(benchmark_dvpp_venc_minimal PRIVATE
    "${ASCEND_CANN_PACKAGE_PATH}/include"
  )
  target_link_directories(benchmark_dvpp_venc_minimal PRIVATE "${ASCEND_CANN_LIB_DIR}")
  target_compile_options(benchmark_dvpp_venc_minimal PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
  )

  add_executable(extract_acl_video_frames tools/data/extract_acl_video_frames.cc)
  target_link_libraries(extract_acl_video_frames PRIVATE mlvc_codec)
  target_compile_options(extract_acl_video_frames PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
  )

  add_executable(mlvc_encode apps/encode_main.cc)
  set_target_properties(mlvc_encode PROPERTIES OUTPUT_NAME "mlvc_encode${MLVC_APP_SUFFIX}")
  target_link_libraries(mlvc_encode PRIVATE mlvc_application)
  target_compile_options(mlvc_encode PRIVATE
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

  if(BUILD_TESTING)
    add_executable(test_p1_model_loading tools/cpp/tests/test_p1_model_loading.cc)
    target_link_libraries(test_p1_model_loading PRIVATE mlvc_common)
    set(MLVC_TEST_MANIFEST "" CACHE FILEPATH "Manifest for optional on-device model tests")
    if(MLVC_TEST_MANIFEST)
      add_test(NAME p1_model_loading COMMAND test_p1_model_loading "${MLVC_TEST_MANIFEST}")
    endif()
    add_executable(test_fp16_yuv444_to_nv12
      tools/cpp/tests/test_fp16_yuv444_to_nv12.cc
    )
    target_link_libraries(test_fp16_yuv444_to_nv12 PRIVATE mlvc_codec)
    target_compile_options(test_fp16_yuv444_to_nv12 PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
    add_test(NAME fp16_yuv444_to_nv12 COMMAND test_fp16_yuv444_to_nv12)

    add_executable(test_fp16_yuv444_to_nv12_acl
      tools/cpp/tests/test_fp16_yuv444_to_nv12_acl.cc
    )
    target_link_libraries(test_fp16_yuv444_to_nv12_acl PRIVATE mlvc_codec)
    target_compile_options(test_fp16_yuv444_to_nv12_acl PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )

    add_executable(test_rtsp_video_publisher
      tools/cpp/tests/test_rtsp_video_publisher.cc
    )
    target_link_libraries(test_rtsp_video_publisher PRIVATE mlvc_codec)
    target_compile_options(test_rtsp_video_publisher PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
    add_test(NAME rtsp_video_publisher COMMAND test_rtsp_video_publisher)

    add_executable(test_udp_initial_wait
      tools/cpp/tests/test_udp_initial_wait.cc
    )
    target_link_libraries(test_udp_initial_wait PRIVATE mlvc_common)
    target_compile_options(test_udp_initial_wait PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
    add_test(NAME udp_initial_wait COMMAND test_udp_initial_wait)

    add_executable(test_udp_receiver_burst
      tools/cpp/tests/test_udp_receiver_burst.cc
    )
    target_link_libraries(test_udp_receiver_burst PRIVATE mlvc_common)
    target_compile_options(test_udp_receiver_burst PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
    add_test(NAME udp_receiver_burst COMMAND test_udp_receiver_burst)

    add_executable(test_stage_output_policy
      tools/cpp/validation/check_stage_output_policy.cc
    )
    target_link_libraries(test_stage_output_policy PRIVATE mlvc_codec)
    target_compile_options(test_stage_output_policy PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
    add_test(NAME stage_output_policy COMMAND test_stage_output_policy)

    add_executable(check_acl_tensor_handle
      tools/cpp/validation/check_acl_tensor_handle.cc
    )
    target_link_libraries(check_acl_tensor_handle PRIVATE mlvc_common)
    target_compile_options(check_acl_tensor_handle PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )

    add_executable(test_ordered_future_window
      tools/cpp/tests/test_ordered_future_window.cc
    )
    target_link_libraries(test_ordered_future_window PRIVATE mlvc_application)
    target_compile_options(test_ordered_future_window PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
    add_test(NAME ordered_future_window COMMAND test_ordered_future_window)

    add_executable(test_reference_reset_schedule
      tools/cpp/tests/test_reference_reset_schedule.cc
    )
    target_link_libraries(test_reference_reset_schedule PRIVATE mlvc_application)
    target_compile_options(test_reference_reset_schedule PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
    add_test(NAME reference_reset_schedule COMMAND test_reference_reset_schedule)

    add_executable(test_streaming_pipeline_order
      tools/cpp/tests/test_streaming_pipeline_order.cc
    )
    target_link_libraries(test_streaming_pipeline_order PRIVATE mlvc_common)
    target_compile_options(test_streaming_pipeline_order PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
    add_test(NAME streaming_pipeline_order COMMAND test_streaming_pipeline_order)

    add_executable(test_mlvc_bitstream_validation
      tools/cpp/tests/test_mlvc_bitstream_validation.cc
    )
    target_link_libraries(test_mlvc_bitstream_validation PRIVATE mlvc_codec)
    target_compile_options(test_mlvc_bitstream_validation PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
    add_test(NAME mlvc_bitstream_validation COMMAND test_mlvc_bitstream_validation)

    add_executable(test_sidecar_validation tools/cpp/tests/test_sidecar_validation.cc)
    target_link_libraries(test_sidecar_validation PRIVATE mlvc_codec)
    target_compile_options(test_sidecar_validation PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
    add_test(NAME sidecar_validation COMMAND test_sidecar_validation)

    add_executable(test_entropy_decode_equivalence
      tools/cpp/tests/test_entropy_decode_equivalence.cc
    )
    target_link_libraries(test_entropy_decode_equivalence PRIVATE mlvc_application)
    target_compile_options(test_entropy_decode_equivalence PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )

    add_executable(test_decode_config tools/cpp/tests/test_decode_config.cc)
    target_link_libraries(test_decode_config PRIVATE mlvc_application)
    target_compile_options(test_decode_config PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
    if(MLVC_TEST_MANIFEST)
      add_test(NAME decode_config COMMAND test_decode_config "${MLVC_TEST_MANIFEST}")
    else()
      add_test(NAME decode_config COMMAND test_decode_config
        "${CMAKE_CURRENT_SOURCE_DIR}/../mlvc1080p/manifest_310P1.json")
    endif()

    add_executable(test_encode_config tools/cpp/tests/test_encode_config.cc)
    target_link_libraries(test_encode_config PRIVATE mlvc_application)
    target_compile_options(test_encode_config PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
    add_test(NAME encode_config COMMAND test_encode_config)
  endif()

endif()
