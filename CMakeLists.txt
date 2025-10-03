cmake_minimum_required(VERSION 3.14)
project(voice_bio_pipeline_cc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
if(EXISTS "${ONNXRUNTIME_ROOT}/include" AND EXISTS "${ONNXRUNTIME_ROOT}/lib")
  include_directories(${ONNXRUNTIME_ROOT}/include)
  link_directories(${ONNXRUNTIME_ROOT}/lib)
else()
  message(FATAL_ERROR "ONNX Runtime not found under ${ONNXRUNTIME_ROOT}. Run bash setup_onnxruntime.sh to install.")
endif()

# Kaldi Native Fbank
set(KALDI_FBANK_ROOT "${CMAKE_SOURCE_DIR}/third_party/kaldi-native-fbank")
include_directories(${KALDI_FBANK_ROOT})
link_directories(${KALDI_FBANK_ROOT}/build/lib)


find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED fftw3)
include_directories(${FFTW_INCLUDE_DIRS})
link_directories(${FFTW_LIBRARY_DIRS})

include_directories(${CMAKE_SOURCE_DIR})

add_executable(voice_bio
    main.cpp
    audio_features.cpp
    voice_bio_pipeline.cpp
    ecapa_engine.cpp
    fbank.cpp
    wav.cpp
    vad_engine.cpp
)

add_executable(test 
  test/test_fbank.cpp
  fbank.cpp
  audio_features.cpp
)

target_include_directories(voice_bio PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(voice_bio PRIVATE 
    onnxruntime
    kaldi-native-fbank-core
    fftw3
    fftw3f
)

target_link_libraries(test PRIVATE 
    onnxruntime
    kaldi-native-fbank-core
    fftw3
    fftw3f
)

# Ensure the built binary can locate ONNX Runtime at runtime
set_target_properties(voice_bio PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
  INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
)

if(UNIX)
  target_link_libraries(voice_bio PRIVATE pthread dl)
endif()

message(STATUS "ONNXRuntime root: ${ONNXRUNTIME_ROOT}")

