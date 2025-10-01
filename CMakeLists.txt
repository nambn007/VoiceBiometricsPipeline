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

add_executable(voice_bio
    main.cc
    voice_bio_pipeline.cc
    ecapa_engine.cc
    mel_extractor.cc
    vad_engine.cc
    vad_engine_2.cpp
)

add_executable(voice_bio_pipeline
    main_2.cc 
    VoiceBiometricsPipeline.cpp
    VoiceBiometricsPipeline.h
    ecapa_engine.cc
    mel_extractor.cc
    vad_engine_2.cpp
)

target_include_directories(voice_bio PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(voice_bio PRIVATE 
    onnxruntime
    kaldi-native-fbank-core
)

target_link_libraries(voice_bio_pipeline PRIVATE 
    onnxruntime
    kaldi-native-fbank-core
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

