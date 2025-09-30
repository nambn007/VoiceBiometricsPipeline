cmake_minimum_required(VERSION 3.14)
project(voice_bio_pipeline_cc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime (adjust these paths if needed)
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/../examples/onnxruntime-linux-x64-1.12.1")
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

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
)

target_include_directories(voice_bio PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(voice_bio PRIVATE 
    onnxruntime
    kaldi-native-fbank-core
)

if(UNIX)
  target_link_libraries(voice_bio PRIVATE pthread dl)
endif()

message(STATUS "ONNXRuntime root: ${ONNXRUNTIME_ROOT}")

