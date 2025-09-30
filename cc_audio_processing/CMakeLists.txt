cmake_minimum_required(VERSION 3.14)

project(cc_audio_processing LANGUAGES CXX)

option(BUILD_EXAMPLE "Build example executable" ON)
option(USE_FETCHCONTENT "Fetch kaldi-native-fbank via FetchContent" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ----------------------------------------------------------------------------
# kaldi-native-fbank dependency
# You can either:
# 1) Set KALDI_NATIVE_FBANK_DIR to a local checkout, or
# 2) Enable USE_FETCHCONTENT to download it automatically.
# ----------------------------------------------------------------------------

if(DEFINED KALDI_NATIVE_FBANK_DIR)
  message(STATUS "Using local kaldi-native-fbank at ${KALDI_NATIVE_FBANK_DIR}")
  add_subdirectory(${KALDI_NATIVE_FBANK_DIR} ${CMAKE_BINARY_DIR}/kaldi-native-fbank)
  set(KNF_SRC ${KALDI_NATIVE_FBANK_DIR})
elseif(USE_FETCHCONTENT)
  include(FetchContent)
  FetchContent_Declare(
    kaldi_native_fbank
    GIT_REPOSITORY https://github.com/csukuangfj/kaldi-native-fbank.git
    GIT_TAG v1.22.2
  )
  # Disable Python bindings and tests for kaldi-native-fbank
  set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
  set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(kaldi_native_fbank)
  # Expose source dir for includes
  message(STATUS "kaldi_native_fbank source: ${kaldi_native_fbank_SOURCE_DIR}")
  set(KNF_SRC ${kaldi_native_fbank_SOURCE_DIR})
else()
  message(FATAL_ERROR "Please set KALDI_NATIVE_FBANK_DIR or enable USE_FETCHCONTENT")
endif()

# kaldi-native-fbank exports target: kaldi-native-fbank-core

add_library(audio_feature_processor
  src/audio_feature_processor.cc
)
target_include_directories(audio_feature_processor
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(audio_feature_processor
  PUBLIC kaldi-native-fbank-core
)
target_include_directories(audio_feature_processor
  PUBLIC $<TARGET_PROPERTY:kaldi-native-fbank-core,INTERFACE_INCLUDE_DIRECTORIES>
  PUBLIC ${KNF_SRC}
  PUBLIC ${KNF_SRC}/kaldi-native-fbank
  PUBLIC ${KNF_SRC}/csrc
)

if(BUILD_EXAMPLE)
  # Optional libsndfile for reading wav in example
  find_package(SndFile QUIET)

  add_executable(example_fbank_cmvn
    src/main_example.cc
  )
  target_link_libraries(example_fbank_cmvn
    PRIVATE audio_feature_processor
  )
  if(SNDFILE_FOUND)
    target_compile_definitions(example_fbank_cmvn PRIVATE USE_LIBSNDFILE)
    target_link_libraries(example_fbank_cmvn PRIVATE SndFile::sndfile)
  endif()

  add_executable(dump_fbank_cmvn
    src/dump_fbank_cmvn.cc
  )
  target_link_libraries(dump_fbank_cmvn PRIVATE audio_feature_processor)
  if(SNDFILE_FOUND)
    target_compile_definitions(dump_fbank_cmvn PRIVATE USE_LIBSNDFILE)
    target_link_libraries(dump_fbank_cmvn PRIVATE SndFile::sndfile)
  endif()

  add_executable(dump_fbank_cmvn_from_raw
    src/dump_fbank_cmvn_from_raw.cc
  )
  target_link_libraries(dump_fbank_cmvn_from_raw PRIVATE audio_feature_processor)
endif()


