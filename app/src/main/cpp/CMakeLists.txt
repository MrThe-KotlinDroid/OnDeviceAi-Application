cmake_minimum_required(VERSION 3.22.1)
project("android_app_jni")

# FIX: Globally link Android logging so protobuf doesn't crash
link_libraries(log)

# 1. Fetch the official Google SentencePiece C++ source code
include(FetchContent)
FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece.git
    GIT_TAG master
)

# 2. Disable desktop binaries/tests
set(SPM_ENABLE_SHARED OFF CACHE BOOL "Disable shared lib" FORCE)
set(SPM_BUILD_TEST OFF CACHE BOOL "Disable tests" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "Disable tcmalloc" FORCE)

FetchContent_MakeAvailable(sentencepiece)

# 3. Create our custom C++ bridge library
add_library(
    tokenizer_bridge
    SHARED
    tokenizer_bridge.cpp
)

# 4. Link everything together
target_link_libraries(
    tokenizer_bridge
    sentencepiece
    android
    log
)

# 5. Include the downloaded headers
target_include_directories(
    tokenizer_bridge
    PRIVATE
    ${sentencepiece_SOURCE_DIR}/src
)