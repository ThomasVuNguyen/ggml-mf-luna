cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target dir
project("luna-inference" C CXX) #Include both C and C++
include(CheckIncludeFileCXX) 

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) # Make sure the output
add_subdirectory(ggml) # Add files in ggml/ folder, which is where ggml.h exists

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}) # Include current directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ggml/include) # Include GGML headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ggml/examples) # Include examples directory where llama.h might be

# Create the tokenizer library
add_library(tokenizer_lib STATIC code/tokenizer.cpp)
target_link_libraries(tokenizer_lib PRIVATE ggml)
target_compile_features(tokenizer_lib PRIVATE cxx_std_17)

# Create a library for embedding implementation
add_library(embedding_lib STATIC code/embedding.cpp)
target_compile_features(embedding_lib PRIVATE cxx_std_11)

# Create standalone tokenizer executable
add_executable(tokenizer code/tokenizer.cpp)
target_link_libraries(tokenizer PRIVATE ggml)
target_compile_features(tokenizer PRIVATE cxx_std_17)
target_compile_definitions(tokenizer PRIVATE TOKENIZER_STANDALONE=1)

# Create tokenizer_test executable
add_executable(tokenizer_test code/tokenizer_test.cpp)
target_link_libraries(tokenizer_test PRIVATE tokenizer_lib ggml)
target_compile_features(tokenizer_test PRIVATE cxx_std_17)

# Create model executable
add_executable(model code/model.cpp)
target_link_libraries(model PRIVATE tokenizer_lib embedding_lib ggml)
target_compile_features(model PRIVATE cxx_std_17)
