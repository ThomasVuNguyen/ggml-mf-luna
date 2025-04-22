cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target dir>
project("luna-inference" C CXX) #Include both C and C++
include(CheckIncludeFileCXX) 

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) # Make sure the output >
add_subdirectory(ggml) # Add files in ggml/ folder, which is where ggml.h exists

include_directories(${CMAKE_CURRENT_SOURCE_DIR}) # Include current directory - for >
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ggml/include) # Include GGML headers

# Create a library for tokenizer implementation
add_library(tokenizer_lib code/tokenizer_lib.cpp)
target_link_libraries(tokenizer_lib PRIVATE ggml)
target_compile_features(tokenizer_lib PRIVATE cxx_std_17)

# demo
set(CODE_TARGETS
    model
    embedding
    config
    example_matrix
    rms_norm
    embedding_pipeline
    ) # Target all the example files

foreach(TARGET ${CODE_TARGETS}) # For each example file
  add_executable(${TARGET} code/${TARGET}.cpp) # Compile to the demo/file.cpp
  target_link_libraries(${TARGET} PRIVATE ggml) # Include ggml library
  target_compile_features(${TARGET} PRIVATE cxx_std_17) # Requires C++ v17 standard>
endforeach()

# Link model with tokenizer library
target_link_libraries(model PRIVATE tokenizer_lib)
# Link embedding with tokenizer library
target_link_libraries(embedding PRIVATE tokenizer_lib)
# Link config with tokenizer library
target_link_libraries(config PRIVATE tokenizer_lib)
# Link rms_norm with tokenizer library
target_link_libraries(rms_norm PRIVATE tokenizer_lib)
# Link embedding_pipeline with tokenizer library
target_link_libraries(embedding_pipeline PRIVATE tokenizer_lib)