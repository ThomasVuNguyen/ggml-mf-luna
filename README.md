# ggml-mf-luna
Luna Inference, now powered by GGML

## Overview
This project demonstrates how to extract embeddings from a GGUF model in C++ using llama.cpp, and compares the results with Python.

## Setup

```bash
git submodule add github.ggml ggml/
```

## Prerequisites

- CMake (>= 3.10)
- C++ compiler with C++11 support
- GGUF model file (LLaMA model)

## Building

1. Create a build directory:
   ```
   mkdir build
   cd build
   ```

2. Configure with CMake:
   ```
   cmake ..
   ```

3. Build:
   ```
   cmake --build . --config Release
   ```

## Running

### Benchmark tokenizer
```bash
./run_benchmark.sh
```

### Embedding Demo
Run the embedding demo:
```
./bin/model
```

This will:
1. Load the tokenizer
2. Tokenize the text "Thomas the Maker"
3. Extract embeddings for each token
4. Display raw and normalized embeddings
5. Save the results to `output_cpp.json`

## Comparing with Python

Run the Python script to compare:
```
python code/embedding.py
```

The outputs of the C++ and Python implementations should be consistent.

## Why Embeddings May Differ

If you encounter different embeddings between C++ and Python implementations, it's likely due to:

1. The C++ implementation using different/mock data instead of the actual embeddings from the model
2. The Python implementation normalizing embeddings, while the C++ implementation doesn't
3. Using incorrect embedding dimensions in one implementation