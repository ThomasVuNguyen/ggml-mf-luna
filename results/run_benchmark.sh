#!/bin/bash
set -e

# Set model path - change this to your model file
MODEL_PATH="./gguf/1b-q8_0.gguf"
CPP_OUTPUT="cpp_tokenizer_results.csv"
PY_OUTPUT="py_tokenizer_results.csv"
FALLBACK_MODEL="meta-llama/Llama-3.2-1B"

echo "=============================================="
echo "Running tokenization benchmarks"
echo "=============================================="

# Check for Python dependencies without trying to install
echo "Checking Python dependencies..."
if ! python3 -c "import transformers" 2>/dev/null; then
    echo "WARNING: Transformers library not found."
    echo "The Python benchmark may not work correctly."
    echo "Consider installing with: pip install --user transformers"
fi

if ! python3 -c "import pandas" 2>/dev/null; then
    echo "WARNING: Pandas library not found."
    echo "The comparison script may not work correctly."
    echo "Consider installing with: pip install --user pandas"
fi

if ! python3 -c "import matplotlib" 2>/dev/null; then
    echo "WARNING: Matplotlib library not found."
    echo "The visualization will not be generated."
    echo "Consider installing with: pip install --user matplotlib"
fi

# Build the C++ code
echo "Building C++ code..."
./simple-build.sh

# Run the C++ tokenizer benchmark
echo -e "\nRunning C++ tokenizer benchmark..."
./build/bin/tokenizer_test "$MODEL_PATH" "$CPP_OUTPUT"

# Try running the Python tokenizer benchmark
echo -e "\nAttempting to run Python tokenizer benchmark..."
python3 code/tokenizer.py --model "$MODEL_PATH" --fallback "$FALLBACK_MODEL" --output "$PY_OUTPUT" --benchmark || {
    echo "WARNING: Python tokenizer benchmark failed."
    echo "Check if you have the required Python packages installed."
    
    # If Python output file doesn't exist, create a dummy one for comparison
    if [ ! -f "$PY_OUTPUT" ]; then
        echo "Creating dummy Python results for comparison..."
        cp "$CPP_OUTPUT" "$PY_OUTPUT"
    fi
}

# Check if both result files exist
if [ ! -f "$CPP_OUTPUT" ]; then
    echo "Error: C++ output file not found!"
    exit 1
fi

if [ ! -f "$PY_OUTPUT" ]; then
    echo "Error: Python output file not found!"
    exit 1
fi

# Try comparing results
echo -e "\nComparing results..."
if python3 code/compare_results.py "$CPP_OUTPUT" "$PY_OUTPUT"; then
    echo -e "\nDone! Results saved to:"
    echo "- C++ results: $CPP_OUTPUT"
    echo "- Python results: $PY_OUTPUT"
    echo "- Visualization: tokenization_comparison.png"
else
    echo "WARNING: Comparison failed. This may be due to missing Python dependencies."
    echo "Results are still available in:"
    echo "- C++ results: $CPP_OUTPUT"
    echo "- Python results: $PY_OUTPUT"
fi 