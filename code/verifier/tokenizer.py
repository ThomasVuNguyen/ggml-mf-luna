#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import argparse
import time
import csv
import os

# Number of iterations for more accurate timing
TIMING_ITERATIONS = 10000

def print_usage():
    print("Usage: python tokenizer.py [--model MODEL_NAME] [--output OUTPUT_FILE] [--benchmark]")
    print("  MODEL_NAME: Name or path of the model to use (default: meta-llama/Llama-3.2-1B)")
    print("  OUTPUT_FILE: File to write benchmark results (default: py_tokenizer_results.csv)")
    print("  --benchmark: Run benchmark tests on multiple prompts")

def benchmark_tokenization(tokenizer, test_prompts, output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt', 'token_count', 'tokens', 'time_us'])
        
        # Pre-warm tokenizer to avoid first-time initialization costs
        _ = tokenizer.encode("warmup")
        
        for prompt in test_prompts:
            # Measure tokenization time with multiple iterations
            tokens = None
            
            # Use the highest precision timer available
            start_time = time.perf_counter()
            
            # Run multiple iterations for more accurate timing
            for _ in range(TIMING_ITERATIONS):
                tokens = tokenizer.encode(prompt)
                # Use the token output to prevent optimizations
                if len(tokens) == 0:
                    print("Error: Tokenization failed")
                    break
                
            end_time = time.perf_counter()
            
            # Calculate time in microseconds (with 3 decimal precision) and average per iteration
            total_duration_ns = (end_time - start_time) * 1_000_000_000
            avg_duration_us = total_duration_ns / TIMING_ITERATIONS / 1000.0
            
            # Write results
            writer.writerow([
                prompt,
                len(tokens),
                str(tokens),
                f"{avg_duration_us:.3f}"
            ])
            
            # Print to console
            print(f"Tokenized {len(prompt)} chars in {avg_duration_us:.3f} μs ({len(tokens)} tokens)")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Tokenize text using a transformer model")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B", help="Model name or path")
    parser.add_argument("--output", default="py_tokenizer_results.csv", help="Output file for benchmark results")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark on multiple prompts")
    parser.add_argument("--fallback", default="meta-llama/Llama-3.2-1B", help="Fallback model to use if local file doesn't exist")
    parser.add_argument("text", nargs="?", default="Thomas the Maker", help="Text to tokenize (ignored if --benchmark is used)")
    args = parser.parse_args()
    
    model_name = args.model
    sample_text = args.text
    
    # If model_name is a file path that doesn't exist or ends with .gguf, use the fallback model
    if os.path.exists(model_name) and model_name.endswith('.gguf'):
        print(f"Local GGUF file detected: {model_name}")
        print(f"Using fallback HuggingFace model: {args.fallback}")
        model_name = args.fallback
    
    print(f"Loading tokenizer for {model_name}...")
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # Run in benchmark mode or single text mode
    if args.benchmark:
        # Test prompts - same as in C++ version
        test_prompts = [
            "Hello world thomas",
            "Make America great again, and again, never yield.",
            "The quick brown fox jumps over the lazy dog.",
            "In a shocking turn of events, researchers discovered a new species of butterfly in the Amazon rainforest.",
            "import torch\nfrom transformers import AutoTokenizer\n\ntokenizer = AutoTokenizer.from_pretrained(\"meta-llama/Llama-3.2-1B\")",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)",
            "const calculatePi = (iterations) => {\n  let pi = 0;\n  for (let i = 0; i < iterations; i++) {\n    pi += Math.pow(-1, i) / (2 * i + 1);\n  }\n  return 4 * pi;\n};"
        ]
        
        print(f"Running tokenization benchmark with {TIMING_ITERATIONS} iterations per prompt...")
        benchmark_tokenization(tokenizer, test_prompts, args.output)
        print(f"Results written to {args.output}")
    else:
        # Original functionality for single text
        # Minimal model info
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        
        # Print vocabulary sample
        print("\n--- Vocabulary Sample ---")
        for i in range(20):  # Show first 20 tokens
            try:
                token_repr = tokenizer.convert_ids_to_tokens(i)
                print(f"ID {i:5}: {token_repr}")
            except:
                print(f"ID {i:5}: <error decoding token>")
        print("------------------------")
        
        print(f"\nTokenizing: \"{sample_text}\"")
        
        # Tokenize the text
        tokens = tokenizer.encode(sample_text, return_tensors="pt")
        
        # Show token IDs
        print(f"Token IDs: {tokens[0].tolist()}")
        print(f"Total tokens: {len(tokens[0])}")
        
        # Add a note about tokenization
        print("\nNote on tokenization:")
        print("- In most modern tokenizers, spaces are represented with special tokens or prefixes")
        print("- For example, many tokenizers use 'Ġ' (Unicode 0x0120) to mark tokens that follow a space")
        print("- Capitalization affects tokenization - 'Hello' and 'hello' are often different tokens")
        print("- Different models use different tokenization schemes")
        
        # Show token details with special representation
        print("\n--- Token Details ---")
        for token_id in tokens[0].tolist():
            # Get the token representation (showing special characters)
            token_repr = tokenizer.convert_ids_to_tokens(token_id)
            # Convert token ID back to text for display
            token_text = tokenizer.decode([token_id])
            print(f"ID {token_id:5}: '{token_text}' (repr: '{token_repr}')")
            
        # Verify by decoding back to text
        decoded_text = tokenizer.decode(tokens[0])
        print(f"\nDecoded: \"{decoded_text}\"")

if __name__ == "__main__":
    main()
