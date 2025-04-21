#!/usr/bin/env python3
import subprocess
import numpy as np
import argparse
import sys
import os
from embedding import EmbeddingSystem

def run_cpp_embedding(cpp_binary, embedding_file, token_id):
    """Run the C++ embedding demo with the specified token ID"""
    try:
        cmd = [cpp_binary, embedding_file, str(token_id)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running C++ embedding demo: {e}")
        print(f"Error output: {e.stderr}")
        return None

def parse_cpp_output(output):
    """Parse the output from the C++ embedding program to extract embedding values"""
    if not output:
        return None
    
    values = []
    lines = output.strip().split('\n')
    for line in lines:
        if ': ' in line:
            try:
                # Extract the value part after the colon
                idx, value = line.split(': ', 1)
                values.append(float(value))
            except (ValueError, IndexError):
                pass  # Skip lines that don't match expected format
                
    return np.array(values) if values else None

def main():
    parser = argparse.ArgumentParser(description="Compare Python and C++ embedding implementations")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B", help="HuggingFace model name or path")
    parser.add_argument("--cpp-binary", default="./embedding_demo", help="Path to C++ embedding demo binary")
    parser.add_argument("--embedding-file", default="./embeddings.bin", help="Path to embedding weights file for C++")
    parser.add_argument("--token-id", type=int, default=42, help="Token ID to compare")
    parser.add_argument("--text", default=None, help="Text to tokenize and get embeddings for")
    args = parser.parse_args()
    
    # Check if C++ binary exists
    if not os.path.exists(args.cpp_binary):
        print(f"Warning: C++ binary {args.cpp_binary} does not exist. Python-only mode.")
        cpp_available = False
    else:
        cpp_available = True
    
    # Initialize Python embedding system
    print("Initializing Python embedding system...")
    py_embeddings = EmbeddingSystem(model_name=args.model)
    
    if args.text is not None:
        # Process text
        print(f"\nProcessing text: \"{args.text}\"")
        token_ids = py_embeddings.tokenizer.encode(args.text)
        print(f"Token IDs: {token_ids}")
        
        # Display token information
        for token_id in token_ids:
            token_text = py_embeddings.tokenizer.decode([token_id])
            print(f"\nToken: '{token_text}' (ID: {token_id})")
            
            # Get Python embedding
            py_embedding = py_embeddings.get_token_embedding(token_id)
            print(f"Python embedding (first 5 values): {py_embedding[:5]}")
            
            # Get C++ embedding if available
            if cpp_available:
                print(f"Running C++ embedding for token ID {token_id}...")
                cpp_output = run_cpp_embedding(args.cpp_binary, args.embedding_file, token_id)
                cpp_embedding = parse_cpp_output(cpp_output)
                
                if cpp_embedding is not None:
                    print(f"C++ embedding (first 5 values): {cpp_embedding[:5]}")
                    
                    # Compare
                    if len(cpp_embedding) >= 5:
                        diffs = np.abs(py_embedding[:len(cpp_embedding)] - cpp_embedding)
                        max_diff = diffs.max()
                        avg_diff = diffs.mean()
                        print(f"Max absolute difference: {max_diff}")
                        print(f"Average absolute difference: {avg_diff}")
                    else:
                        print("Not enough values to compare")
                else:
                    print("Failed to parse C++ output")
    else:
        # Process a single token ID
        token_id = args.token_id
        print(f"\nComparing embeddings for token ID: {token_id}")
        
        # Get Python embedding
        try:
            py_embedding = py_embeddings.get_token_embedding(token_id)
            token_text = py_embeddings.tokenizer.decode([token_id])
            print(f"Token text: '{token_text}'")
            print(f"Python embedding (first 10 values):")
            for i in range(min(10, len(py_embedding))):
                print(f"{i}: {py_embedding[i]}")
                
            # Get C++ embedding if available
            if cpp_available:
                print(f"\nRunning C++ embedding...")
                cpp_output = run_cpp_embedding(args.cpp_binary, args.embedding_file, token_id)
                cpp_embedding = parse_cpp_output(cpp_output)
                
                if cpp_embedding is not None:
                    print(f"C++ embedding (parsed values):")
                    for i in range(min(10, len(cpp_embedding))):
                        print(f"{i}: {cpp_embedding[i]}")
                        
                    # Calculate differences
                    min_len = min(len(py_embedding), len(cpp_embedding))
                    diffs = np.abs(py_embedding[:min_len] - cpp_embedding[:min_len])
                    max_diff = diffs.max()
                    avg_diff = diffs.mean()
                    
                    print(f"\nComparison:")
                    print(f"Max absolute difference: {max_diff}")
                    print(f"Average absolute difference: {avg_diff}")
                    print(f"Within tolerance (< 1e-5): {max_diff < 1e-5}")
                else:
                    print("Failed to parse C++ output")
        except ValueError as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main() 