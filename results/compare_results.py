#!/usr/bin/env python3
import csv
import sys
import ast
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_results(cpp_file, py_file):
    # Load results from CSV files
    cpp_data = pd.read_csv(cpp_file)
    py_data = pd.read_csv(py_file)
    
    # Process the token lists - convert string representations to actual lists
    cpp_data['tokens'] = cpp_data['tokens'].apply(lambda x: ast.literal_eval(x))
    py_data['tokens'] = py_data['tokens'].apply(lambda x: ast.literal_eval(x))
    
    return cpp_data, py_data

def compare_tokenization(cpp_data, py_data):
    print("=== Tokenization Comparison ===")
    print(f"{'Prompt':<50} | {'CPP Tokens':<10} | {'PY Tokens':<10} | {'Match':<5} | {'CPP Time (μs)':<14} | {'PY Time (μs)':<14}")
    print("-" * 112)
    
    # Make sure the dataframes are aligned by prompt
    merged = pd.merge(cpp_data, py_data, on='prompt', suffixes=('_cpp', '_py'))
    
    matches = 0
    for idx, row in merged.iterrows():
        cpp_tokens = row['tokens_cpp']
        py_tokens = row['tokens_py']
        
        prompt = row['prompt']
        if len(prompt) > 47:
            prompt = prompt[:44] + "..."
        
        # Check if tokenization matches
        tokens_match = cpp_tokens == py_tokens
        if tokens_match:
            matches += 1
        
        # Get timing data - assumes microseconds in both cases
        cpp_time = row.get('time_us_cpp', 0)
        py_time = row.get('time_us_py', 0)
        
        print(f"{prompt:<50} | {row['token_count_cpp']:<10} | {row['token_count_py']:<10} | {'✓' if tokens_match else '✗':<5} | {cpp_time:<14} | {py_time:<14}")
    
    print("-" * 112)
    print(f"Match rate: {matches}/{len(merged)} ({matches/len(merged)*100:.2f}%)")
    
    # Calculate average speed difference
    avg_cpp_time = merged['time_us_cpp'].mean()
    avg_py_time = merged['time_us_py'].mean()
    speedup = avg_py_time / avg_cpp_time if avg_cpp_time > 0 else float('inf')
    
    print(f"\nAverage C++ time: {avg_cpp_time:.2f} μs")
    print(f"Average Python time: {avg_py_time:.2f} μs")
    print(f"C++ is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than Python")
    
    return merged

def visualize_comparison(merged_data):
    # Create a figure and a set of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Data for plots
    prompts = [f"Prompt {i+1}" for i in range(len(merged_data))]
    cpp_times = merged_data['time_us_cpp'].values
    py_times = merged_data['time_us_py'].values
    
    # Bar chart for execution times
    x = np.arange(len(prompts))
    width = 0.35
    
    ax1.bar(x - width/2, cpp_times, width, label='C++')
    ax1.bar(x + width/2, py_times, width, label='Python')
    
    ax1.set_ylabel('Time (μs)')
    ax1.set_title('Tokenization Time by Implementation')
    ax1.set_xticks(x)
    ax1.set_xticklabels(prompts)
    ax1.legend()
    
    # Bar chart for token counts
    cpp_counts = merged_data['token_count_cpp'].values
    py_counts = merged_data['token_count_py'].values
    
    ax2.bar(x - width/2, cpp_counts, width, label='C++')
    ax2.bar(x + width/2, py_counts, width, label='Python')
    
    ax2.set_ylabel('Token Count')
    ax2.set_title('Token Count by Implementation')
    ax2.set_xticks(x)
    ax2.set_xticklabels(prompts)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('tokenization_comparison.png')
    print("\nVisualization saved to 'tokenization_comparison.png'")

def analyze_token_differences(merged_data):
    print("\n=== Token Difference Analysis ===")
    
    for idx, row in merged_data.iterrows():
        cpp_tokens = row['tokens_cpp']
        py_tokens = row['tokens_py']
        
        if cpp_tokens != py_tokens:
            prompt = row['prompt']
            if len(prompt) > 47:
                prompt = prompt[:44] + "..."
                
            print(f"\nDifference in prompt: {prompt}")
            print(f"CPP tokens ({len(cpp_tokens)}): {cpp_tokens}")
            print(f"PY tokens ({len(py_tokens)}): {py_tokens}")
            
            # Find the first difference
            min_len = min(len(cpp_tokens), len(py_tokens))
            for i in range(min_len):
                if cpp_tokens[i] != py_tokens[i]:
                    print(f"First difference at position {i}: CPP={cpp_tokens[i]} vs PY={py_tokens[i]}")
                    break
            else:
                if len(cpp_tokens) != len(py_tokens):
                    print(f"Length mismatch: CPP has {len(cpp_tokens)} tokens, PY has {len(py_tokens)} tokens")

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_results.py <cpp_results.csv> <py_results.csv>")
        return
        
    cpp_file = sys.argv[1]
    py_file = sys.argv[2]
    
    print(f"Comparing results from {cpp_file} and {py_file}...")
    cpp_data, py_data = load_results(cpp_file, py_file)
    
    merged = compare_tokenization(cpp_data, py_data)
    analyze_token_differences(merged)
    
    try:
        visualize_comparison(merged)
    except Exception as e:
        print(f"Could not generate visualization: {e}")

if __name__ == "__main__":
    main() 