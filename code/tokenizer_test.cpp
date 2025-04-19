#include "tokenizer.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

// Number of iterations to run for more accurate timing
const int TIMING_ITERATIONS = 10000;

// Function to measure tokenization performance
void benchmark_tokenization(Tokenizer& tokenizer, const std::vector<std::string>& test_prompts, const std::string& output_file) {
    std::ofstream outfile(output_file);
    if (!outfile) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return;
    }
    
    outfile << "prompt,token_count,tokens,time_us" << std::endl;
    
    // Pre-warm tokenizer to avoid first-time initialization costs
    volatile auto warmup = tokenizer.tokenize("warmup");
    
    for (const auto& prompt : test_prompts) {
        // Measure tokenization time with multiple iterations for accuracy
        std::vector<int> tokens;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Run iterations and store result to prevent optimization
        for (int i = 0; i < TIMING_ITERATIONS; i++) {
            tokens = tokenizer.tokenize(prompt);
            // Use the result to prevent optimization
            if (tokens.empty()) {
                std::cerr << "Error: Tokenization failed" << std::endl;
                break;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        
        // Calculate time in nanoseconds and average per iteration
        auto total_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        auto avg_duration_us = static_cast<double>(total_duration) / TIMING_ITERATIONS / 1000.0;
        
        // Write token IDs to the output file
        outfile << "\"" << prompt << "\",";
        outfile << tokens.size() << ",\"[";
        
        for (size_t i = 0; i < tokens.size(); i++) {
            outfile << tokens[i];
            if (i < tokens.size() - 1) {
                outfile << ", ";
            }
        }
        
        outfile << "]\",";
        outfile << std::fixed << std::setprecision(3) << avg_duration_us << std::endl;
        
        // Print to console as well
        std::cout << "Tokenized " << prompt.length() << " chars in " << std::fixed << std::setprecision(3) 
                  << avg_duration_us << " μs (" << tokens.size() << " tokens)" << std::endl;
    }
    
    outfile.close();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path> [output_file]" << std::endl;
        return 1;
    }
    
    const std::string model_path = argv[1];
    const std::string output_file = (argc >= 3) ? argv[2] : "cpp_tokenizer_results.csv";
    
    // Test prompts
    std::vector<std::string> test_prompts = {
        "Hello world thomas",
        "Make America great again, and again, never yield.",
        "The quick brown fox jumps over the lazy dog.",
        "In a shocking turn of events, researchers discovered a new species of butterfly in the Amazon rainforest.",
        "import torch\nfrom transformers import AutoTokenizer\n\ntokenizer = AutoTokenizer.from_pretrained(\"meta-llama/Llama-3.2-1B\")",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)",
        "const calculatePi = (iterations) => {\n  let pi = 0;\n  for (let i = 0; i < iterations; i++) {\n    pi += Math.pow(-1, i) / (2 * i + 1);\n  }\n  return 4 * pi;\n};"
    };
    
    std::cout << "Loading tokenizer from " << model_path << "..." << std::endl;
    Tokenizer tokenizer(model_path);
    
    std::cout << "Running tokenization benchmark with " << TIMING_ITERATIONS << " iterations per prompt..." << std::endl;
    benchmark_tokenization(tokenizer, test_prompts, output_file);
    
    std::cout << "Results written to " << output_file << std::endl;
    
    return 0;
} 