#include "tokenizer.h"
#include "config.h"
#include "rms_norm.h"
#include "ggml/include/gguf.h"
#include "ggml/include/ggml.h"
#include "ggml/include/ggml-cpu.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

int main() {
    ggml_time_init();
    // Load model and initialize tokenizer
    std::cout << "Loading model..." << std::endl;
    Tokenizer tokenizer("./gguf/1b-q8_0.gguf");
    
    // Get model epsilon value
    float model_epsilon = get_rms_epsilon(tokenizer);
    
    // Get RMS norm weights for output layer
    std::string norm_layer = "output_norm.weight";
    std::vector<float> norm_weights = get_layer_rms_weights(tokenizer, norm_layer);
    
    // Tokenize input
    std::string text = "Thomas";
    std::cout << "\n======= RMS Normalization Analysis =======" << std::endl;
    std::cout << "Prompt: \"" << text << "\"" << std::endl;
    
    std::vector<int> tokens = tokenizer.tokenize(text);
    
    // Create headers for the data table
    std::cout << "\n" << std::left 
              << std::setw(10) << "Token ID" 
              << std::setw(20) << "Token" 
              << std::setw(15) << "Epsilon" 
              << std::setw(20) << "Layer" << std::endl;
    
    std::cout << std::string(80, '-') << std::endl;
    
    for (size_t i = 0; i < tokens.size(); i++) {
        int token_id = tokens[i];
        
        // Get token text
        std::string token_text = "(unknown)";
        auto it = tokenizer.id_to_token.find(token_id);
        if (it != tokenizer.id_to_token.end()) {
            token_text = it->second;
        }
        
        // Print token info
        std::cout << std::left 
                  << std::setw(10) << token_id 
                  << std::setw(20) << token_text
                  << std::setw(15) << model_epsilon
                  << std::setw(20) << norm_layer << std::endl;
        
        // Get embeddings
        std::vector<float> embeddings = tokenizer.get_embeddings(token_id);
        
        // Apply RMS normalization with model weights
        std::vector<float> normalized = compute_rms_norm(
            tokenizer, embeddings, norm_layer);
        
        // Print data table of values
        std::cout << "\nData (first 5 values):" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        print_vector_head(embeddings, "Embeddings:", 5);
        print_vector_head(norm_weights, "RMS Weights:", 5);
        print_vector_head(normalized, "Normalized:", 5);
        std::cout << std::string(80, '-') << std::endl;
        
        // Try different layers for comparison
        std::cout << "\nComparison across layers:" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        std::vector<std::string> layers = {
            "blk.0.attn_norm.weight",
            "blk.15.attn_norm.weight",
            "output_norm.weight"
        };
        
        for (const auto& layer : layers) {
            std::vector<float> layer_norm = compute_rms_norm(
                tokenizer, embeddings, layer);
            
            std::cout << std::left << std::setw(20) << layer;
            for (int j = 0; j < std::min(5, (int)layer_norm.size()); j++) {
                std::cout << std::fixed << std::setprecision(6) << std::setw(15) << layer_norm[j];
            }
            std::cout << std::endl;
        }
        
        std::cout << "\n" << std::string(80, '=') << std::endl;
    }
    
    return 0;
}