#include <iostream>
#include <vector>
#include <string>
#include "tokenizer.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: token_test <model_path> <text>" << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string text = argv[2];

    // Initialize tokenizer
    Tokenizer tokenizer(model_path);
    
    // Load the model
    if (!tokenizer.load_model()) {
        return 1;
    }

    // First tokenize without special tokens
    std::vector<int> tokens = tokenizer.tokenize(text);
    
    // Display token IDs
    std::cout << "Regular tokens: [";
    for (size_t i = 0; i < tokens.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << tokens[i];
    }
    std::cout << "]" << std::endl;
    
    // Then tokenize with BOS token
    std::vector<int> tokens_with_bos = tokenizer.tokenize_with_special_tokens(text, true, false);
    
    // Display token IDs
    std::cout << "Tokens with BOS: [";
    for (size_t i = 0; i < tokens_with_bos.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << tokens_with_bos[i];
    }
    std::cout << "]" << std::endl;
    
    return 0;
} 