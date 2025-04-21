#include "tokenizer.h"
#include <iostream>
#include <vector>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: tokenizer_test <model_path> <text>" << std::endl;
        std::cout << "  model_path: Path to the GGUF model file" << std::endl;
        std::cout << "  text: Text to tokenize" << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string text = argv[2];

    // Initialize tokenizer
    Tokenizer tokenizer(model_path);
    
    // Load the tokenizer model
    if (!tokenizer.load_model()) {
        std::cerr << "Failed to load tokenizer model" << std::endl;
        return 1;
    }
    
    // Print model info
    tokenizer.print_config();
    
    // Print vocabulary sample
    tokenizer.print_vocabulary_sample(10);
    
    // Tokenize the input text without special tokens
    std::vector<int> tokens = tokenizer.tokenize(text);
    
    // Also tokenize with special tokens (BOS added)
    std::vector<int> tokens_with_special = tokenizer.tokenize_with_special_tokens(text, true, false);
    
    std::cout << "\nTokens with BOS token: ";
    for (int token : tokens_with_special) {
        if (token == tokenizer.bos_token_id) {
            std::cout << token << "(BOS) ";
        } else {
            std::cout << token << " ";
        }
    }
    std::cout << "\nTotal tokens (with special): " << tokens_with_special.size() << std::endl;
    
    return 0;
} 