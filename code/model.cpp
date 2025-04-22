#include "tokenizer.h"
#include "config.h"
#include "ggml/include/gguf.h"
#include "ggml/include/ggml.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

int main() {
    std::cout << "Loading tokenizer..." << std::endl;
    Tokenizer tokenizer("./gguf/1b-q8_0.gguf");
    
    // ===== Tokenization Example =====
    std::cout << "Tokenizing text..." << std::endl;
    std::string text = "!";
    std::vector<int> tokens = tokenizer.tokenize(text);
    
    std::cout << "Tokens for '" << text << "':" << std::endl;
    for (size_t i = 0; i < tokens.size(); i++) {
        std::cout << "Token " << i << ": ID " << std::setw(5) << tokens[i];
        
        // Try to get the actual token text
        auto it = tokenizer.id_to_token.find(tokens[i]);
        if (it != tokenizer.id_to_token.end()) {
            std::cout << " = '" << it->second << "'";
        } else {
            std::cout << " (unknown)";
        }
        std::cout << std::endl;
        
        // Print embeddings for this token
        std::cout << "\nExtract embeddings for token ID " << tokens[i] << ":" << std::endl;
        print_token_embedding(tokenizer, tokens[i]);
        std::cout << "----------------------------------------------" << std::endl;
    }

    return 0;
}