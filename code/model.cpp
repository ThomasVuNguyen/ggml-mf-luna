#include "tokenizer.h"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "Loading tokenizer..." << std::endl;
    Tokenizer tokenizer("./gguf/1b-q8_0.gguf");
    
    // Print vocabulary sample to inspect tokens
    // tokenizer.print_vocabulary_sample(30);
    
    std::cout << "Tokenizing text..." << std::endl;
    std::string text = "Make America great again, and again, never yield.";
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
    }
    
    return 0;
}