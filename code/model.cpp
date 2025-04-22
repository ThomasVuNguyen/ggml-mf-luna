#include "tokenizer.h"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "Loading tokenizer..." << std::endl;
    Tokenizer tokenizer("./gguf/1b-q8_0.gguf");
    
    std::cout << "Tokenizing text..." << std::endl;
    std::string text = "Thomas the Maker";
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

    // Get embedding matrix size and vocabulary size
    std::cout << "\nModel information:" << std::endl;
    
    // Get vocabulary size
    size_t vocab_size = tokenizer.id_to_token.size();
    std::cout << "Vocabulary size: " << vocab_size << " tokens" << std::endl;
    
    // Get embedding matrix size by checking the first token's embedding dimension
    if (!tokens.empty()) {
        std::vector<float> sample_embedding = tokenizer.get_embeddings(tokens[0]);
        size_t embedding_dim = sample_embedding.size();
        std::cout << "Embedding matrix size: " << vocab_size << " x " << embedding_dim;
        std::cout << " (" << (vocab_size * embedding_dim * sizeof(float)) / (1024 * 1024) << " MB)" << std::endl;
    } else {
        std::cout << "Cannot determine embedding size: no tokens available" << std::endl;
    }

    
    return 0;

}