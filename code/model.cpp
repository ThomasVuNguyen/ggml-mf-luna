#include "tokenizer.h"
#include "embedding.h"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "Loading tokenizer..." << std::endl;
    Tokenizer tokenizer("./gguf/1b-q8_0.gguf");
    
    // Tokenize text
    std::string text = "Make America great again, and again, hi sfasd asdwe sfd ad";
    std::vector<int> tokens = tokenizer.tokenize(text);
    
    std::cout << "Tokens for '" << text << "':" << std::endl;
    for (size_t i = 0; i < tokens.size(); i++) {
        std::cout << "Token " << i << ": ID " << tokens[i] << std::endl;
    }
    
    // Set up embedding configuration
    EmbeddingConfig config;
    config.dim = 384;  // Example dimension
    config.vocab_size = tokenizer.id_to_token.size();
    
    // Initialize embedding weights with mock data (normally loaded from a file)
    EmbeddingWeights weights;
    weights.token_embedding_table = (float*)malloc(config.vocab_size * config.dim * sizeof(float));
    
    // Fill with dummy data (in real usage, this would be loaded from a model file)
    for (int i = 0; i < config.vocab_size * config.dim; i++) {
        weights.token_embedding_table[i] = (float)rand() / RAND_MAX;
    }
    
    // Process each token
    for (int token : tokens) {
        // Get the embedding for this token
        float* embedding = process_token(token, &weights, &config);
        
        // Display first few values of the embedding
        std::cout << "Embedding for token " << token << " (first 5 values): ";
        for (int i = 0; i < 5; i++) {
            std::cout << embedding[i] << " ";
        }
        std::cout << std::endl;
        
        free(embedding);
    }
    
    // Clean up
    free_embedding_weights(&weights);
    
    return 0;
}