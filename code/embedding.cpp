#include "tokenizer.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

int main() {
    std::cout << "Loading model..." << std::endl;
    Tokenizer model("./gguf/1b-q8_0.gguf");
    
    // Get embedding matrix size and vocabulary size
    std::cout << "\nModel embedding information:" << std::endl;
    
    // Get vocabulary size
    size_t vocab_size = model.id_to_token.size();
    std::cout << "Vocabulary size: " << vocab_size << " tokens" << std::endl;
    
    // Get embedding matrix size by checking a sample token's embedding dimension
    int sample_token_id = 1; // Using token ID 1 as a sample
    std::vector<float> sample_embedding = model.get_embeddings(sample_token_id);
    
    if (!sample_embedding.empty()) {
        size_t embedding_dim = sample_embedding.size();
        std::cout << "Embedding dimension: " << embedding_dim << std::endl;
        std::cout << "Embedding matrix size: " << vocab_size << " x " << embedding_dim;
        std::cout << " (" << (vocab_size * embedding_dim * sizeof(float)) / (1024 * 1024) << " MB)" << std::endl;
        
        // Print a sample of the embedding vector
        std::cout << "\nSample embedding vector (first 10 dimensions):" << std::endl;
        for (size_t i = 0; i < std::min(size_t(10), embedding_dim); i++) {
            std::cout << std::setw(2) << i << ": " << std::setprecision(6) << sample_embedding[i] << std::endl;
        }
    } else {
        std::cout << "Cannot determine embedding size: failed to get embeddings" << std::endl;
    }
    
    return 0;
}