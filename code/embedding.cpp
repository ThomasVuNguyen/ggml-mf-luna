#include "tokenizer.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

int main() {
    std::cout << "Loading model..." << std::endl;
    Tokenizer model("./gguf/1b-q8_0.gguf");
    
    // Get embedding matrix size directly from the tensor
    std::cout << "\nModel embedding information:" << std::endl;
    
    // Find the embedding tensor
    int64_t embedding_tensor_idx = gguf_find_tensor(model.get_gguf_context(), "token_embd.weight");
    if (embedding_tensor_idx < 0) {
        std::cerr << "Embedding tensor not found in model" << std::endl;
        return 1;
    }

    // Get tensor and its dimensions
    const char* tensor_name = gguf_get_tensor_name(model.get_gguf_context(), embedding_tensor_idx);
    struct ggml_tensor* embd_tensor = ggml_get_tensor(model.get_ggml_context(), tensor_name);
    if (!embd_tensor) {
        std::cerr << "Failed to get embedding tensor" << std::endl;
        return 1;
    }

    // Get dimensions directly from the tensor
    const int64_t vocab_size = embd_tensor->ne[0]; // vocabulary size
    const int64_t embedding_dim = embd_tensor->ne[1]; // embedding dimension
    
    std::cout << "Embedding matrix dimensions: " << vocab_size << " x " << embedding_dim << std::endl;
    std::cout << "Matrix size: " << (vocab_size * embedding_dim * sizeof(float)) / (1024 * 1024) << " MB" << std::endl;
    
    // Get a sample embedding to verify
    int sample_token_id = 1;
    std::vector<float> sample_embedding = model.get_embeddings(sample_token_id);
    
    if (!sample_embedding.empty()) {
        std::cout << "\nSample embedding vector (first 10 dimensions):" << std::endl;
        for (size_t i = 0; i < std::min(size_t(10), sample_embedding.size()); i++) {
            std::cout << std::setw(2) << i << ": " << std::setprecision(6) << sample_embedding[i] << std::endl;
        }
    } else {
        std::cout << "Cannot get sample embedding" << std::endl;
    }
    
    return 0;
}