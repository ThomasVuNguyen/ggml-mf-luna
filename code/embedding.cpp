/**
 * embedding.cpp - Token ID to embedding conversion for language models
 * 
 * This file provides functions to convert token IDs to embeddings
 * and related data structures for language model implementations.
 */

#include "embedding.h"
#include <iostream>
#include <cstring>

// Use GGML directly
#include "ggml/include/ggml.h"
#include "ggml/include/gguf.h"

// Load embedding weights from a file
void load_embedding_weights(EmbeddingWeights* weights, const char* path, EmbeddingConfig* config) {
    // Initialize GGUF context
    struct ggml_context* ggml_ctx = NULL;
    struct gguf_init_params params = {
        .no_alloc = false,
        .ctx = &ggml_ctx
    };
    
    // Load model from file
    struct gguf_context* ctx = gguf_init_from_file(path, params);
    if (!ctx) {
        fprintf(stderr, "Error: Failed to load model from %s\n", path);
        exit(EXIT_FAILURE);
    }
    
    // Find embedding tensor
    int64_t embd_idx = gguf_find_tensor(ctx, "token_embd.weight");
    if (embd_idx < 0) {
        fprintf(stderr, "Error: Could not find token embeddings in model\n");
        gguf_free(ctx);
        exit(EXIT_FAILURE);
    }
    
    // Get tensor properties
    struct ggml_tensor* embd_tensor = ggml_get_tensor(ggml_ctx, gguf_get_tensor_name(ctx, embd_idx));
    if (!embd_tensor) {
        fprintf(stderr, "Error: Failed to get embedding tensor\n");
        gguf_free(ctx);
        exit(EXIT_FAILURE);
    }
    
    // Get dimensions
    config->vocab_size = embd_tensor->ne[0];
    config->dim = embd_tensor->ne[1];
    
    printf("Model loaded: embedding dimension = %d, vocabulary size = %d\n", 
           config->dim, config->vocab_size);
    
    // Allocate memory for the embedding table
    size_t embedding_table_size = config->vocab_size * config->dim * sizeof(float);
    weights->token_embedding_table = (float*)malloc(embedding_table_size);
    if (!weights->token_embedding_table) {
        fprintf(stderr, "Error: Failed to allocate memory for embedding table\n");
        gguf_free(ctx);
        exit(EXIT_FAILURE);
    }
    
    // Copy embeddings
    // Get tensor type and prepare for conversion
    enum ggml_type tensor_type = gguf_get_tensor_type(ctx, embd_idx);
    const size_t row_size = ggml_row_size(tensor_type, config->dim);
    float* row_f32 = (float*)malloc(config->dim * sizeof(float));
    
    // Convert and copy each token embedding
    for (int token_id = 0; token_id < config->vocab_size; token_id++) {
        float* token_embedding = weights->token_embedding_table + token_id * config->dim;
        
        // Get pointer to the token's embedding in the tensor
        void* src = (char*)embd_tensor->data + token_id * row_size;
        
        // Convert to float if needed
        const struct ggml_type_traits* qtype = ggml_get_type_traits(tensor_type);
        if (qtype->to_float) {
            qtype->to_float(src, row_f32, config->dim);
            memcpy(token_embedding, row_f32, config->dim * sizeof(float));
        } else {
            // Already in float format
            memcpy(token_embedding, src, config->dim * sizeof(float));
        }
    }
    
    // Store model in weights
    weights->ctx = ctx;
    weights->ggml_ctx = ggml_ctx;
    weights->model = NULL; // Not using llama_model now
    
    // Clean up conversion buffer
    free(row_f32);
}

// Free the memory used by embedding weights
void free_embedding_weights(EmbeddingWeights* weights) {
    // Free the embedding table
    free(weights->token_embedding_table);
    weights->token_embedding_table = NULL;
    
    // Free GGUF context
    if (weights->ctx) {
        gguf_free(weights->ctx);
        weights->ctx = NULL;
        weights->ggml_ctx = NULL; // Freed by gguf_free
    }
}

// Get embedding vector for a token ID
void get_token_embedding(float* embedding, const EmbeddingWeights* weights, int token_id, int dim) {
    // Check if token_id is valid
    if (token_id < 0) {
        fprintf(stderr, "Error: Invalid token ID: %d\n", token_id);
        return;
    }
    
    // Get pointer to the embedding vector for this token
    float* content_row = weights->token_embedding_table + token_id * dim;
    
    // Copy the embedding vector to the output buffer
    memcpy(embedding, content_row, dim * sizeof(float));
}

// Apply layer normalization to embeddings (RMSNorm variant)
void rmsnorm(float* output, const float* input, const float* weight, int size) {
    // Calculate sum of squares
    float ss = 0.0f;
    for (int i = 0; i < size; i++) {
        ss += input[i] * input[i];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    
    // Normalize and scale
    for (int i = 0; i < size; i++) {
        output[i] = weight[i] * (ss * input[i]);
    }
}

// Initialize embedding system
void init_embedding_system(EmbeddingConfig* config, EmbeddingWeights* weights, const char* weight_path) {
    // Initialize with zeros
    memset(weights, 0, sizeof(EmbeddingWeights));
    
    // Load embedding weights
    load_embedding_weights(weights, weight_path, config);
}

// Normalize the embedding vector using L2 normalization (like Python implementation)
void normalize_embedding(float* embedding, int dim) {
    // Calculate the L2 norm
    float norm = 0.0f;
    for (int i = 0; i < dim; i++) {
        norm += embedding[i] * embedding[i];
    }
    norm = sqrtf(norm);
    
    // Avoid division by zero
    if (norm < 1e-5f) {
        norm = 1e-5f;
    }
    
    // Normalize
    for (int i = 0; i < dim; i++) {
        embedding[i] /= norm;
    }
}

// Example usage function
float* process_token(int token_id, const EmbeddingWeights* weights, const EmbeddingConfig* config) {
    // Allocate memory for the embedding
    float* embedding = (float*)malloc(config->dim * sizeof(float));
    if (!embedding) {
        fprintf(stderr, "Error: Failed to allocate memory for embedding\n");
        return NULL;
    }
    
    // Get embedding for the token
    get_token_embedding(embedding, weights, token_id, config->dim);
    
    // Normalize the embedding (to match Python implementation)
    normalize_embedding(embedding, config->dim);
    
    return embedding;  // Caller is responsible for freeing this memory
}

#ifdef EMBEDDING_DEMO
// Demo main function
int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <embedding_file>\n", argv[0]);
        return 1;
    }
    
    // Initialize configuration (will be updated with actual values during loading)
    EmbeddingConfig config = {0};
    
    // Initialize weights and load the model
    EmbeddingWeights weights = {0};
    init_embedding_system(&config, &weights, argv[1]);
    
    // Process a token (example: token ID 42)
    int token_id = 42;
    float* embedding = process_token(token_id, &weights, &config);
    
    // Print first few values of the embedding
    printf("Embedding for token %d (first 10 values):\n", token_id);
    for (int i = 0; i < 10 && i < config.dim; i++) {
        printf("%d: %f\n", i, embedding[i]);
    }
    
    // Clean up
    free(embedding);
    free_embedding_weights(&weights);
    
    return 0;
}
#endif
