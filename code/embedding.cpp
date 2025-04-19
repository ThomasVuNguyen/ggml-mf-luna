/**
 * embedding.cpp - Token ID to embedding conversion for language models
 * 
 * This file provides functions to convert token IDs to embeddings
 * and related data structures for language model implementations.
 */

#include "embedding.h"

// Load embedding weights from a file
void load_embedding_weights(EmbeddingWeights* weights, const char* path, EmbeddingConfig* config) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "Error: Could not open embedding file: %s\n", path);
        exit(EXIT_FAILURE);
    }
    
    // Allocate memory for the embedding table
    size_t embedding_table_size = config->vocab_size * config->dim * sizeof(float);
    weights->token_embedding_table = (float*)malloc(embedding_table_size);
    if (!weights->token_embedding_table) {
        fprintf(stderr, "Error: Failed to allocate memory for embedding table\n");
        exit(EXIT_FAILURE);
    }
    
    // Read the embedding table
    if (fread(weights->token_embedding_table, embedding_table_size, 1, file) != 1) {
        fprintf(stderr, "Error: Failed to read embedding table\n");
        exit(EXIT_FAILURE);
    }
    
    fclose(file);
}

// Free the memory used by embedding weights
void free_embedding_weights(EmbeddingWeights* weights) {
    free(weights->token_embedding_table);
    weights->token_embedding_table = NULL;
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
    // Load embedding weights
    load_embedding_weights(weights, weight_path, config);
    printf("Loaded embedding table with vocabulary size %d and dimension %d\n", 
           config->vocab_size, config->dim);
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
    
    // Here you would typically proceed with further processing,
    // such as passing through transformer layers
    
    return embedding;  // Caller is responsible for freeing this memory
}

#ifdef EMBEDDING_DEMO
// Demo main function
int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <embedding_file>\n", argv[0]);
        return 1;
    }
    
    // Initialize configuration
    EmbeddingConfig config = {
        .dim = 768,         // Example dimension
        .vocab_size = 32000 // Example vocabulary size
    };
    
    // Initialize weights
    EmbeddingWeights weights;
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
