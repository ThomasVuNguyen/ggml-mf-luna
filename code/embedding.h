/**
 * embedding.h - Header for token ID to embedding conversion
 */

#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Configuration structure for the model
typedef struct {
    int dim;         // embedding dimension
    int vocab_size;  // size of vocabulary
} EmbeddingConfig;

// Structure to hold the embedding table
typedef struct {
    float* token_embedding_table;  // (vocab_size, dim)
} EmbeddingWeights;

// Load embedding weights from a file
void load_embedding_weights(EmbeddingWeights* weights, const char* path, EmbeddingConfig* config);

// Free the memory used by embedding weights
void free_embedding_weights(EmbeddingWeights* weights);

// Get embedding vector for a token ID
void get_token_embedding(float* embedding, const EmbeddingWeights* weights, int token_id, int dim);

// Apply layer normalization to embeddings (RMSNorm variant)
void rmsnorm(float* output, const float* input, const float* weight, int size);

// Initialize embedding system
void init_embedding_system(EmbeddingConfig* config, EmbeddingWeights* weights, const char* weight_path);

// Process a token into an embedding
float* process_token(int token_id, const EmbeddingWeights* weights, const EmbeddingConfig* config);

#endif /* EMBEDDING_H */ 