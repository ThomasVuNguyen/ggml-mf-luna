#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <cstring>
#include <map>
#include <cmath>
#include <algorithm>
#include <iomanip>

#include "ggml/include/ggml.h"
#include "ggml/include/gguf.h"

// Simple tokenizer that loads a GGUF model and outputs embeddings for a prompt
class Tokenizer {
private:
    struct gguf_context* ctx;
    struct ggml_context* ggml_ctx;
    std::string model_path;
    std::map<int, std::string> id_to_token;
    std::map<std::string, int> token_to_id;

public:
    Tokenizer(const std::string& model_path) : model_path(model_path), ctx(nullptr), ggml_ctx(nullptr) {}

    ~Tokenizer() {
        if (ctx) {
            gguf_free(ctx);
        }
    }

    bool load_model() {
        // Initialize GGUF context from file
        struct gguf_init_params params = {
            .no_alloc = false,
            .ctx = &ggml_ctx
        };

        ctx = gguf_init_from_file(model_path.c_str(), params);
        if (!ctx) {
            std::cerr << "Failed to load model: " << model_path << std::endl;
            return false;
        }

        // Load vocabulary
        if (!load_vocabulary()) {
            std::cerr << "Failed to load vocabulary" << std::endl;
            return false;
        }
        
        return true;
    }

    bool load_vocabulary() {
        // Find tokenizer vocab
        int64_t vocab_key = gguf_find_key(ctx, "tokenizer.ggml.tokens");
        if (vocab_key < 0) {
            // Try alternative key
            vocab_key = gguf_find_key(ctx, "tokenizer.model.tokens");
        }
        
        if (vocab_key < 0) {
            std::cerr << "Vocabulary not found in model" << std::endl;
            return false;
        }

        // Check type
        if (gguf_get_kv_type(ctx, vocab_key) != GGUF_TYPE_ARRAY) {
            std::cerr << "Vocabulary is not an array" << std::endl;
            return false;
        }

        if (gguf_get_arr_type(ctx, vocab_key) != GGUF_TYPE_STRING) {
            std::cerr << "Vocabulary array does not contain strings" << std::endl;
            return false;
        }

        // Get vocabulary size
        const size_t vocab_size = gguf_get_arr_n(ctx, vocab_key);

        // Load token to id mapping
        for (size_t i = 0; i < vocab_size; i++) {
            const char* token = gguf_get_arr_str(ctx, vocab_key, i);
            id_to_token[i] = token;
            token_to_id[token] = i;
        }

        return true;
    }

    void print_config() {
        // Simplified - removed verbose configuration output
        std::cout << "Model loaded: " << model_path << std::endl;
        std::cout << "Vocabulary size: " << id_to_token.size() << std::endl;
    }

    void print_vocabulary_sample(int n = 20) {
        std::cout << "--- Vocabulary Sample ---" << std::endl;
        
        int count = 0;
        for (const auto& entry : id_to_token) {
            if (count >= n) break;
            
            // Simplified - just show ID and token
            std::cout << "ID " << std::setw(5) << entry.first << ": ";
            std::cout << entry.second << std::endl;
            
            count++;
        }
        
        std::cout << "------------------------" << std::endl;
    }

    std::vector<float> get_embeddings(int token_id) {
        std::vector<float> embeddings;
        
        if (!ctx) {
            std::cerr << "Model not loaded" << std::endl;
            return embeddings;
        }

        // Find the embedding tensor
        int64_t embedding_tensor_idx = gguf_find_tensor(ctx, "token_embd.weight");
        if (embedding_tensor_idx < 0) {
            std::cerr << "Embedding tensor not found in model" << std::endl;
            return embeddings;
        }

        // Get information about the embedding tensor
        const char* tensor_name = gguf_get_tensor_name(ctx, embedding_tensor_idx);
        enum ggml_type tensor_type = gguf_get_tensor_type(ctx, embedding_tensor_idx);
        
        // Create a ggml tensor to properly handle the embedding data
        struct ggml_tensor* embd_tensor = ggml_get_tensor(ggml_ctx, tensor_name);
        if (!embd_tensor) {
            std::cerr << "Failed to get embedding tensor" << std::endl;
            return embeddings;
        }

        // Get dimensions of the embedding tensor
        const int64_t n_vocab = embd_tensor->ne[0]; // vocab size
        const int64_t n_embd = embd_tensor->ne[1];  // embedding dimension
        
        if (token_id >= n_vocab) {
            std::cerr << "Token ID out of range" << std::endl;
            return embeddings;
        }

        // Allocate memory for the embedding
        embeddings.resize(n_embd);

        // Convert the raw tensor data to float, handling quantization
        const size_t row_size = ggml_row_size(tensor_type, n_embd);
        std::vector<float> row_f32(n_embd);

        // Get pointer to the token's embedding in the tensor
        void* token_embedding = (char*)embd_tensor->data + token_id * row_size;
        
        // Convert quantized embedding to float
        const struct ggml_type_traits* qtype = ggml_get_type_traits(tensor_type);
        if (qtype->to_float) {
            qtype->to_float(token_embedding, row_f32.data(), n_embd);
        } else {
            // Fallback for non-quantized types
            memcpy(row_f32.data(), token_embedding, n_embd * sizeof(float));
        }

        // Copy to output
        for (int i = 0; i < n_embd; i++) {
            embeddings[i] = row_f32[i];
        }

        return embeddings;
    }

    std::vector<float> get_embeddings(const std::string& text) {
        // Tokenize the input text
        std::vector<int> tokens = tokenize(text);
        
        // Use the first token if available, otherwise default to token ID 1
        int token_id = tokens.empty() ? 1 : tokens[0];
        std::cout << "Using token ID " << token_id << " for embedding" << std::endl;
        
        return get_embeddings(token_id);
    }

    // Find the closest token to a given embedding using cosine similarity
    std::string find_closest_token(const std::vector<float>& embedding) {
        if (!ctx || embedding.empty()) {
            return "";
        }
        
        // Find the embedding tensor
        int64_t embedding_tensor_idx = gguf_find_tensor(ctx, "token_embd.weight");
        if (embedding_tensor_idx < 0) {
            return "";
        }
        
        // Get the tensor
        const char* tensor_name = gguf_get_tensor_name(ctx, embedding_tensor_idx);
        enum ggml_type tensor_type = gguf_get_tensor_type(ctx, embedding_tensor_idx);
        struct ggml_tensor* embd_tensor = ggml_get_tensor(ggml_ctx, tensor_name);
        
        if (!embd_tensor) {
            return "";
        }
        
        // Get dimensions
        const int64_t n_vocab = embd_tensor->ne[0];
        const int64_t n_embd = embd_tensor->ne[1];
        
        if (embedding.size() != n_embd) {
            return "Error: Embedding dimension mismatch";
        }
        
        // Calculate the norm of the query embedding
        float query_norm = 0.0f;
        for (size_t i = 0; i < embedding.size(); i++) {
            query_norm += embedding[i] * embedding[i];
        }
        query_norm = std::sqrt(query_norm);
        
        // Find the token with the highest cosine similarity
        float best_score = -1.0f;
        int best_token_id = -1;
        
        const size_t row_size = ggml_row_size(tensor_type, n_embd);
        std::vector<float> token_embd(n_embd);
        
        const struct ggml_type_traits* qtype = ggml_get_type_traits(tensor_type);
        
        for (int token_id = 0; token_id < n_vocab; token_id++) {
            // Get the token embedding
            void* token_embedding = (char*)embd_tensor->data + token_id * row_size;
            
            // Convert to float
            if (qtype->to_float) {
                qtype->to_float(token_embedding, token_embd.data(), n_embd);
            } else {
                memcpy(token_embd.data(), token_embedding, n_embd * sizeof(float));
            }
            
            // Calculate cosine similarity
            float dot_product = 0.0f;
            float token_norm = 0.0f;
            
            for (int i = 0; i < n_embd; i++) {
                dot_product += embedding[i] * token_embd[i];
                token_norm += token_embd[i] * token_embd[i];
            }
            
            token_norm = std::sqrt(token_norm);
            float cosine_similarity = dot_product / (query_norm * token_norm);
            
            if (cosine_similarity > best_score) {
                best_score = cosine_similarity;
                best_token_id = token_id;
            }
        }
        
        // Return the token text
        if (best_token_id >= 0 && id_to_token.find(best_token_id) != id_to_token.end()) {
            return id_to_token[best_token_id] + " (id: " + std::to_string(best_token_id) + 
                   ", score: " + std::to_string(best_score) + ")";
        } else {
            return "Token not found";
        }
    }

    std::vector<int> tokenize(const std::string& text) {
        std::vector<int> tokens;
        
        // In a production tokenizer, we would have proper byte-pair encoding or similar
        // This is a simplified version that just looks for exact token matches
        
        std::string remaining = text;
        std::cout << "Tokenizing: \"" << text << "\"" << std::endl;
        std::cout << "Token IDs: ";
        
        // Note about tokenization
        std::cout << "\n(Note: LLM tokenizers typically have special tokens for spaces. In many tokenizers," << std::endl;
        std::cout << "spaces are represented with a prefix like 'Ġ'. Capitalization also affects tokenization.)" << std::endl;
        std::cout << "\nTokens from C++ implementation: ";
        
        while (!remaining.empty()) {
            // Simple greedy tokenization - find the longest matching token
            std::string best_token = "";
            int best_id = -1;
            
            for (const auto& entry : token_to_id) {
                const std::string& token = entry.first;
                
                if (token.length() > best_token.length() && 
                    remaining.substr(0, token.length()) == token) {
                    best_token = token;
                    best_id = entry.second;
                }
            }
            
            if (best_id == -1) {
                // No match found, skip one character
                remaining = remaining.substr(1);
            } else {
                tokens.push_back(best_id);
                
                // Show token ID and original token representation
                std::cout << best_id;
                if (!best_token.empty()) {
                    std::cout << "(";
                    for (char c : best_token) {
                        if (isprint(c)) {
                            std::cout << c;
                        } else {
                            std::cout << "\\x" << std::hex << std::setw(2) << std::setfill('0') 
                                      << (int)(unsigned char)c;
                        }
                    }
                    std::cout << std::dec << std::setfill(' ') << ") ";
                } else {
                    std::cout << " ";
                }
                
                remaining = remaining.substr(best_token.length());
            }
        }
        
        std::cout << std::endl;
        std::cout << "Total tokens: " << tokens.size() << std::endl;
        
        return tokens;
    }
};

void print_usage() {
    std::cout << "Usage: tokenizer <model_path> <text>" << std::endl;
    std::cout << "  model_path: Path to the GGUF model file" << std::endl;
    std::cout << "  text: Text to embed" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage();
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string text = argv[2];

    // Initialize tokenizer
    Tokenizer tokenizer(model_path);
    
    // Load the model
    if (!tokenizer.load_model()) {
        return 1;
    }
    
    // Print minimal model info
    tokenizer.print_config();
    
    // Print vocabulary sample
    tokenizer.print_vocabulary_sample();
    
    // Tokenize the input text - this is the main focus
    std::vector<int> tokens = tokenizer.tokenize(text);
    
    return 0;
}