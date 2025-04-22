#include "tokenizer.h"
#include <cstring>

Tokenizer::Tokenizer(const std::string& model_path) : model_path(model_path), ctx(nullptr), ggml_ctx(nullptr) {
    load_model();
}

Tokenizer::~Tokenizer() {
    if (ctx) {
        gguf_free(ctx);
    }
}

bool Tokenizer::load_model() {
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

bool Tokenizer::load_vocabulary() {
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

void Tokenizer::print_config() {
    std::cout << "Model loaded: " << model_path << std::endl;
    std::cout << "Vocabulary size: " << id_to_token.size() << std::endl;
}

void Tokenizer::print_vocabulary_sample(int n) {
    std::cout << "--- Vocabulary Sample ---" << std::endl;
    
    int count = 0;
    for (const auto& entry : id_to_token) {
        if (count >= n) break;
        
        std::cout << "ID " << std::setw(5) << entry.first << ": ";
        std::cout << entry.second << std::endl;
        
        count++;
    }
    
    std::cout << "------------------------" << std::endl;
}

std::vector<float> Tokenizer::get_embeddings(int token_id) const {
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
    const int64_t n_vocab = embd_tensor->ne[1]; // vocab size (second dimension)
    const int64_t n_embd = embd_tensor->ne[0];  // embedding dimension (first dimension)
    
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
        std::memcpy(row_f32.data(), token_embedding, n_embd * sizeof(float));
    }

    // Copy to output
    for (int i = 0; i < n_embd; i++) {
        embeddings[i] = row_f32[i];
    }

    return embeddings;
}

std::vector<float> Tokenizer::get_embeddings(const std::string& text) const {
    // Tokenize the input text
    std::vector<int> tokens = tokenize(text);
    
    // Use the first token if available, otherwise default to token ID 1
    int token_id = tokens.empty() ? 1 : tokens[0];
    std::cout << "Using token ID " << token_id << " for embedding" << std::endl;
    
    return get_embeddings(token_id);
}

std::string Tokenizer::find_closest_token(const std::vector<float>& embedding) {
    if (!ctx || embedding.empty()) {
        return "";
    }
    
    // Find the embedding tensor
    int64_t embedding_tensor_idx = gguf_find_tensor(ctx, "token_embd.weight");
    if (embedding_tensor_idx < 0) {
        return "";
    }
    
    // Get information about the embedding tensor
    const char* tensor_name = gguf_get_tensor_name(ctx, embedding_tensor_idx);
    enum ggml_type tensor_type = gguf_get_tensor_type(ctx, embedding_tensor_idx);
    
    // Create a ggml tensor to properly handle the embedding data
    struct ggml_tensor* embd_tensor = ggml_get_tensor(ggml_ctx, tensor_name);
    if (!embd_tensor) {
        return "";
    }

    // Get dimensions of the embedding tensor
    const int64_t n_vocab = embd_tensor->ne[0]; // vocab size
    const int64_t n_embd = embd_tensor->ne[1];  // embedding dimension
    
    // Make sure the provided embedding has the correct dimension
    if (embedding.size() != n_embd) {
        return "";
    }

    // Variables to track the closest token
    std::string closest_token = "";
    float max_similarity = -1.0f;

    // Memory for token embedding
    std::vector<float> token_embd(n_embd);
    
    // Process each token in the vocabulary
    const size_t row_size = ggml_row_size(tensor_type, n_embd);
    const struct ggml_type_traits* qtype = ggml_get_type_traits(tensor_type);
    
    for (int id = 0; id < n_vocab; id++) {
        // Get pointer to the token's embedding in the tensor
        void* token_embedding = (char*)embd_tensor->data + id * row_size;
        
        // Convert quantized embedding to float
        if (qtype->to_float) {
            qtype->to_float(token_embedding, token_embd.data(), n_embd);
        } else {
            // Fallback for non-quantized types
            std::memcpy(token_embd.data(), token_embedding, n_embd * sizeof(float));
        }
        
        // Calculate cosine similarity
        float dot_product = 0.0f;
        float norm_embd = 0.0f;
        float norm_token = 0.0f;
        
        for (int i = 0; i < n_embd; i++) {
            dot_product += embedding[i] * token_embd[i];
            norm_embd += embedding[i] * embedding[i];
            norm_token += token_embd[i] * token_embd[i];
        }
        
        norm_embd = std::sqrt(norm_embd);
        norm_token = std::sqrt(norm_token);
        
        float similarity = dot_product / (norm_embd * norm_token);
        
        // Update if this is the most similar token
        if (similarity > max_similarity) {
            max_similarity = similarity;
            closest_token = id_to_token[id];
        }
    }
    
    return closest_token;
}

std::vector<int> Tokenizer::tokenize(const std::string& text) const {
    std::vector<int> tokens;
    
    if (text.empty() || token_to_id.empty()) {
        return tokens;
    }
    
    // Check if there's a BOS (beginning of string) token
    std::string bos_token = "<s>";
    if (token_to_id.find(bos_token) != token_to_id.end()) {
        tokens.push_back(token_to_id.at(bos_token));
    }
    
    // Process text with special handling for spaces
    std::string processed_text = text;
    std::string current_text;
    bool first_token = true;
    
    size_t pos = 0;
    while (pos < processed_text.length()) {
        bool found = false;
        size_t max_len = std::min(static_cast<size_t>(20), processed_text.length() - pos);
        
        // Create possible prefixes, with special handling for spaces
        for (size_t len = max_len; len > 0; len--) {
            std::string substr = processed_text.substr(pos, len);
            
            // Try direct match
            if (token_to_id.find(substr) != token_to_id.end()) {
                tokens.push_back(token_to_id.at(substr));
                pos += len;
                found = true;
                first_token = false;
                break;
            }
            
            // Try with space prefix for non-first tokens (Ġ handling)
            if (!first_token && substr[0] == ' ' && token_to_id.find(substr) == token_to_id.end()) {
                // Try space variants - some models encode spaces specially
                std::string with_special_space = "Ġ" + substr.substr(1);  // Ġ prefix
                std::string with_underscore = "_" + substr.substr(1);     // _ prefix
                
                if (token_to_id.find(with_special_space) != token_to_id.end()) {
                    tokens.push_back(token_to_id.at(with_special_space));
                    pos += len;
                    found = true;
                    break;
                } else if (token_to_id.find(with_underscore) != token_to_id.end()) {
                    tokens.push_back(token_to_id.at(with_underscore));
                    pos += len;
                    found = true;
                    break;
                }
            }
        }
        
        // If no token was found, fall back to character-by-character tokenization
        if (!found) {
            std::string single_char = processed_text.substr(pos, 1);
            
            // Check with space prefix for non-first characters
            if (!first_token && single_char == " ") {
                std::string with_special_space = "Ġ";  // Ġ prefix alone
                std::string with_underscore = "_";     // _ prefix alone
                
                if (token_to_id.find(with_special_space) != token_to_id.end()) {
                    tokens.push_back(token_to_id.at(with_special_space));
                } else if (token_to_id.find(with_underscore) != token_to_id.end()) {
                    tokens.push_back(token_to_id.at(with_underscore));
                } else if (token_to_id.find(single_char) != token_to_id.end()) {
                    tokens.push_back(token_to_id.at(single_char));
                } else {
                    // Unknown token
                    tokens.push_back(0);
                }
            } else if (token_to_id.find(single_char) != token_to_id.end()) {
                tokens.push_back(token_to_id.at(single_char));
            } else {
                // Unknown token
                tokens.push_back(0);
            }
            
            pos++;
            first_token = false;
        }
    }
    
    // Check if there's an EOS (end of string) token
    std::string eos_token = "</s>";
    if (token_to_id.find(eos_token) != token_to_id.end()) {
        tokens.push_back(token_to_id.at(eos_token));
    }
    
    return tokens;
}

int64_t Tokenizer::get_hidden_size() {
    if (!ctx) {
        std::cerr << "Model not loaded" << std::endl;
        return -1;
    }

    // Find the embedding tensor
    int64_t embedding_tensor_idx = gguf_find_tensor(ctx, "token_embd.weight");
    if (embedding_tensor_idx < 0) {
        std::cerr << "Embedding tensor not found in model" << std::endl;
        return -1;
    }

    // Get tensor and its dimensions
    const char* tensor_name = gguf_get_tensor_name(ctx, embedding_tensor_idx);
    struct ggml_tensor* embd_tensor = ggml_get_tensor(ggml_ctx, tensor_name);
    if (!embd_tensor) {
        std::cerr << "Failed to get embedding tensor" << std::endl;
        return -1;
    }

    // Return the embedding dimension (hidden size)
    return embd_tensor->ne[1];
}

void print_token_embedding(const Tokenizer& tokenizer, int token_id) {
    std::vector<float> embedding = tokenizer.get_embeddings(token_id);
    if (embedding.empty()) {
        std::cout << "Failed to get embedding for token ID " << token_id << std::endl;
        return;
    }

    std::cout << "Embedding vector (first 10 dimensions):" << std::endl;
    for (size_t i = 0; i < std::min(size_t(10), embedding.size()); i++) {
        std::cout << std::setw(2) << i << ": " << std::setprecision(6) << embedding[i] << std::endl;
    }
} 