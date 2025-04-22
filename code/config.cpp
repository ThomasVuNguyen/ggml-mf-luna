#include "tokenizer.h"
#include "ggml/include/gguf.h"
#include "ggml/include/ggml.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

void extract_model_config(const Tokenizer& tokenizer) {
    std::cout << "\nExtracting model configuration..." << std::endl;
    
    // Get the GGUF context to extract model parameters
    struct gguf_context* ctx = tokenizer.get_gguf_context();
    if (!ctx) {
        std::cerr << "Failed to get GGUF context" << std::endl;
        return;
    }
    
    // Extract common model parameters
    const char* keys[] = {
        "general.architecture",
        "llama.attention.head_count",
        "llama.attention.head_count_kv",
        "llama.attention.layer_norm_rms_epsilon",
        "llama.context_length",
        "llama.embedding_length",
        "llama.feed_forward_length",
        "llama.rope.dimension_count",
        "llama.attention.rope_freq_base",
        "llama.attention.rope_scaling_type",
        "llama.attention.rope_scaling_factor",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "general.file_type",
        "llama.block_count",
        "general.name",
        "general.quantization_version",
        "tokenizer.ggml.model",
        "tokenizer.ggml.tokens",
    };
    
    // Print model configuration in JSON-like format
    std::cout << "{\n";
    
    for (const char* key : keys) {
        if (gguf_find_key(ctx, key) != -1) {
            enum gguf_type type = gguf_get_kv_type(ctx, gguf_find_key(ctx, key));
            
            std::cout << "  \"" << key << "\": ";
            
            switch (type) {
                case GGUF_TYPE_UINT32:
                    std::cout << gguf_get_val_u32(ctx, gguf_find_key(ctx, key));
                    break;
                case GGUF_TYPE_INT32:
                    std::cout << gguf_get_val_i32(ctx, gguf_find_key(ctx, key));
                    break;
                case GGUF_TYPE_FLOAT32:
                    std::cout << gguf_get_val_f32(ctx, gguf_find_key(ctx, key));
                    break;
                case GGUF_TYPE_STRING:
                    std::cout << "\"" << gguf_get_val_str(ctx, gguf_find_key(ctx, key)) << "\"";
                    break;
                case GGUF_TYPE_BOOL:
                    std::cout << (gguf_get_val_bool(ctx, gguf_find_key(ctx, key)) ? "true" : "false");
                    break;
                default:
                    std::cout << "\"<unsupported type>\"";
            }
            
            std::cout << ",\n";
        }
    }
    
    // Add vocabulary size
    std::cout << "  \"vocab_size\": " << tokenizer.id_to_token.size() << "\n";
    
    std::cout << "}" << std::endl;
}

void print_model_weights(const Tokenizer& tokenizer) {
    struct gguf_context* ctx = tokenizer.get_gguf_context();
    if (!ctx) {
        std::cerr << "Model not loaded" << std::endl;
        return;
    }
    
    int n_tensors = gguf_get_n_tensors(ctx);
    
    std::cout << "Model contains " << n_tensors << " tensors:\n";
    std::cout << "----------------------------------------------\n";
    std::cout << std::left << std::setw(40) << "Name" 
              << std::setw(15) << "Type" 
              << std::setw(20) << "Dimensions" 
              << "Size (bytes)\n";
    std::cout << "----------------------------------------------\n";
    
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(ctx, i);
        struct ggml_tensor* tensor = ggml_get_tensor(tokenizer.get_ggml_context(), name);
        
        if (!tensor) {
            continue;
        }
        
        // Get tensor type as string
        std::string type_str;
        switch (tensor->type) {
            case GGML_TYPE_F32:  type_str = "F32"; break;
            case GGML_TYPE_F16:  type_str = "F16"; break;
            case GGML_TYPE_Q4_0: type_str = "Q4_0"; break;
            case GGML_TYPE_Q4_1: type_str = "Q4_1"; break;
            case GGML_TYPE_Q5_0: type_str = "Q5_0"; break;
            case GGML_TYPE_Q5_1: type_str = "Q5_1"; break;
            case GGML_TYPE_Q8_0: type_str = "Q8_0"; break;
            case GGML_TYPE_Q8_1: type_str = "Q8_1"; break;
            case GGML_TYPE_Q2_K: type_str = "Q2_K"; break;
            case GGML_TYPE_Q3_K: type_str = "Q3_K"; break;
            case GGML_TYPE_Q4_K: type_str = "Q4_K"; break;
            case GGML_TYPE_Q5_K: type_str = "Q5_K"; break;
            case GGML_TYPE_Q6_K: type_str = "Q6_K"; break;
            case GGML_TYPE_Q8_K: type_str = "Q8_K"; break;
            default:             type_str = "Unknown"; break;
        }
        
        // Format dimensions
        std::string dims_str = "(";
        for (int j = 0; j < GGML_MAX_DIMS; j++) {
            if (tensor->ne[j] == 1 && j > 0) {
                break;
            }
            dims_str += std::to_string(tensor->ne[j]);
            if (j < GGML_MAX_DIMS - 1 && tensor->ne[j+1] > 1) {
                dims_str += ", ";
            }
        }
        dims_str += ")";
        
        // Calculate size in bytes
        size_t size_bytes = ggml_nbytes(tensor);
        
        // Print tensor info
        std::cout << std::left << std::setw(40) << name 
                  << std::setw(15) << type_str 
                  << std::setw(20) << dims_str 
                  << size_bytes << std::endl;
    }
}

void print_token_embedding(const Tokenizer& tokenizer, int token_id) {
    struct gguf_context* ctx = tokenizer.get_gguf_context();
    if (!ctx) {
        std::cerr << "Model not loaded" << std::endl;
        return;
    }
    
    // Find the token_embd.weight tensor
    const char* embd_name = "token_embd.weight";
    struct ggml_tensor* embd_tensor = ggml_get_tensor(tokenizer.get_ggml_context(), embd_name);
    
    if (!embd_tensor) {
        std::cerr << "Couldn't find token embedding tensor" << std::endl;
        return;
    }
    
    std::cout << "Embeddings for token ID " << token_id << ":" << std::endl;
    
    // Get embedding dimensions
    int embd_dim = embd_tensor->ne[0]; // Embedding dimension
    int vocab_size = embd_tensor->ne[1]; // Vocabulary size
    
    if (token_id >= vocab_size) {
        std::cerr << "Token ID exceeds vocabulary size" << std::endl;
        return;
    }
    
    // Print information about embedding dimensions
    std::cout << "Embedding dimension: " << embd_dim << std::endl;
    std::cout << "Tensor type: ";
    switch (embd_tensor->type) {
        case GGML_TYPE_F32:  std::cout << "F32"; break;
        case GGML_TYPE_F16:  std::cout << "F16"; break;
        case GGML_TYPE_Q4_0: std::cout << "Q4_0"; break;
        case GGML_TYPE_Q4_1: std::cout << "Q4_1"; break;
        case GGML_TYPE_Q5_0: std::cout << "Q5_0"; break;
        case GGML_TYPE_Q5_1: std::cout << "Q5_1"; break;
        case GGML_TYPE_Q8_0: std::cout << "Q8_0"; break;
        case GGML_TYPE_Q8_1: std::cout << "Q8_1"; break;
        default:             std::cout << "Other (" << embd_tensor->type << ")"; break;
    }
    std::cout << std::endl;
    
    // Array to store the first few values we want to display
    const int max_display = 3;
    float display_values[max_display] = {0.0f};
    bool values_extracted = false;
    
    // Handle different tensor types
    switch (embd_tensor->type) {
        case GGML_TYPE_F32: {
            float* data = (float*)embd_tensor->data;
            for (int i = 0; i < max_display; i++) {
                display_values[i] = data[token_id * embd_dim + i];
            }
            values_extracted = true;
            break;
        }
        case GGML_TYPE_F16: {
            uint16_t* data = (uint16_t*)embd_tensor->data;
            for (int i = 0; i < max_display; i++) {
                // Simple F16 to F32 conversion (not perfect but illustrative)
                uint16_t half = data[token_id * embd_dim + i];
                int sign = (half >> 15) & 0x1;
                int exp = (half >> 10) & 0x1F;
                int mant = half & 0x3FF;
                
                float val;
                if (exp == 0) {
                    val = (mant == 0) ? 0.0f : (sign ? -1.0f : 1.0f) * powf(2.0f, -14.0f) * (mant / 1024.0f);
                } else if (exp == 31) {
                    val = mant == 0 ? (sign ? -INFINITY : INFINITY) : NAN;
                } else {
                    val = (sign ? -1.0f : 1.0f) * powf(2.0f, exp - 15.0f) * (1.0f + mant / 1024.0f);
                }
                display_values[i] = val;
            }
            values_extracted = true;
            break;
        }
        case GGML_TYPE_Q8_0: {
            // Q8_0 format: block size is 32 elements
            // Each block has a scale value (f32) followed by 32 int8 values
            // For each block: actual_value = scale * int8_value
            
            const int block_size = 32;
            // Determine which block(s) we need
            int start_block = (token_id * embd_dim) / block_size;
            
            // Get pointer to the beginning of data
            uint8_t* data = (uint8_t*)embd_tensor->data;
            
            // For each value we want to display
            for (int i = 0; i < max_display; i++) {
                // Calculate which block this element belongs to
                int element_idx = token_id * embd_dim + i;
                int block_idx = element_idx / block_size;
                int offset_in_block = element_idx % block_size;
                
                // Each block has a scale (float) followed by int8 values
                size_t block_offset = block_idx * (sizeof(float) + block_size);
                
                // Get scale value (float at the beginning of the block)
                float scale = *(float*)(data + block_offset);
                
                // Get quantized value (int8 at the offset in the block)
                int8_t q_value = *(int8_t*)(data + block_offset + sizeof(float) + offset_in_block);
                
                // Dequantize: scaled_value = scale * int8_value
                display_values[i] = scale * q_value;
            }
            values_extracted = true;
            break;
        }
        default:
            std::cout << "Unsupported tensor type for displaying embeddings" << std::endl;
            break;
    }
    
    if (values_extracted) {
        // Print the extracted values
        std::cout << "First " << max_display << " embedding values: ";
        for (int i = 0; i < max_display; i++) {
            std::cout << display_values[i];
            if (i < max_display - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }
}

int main() {
    std::cout << "Loading tokenizer..." << std::endl;
    Tokenizer tokenizer("./gguf/1b-q8_0.gguf");
    
    // Extract and print model configuration
    extract_model_config(tokenizer);
    
    // Print model weights
    print_model_weights(tokenizer);
    
    // Print embeddings for a sample token
    print_token_embedding(tokenizer, 0);
    
    return 0;
} 