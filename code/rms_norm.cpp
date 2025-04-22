#include "tokenizer.h"
#include "config.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "rms_norm.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <map>

// Structure to hold our model and context
struct rms_model {
    struct ggml_tensor * input;
    struct ggml_tensor * weights;  // RMS norm weights from model
    struct ggml_context * ctx;
};

// Map of layer names to their RMS norm weights (caching)
static std::map<std::string, std::vector<float>> layer_weights_cache;

// Get model weights for a specific layer's RMS norm
std::vector<float> get_layer_rms_weights(const Tokenizer& tokenizer, const std::string& layer_name) {
    struct gguf_context* ctx = tokenizer.get_gguf_context();
    if (!ctx) {
        std::cerr << "Model not loaded" << std::endl;
        return {};
    }
    
    // Find the RMS norm weight tensor
    struct ggml_tensor* weight_tensor = ggml_get_tensor(tokenizer.get_ggml_context(), layer_name.c_str());
    
    if (!weight_tensor) {
        std::cerr << "Couldn't find RMS norm weight tensor: " << layer_name << std::endl;
        return {};
    }
    
    int embd_dim = weight_tensor->ne[0];
    std::vector<float> weights(embd_dim, 1.0f); // Default to 1.0 if extraction fails
    
    // Extract weights based on tensor type
    if (weight_tensor->type == GGML_TYPE_F32) {
        float* data = (float*)weight_tensor->data;
        for (int i = 0; i < embd_dim; i++) {
            weights[i] = data[i];
        }
    } else {
        std::cerr << "Unsupported tensor type for RMS norm weights" << std::endl;
    }
    
    return weights;
}

// Get epsilon value from model config
float get_rms_epsilon(const Tokenizer& tokenizer) {
    struct gguf_context* ctx = tokenizer.get_gguf_context();
    if (!ctx) {
        return 1e-6f; // Default epsilon
    }
    
    const char* eps_key = "llama.attention.layer_norm_rms_epsilon";
    int key_idx = gguf_find_key(ctx, eps_key);
    
    if (key_idx != -1 && gguf_get_kv_type(ctx, key_idx) == GGUF_TYPE_FLOAT32) {
        return gguf_get_val_f32(ctx, key_idx);
    }
    
    return 1e-6f; // Default epsilon if not found
}

// Initialize the model with input tensor and model weights
void init_model(rms_model & model, const std::vector<float>& input_data, const std::vector<float>& weights) {
    size_t ctx_size = 0;
    {
        // Input tensor
        ctx_size += input_data.size() * ggml_type_size(GGML_TYPE_F32);
        // Weights tensor
        ctx_size += weights.size() * ggml_type_size(GGML_TYPE_F32);
        // Intermediate tensors for computation
        ctx_size += 5 * input_data.size() * ggml_type_size(GGML_TYPE_F32);
        // Tensor overheads
        ctx_size += 7 * ggml_tensor_overhead();
        // Graph overhead
        ctx_size += ggml_graph_overhead();
        // Some extra space for safety
        ctx_size += 1024 * 1024;
    }

    struct ggml_init_params params {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    // create context
    model.ctx = ggml_init(params);

    // create input tensor
    model.input = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, input_data.size());
    memcpy(model.input->data, input_data.data(), ggml_nbytes(model.input));

    // create weights tensor with model weights
    model.weights = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, weights.size());
    memcpy(model.weights->data, weights.data(), ggml_nbytes(model.weights));
}

// Build the compute graph for RMS normalization
struct ggml_cgraph * build_graph(const rms_model& model, float eps = 1e-6) {
    struct ggml_cgraph * gf = ggml_new_graph(model.ctx);

    // Calculate sum of squares
    struct ggml_tensor * squared = ggml_sqr(model.ctx, model.input);
    struct ggml_tensor * sum_sq = ggml_sum(model.ctx, squared);
    
    // Calculate mean of squares
    struct ggml_tensor * mean_sq = ggml_scale(model.ctx, sum_sq, 1.0f / model.input->ne[0]);
    
    // Add epsilon and take square root
    // Create a constant tensor for epsilon
    struct ggml_tensor * eps_tensor = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, 1);
    *(float *)eps_tensor->data = eps;
    
    struct ggml_tensor * rms = ggml_sqrt(model.ctx, 
        ggml_add(model.ctx, mean_sq, eps_tensor));
    
    // Normalize input
    struct ggml_tensor * normalized = ggml_div(model.ctx, model.input, rms);
    
    // Apply model weights
    struct ggml_tensor * result = ggml_mul(model.ctx, normalized, model.weights);

    ggml_build_forward_expand(gf, result);
    return gf;
}

// Compute RMS normalization using GGML and model weights
std::vector<float> compute_rms_norm(const Tokenizer& tokenizer, 
                                    const std::vector<float>& input, 
                                    const std::string& layer_name) {
    // Get epsilon from model config
    float eps = get_rms_epsilon(tokenizer);
    
    // Get or cache layer weights
    std::vector<float> weights;
    auto it = layer_weights_cache.find(layer_name);
    if (it != layer_weights_cache.end()) {
        weights = it->second;
    } else {
        weights = get_layer_rms_weights(tokenizer, layer_name);
        layer_weights_cache[layer_name] = weights;
    }
    
    // If no weights found, use default weights (all 1.0)
    if (weights.empty() || weights.size() != input.size()) {
        weights = std::vector<float>(input.size(), 1.0f);
    }
    
    // Initialize model with input and weights
    rms_model model;
    init_model(model, input, weights);

    // Build and compute graph
    struct ggml_cgraph * gf = build_graph(model, eps);
    
    int n_threads = 1;
    ggml_graph_compute_with_ctx(model.ctx, gf, n_threads);

    // Get the result tensor
    struct ggml_tensor * result = ggml_graph_node(gf, -1);
    
    // Copy result to vector
    std::vector<float> output(ggml_nelements(result));
    memcpy(output.data(), result->data, ggml_nbytes(result));

    // Free memory
    ggml_free(model.ctx);
    
    return output;
}

// Print vectors in a clean, tabular format
void print_vector_head(const std::vector<float>& vec, const std::string& label, int n) {
    std::cout << std::left << std::setw(20) << label;
    for (int i = 0; i < std::min(n, (int)vec.size()); i++) {
        std::cout << std::fixed << std::setprecision(6) << std::setw(15) << vec[i];
    }
    std::cout << std::endl;
}

int main() {
    ggml_time_init();
    
    std::cout << "Loading model..." << std::endl;
    Tokenizer tokenizer("./gguf/1b-q8_0.gguf");
    
    // Get a sample embedding
    int sample_token_id = 1;
    std::vector<float> embedding = tokenizer.get_embeddings(sample_token_id);
    
    if (embedding.empty()) {
        std::cerr << "Failed to get embeddings" << std::endl;
        return 1;
    }
    
    // Print original embedding stats
    std::cout << "\nOriginal embedding statistics:" << std::endl;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (float x : embedding) {
        sum += x;
        sum_sq += x * x;
    }
    float mean = sum / embedding.size();
    float std_dev = std::sqrt(sum_sq / embedding.size() - mean * mean);
    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Standard deviation: " << std_dev << std::endl;
    
    // Perform RMS normalization using GGML
    std::vector<float> normalized = compute_rms_norm(tokenizer, embedding, "output_norm.weight");
    
    // Print normalized embedding stats
    std::cout << "\nAfter RMS normalization:" << std::endl;
    sum = 0.0f;
    sum_sq = 0.0f;
    for (float x : normalized) {
        sum += x;
        sum_sq += x * x;
    }
    mean = sum / normalized.size();
    std_dev = std::sqrt(sum_sq / normalized.size() - mean * mean);
    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Standard deviation: " << std_dev << std::endl;
    
    // Print first few values before and after normalization
    std::cout << "\nFirst 5 values comparison:" << std::endl;
    print_vector_head(embedding, "Original:", 5);
    print_vector_head(normalized, "Normalized:", 5);
    
    return 0;
}
