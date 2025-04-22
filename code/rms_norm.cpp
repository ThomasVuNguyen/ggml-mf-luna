#include "tokenizer.h"
#include "config.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

// Structure to hold our model and context
struct rms_model {
    struct ggml_tensor * input;
    struct ggml_tensor * weights;  // Optional weights for learned RMS norm
    struct ggml_context * ctx;
};

// Initialize the model with input tensor
void init_model(rms_model & model, const std::vector<float>& input_data) {
    size_t ctx_size = 0;
    {
        // Input tensor
        ctx_size += input_data.size() * ggml_type_size(GGML_TYPE_F32);
        // Weights tensor
        ctx_size += input_data.size() * ggml_type_size(GGML_TYPE_F32);
        // Intermediate tensors for computation
        ctx_size += 5 * input_data.size() * ggml_type_size(GGML_TYPE_F32); // squared, sum_sq, mean_sq, rms, normalized
        // Tensor overheads
        ctx_size += 7 * ggml_tensor_overhead(); // input, weights, squared, sum_sq, mean_sq, rms, normalized
        // Graph overhead
        ctx_size += ggml_graph_overhead();
        // Some extra space for safety
        ctx_size += 1024 * 1024; // 1MB extra
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

    // create weights tensor (initialized to 1.0 for basic RMS norm)
    model.weights = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, input_data.size());
    std::vector<float> ones(input_data.size(), 1.0f);
    memcpy(model.weights->data, ones.data(), ggml_nbytes(model.weights));
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
    struct ggml_tensor * eps_tensor = ggml_new_f32(model.ctx, eps);
    struct ggml_tensor * rms = ggml_sqrt(model.ctx, 
        ggml_add(model.ctx, mean_sq, eps_tensor));
    
    // Normalize input
    struct ggml_tensor * normalized = ggml_div(model.ctx, model.input, rms);
    
    // Apply weights if needed
    struct ggml_tensor * result = ggml_mul(model.ctx, normalized, model.weights);

    ggml_build_forward_expand(gf, result);
    return gf;
}

// Compute RMS normalization using GGML
std::vector<float> compute_rms_norm(const std::vector<float>& input, float eps = 1e-6) {
    rms_model model;
    init_model(model, input);

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
    std::vector<float> normalized = compute_rms_norm(embedding);
    
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
    std::cout << std::setw(10) << "Original" << std::setw(15) << "Normalized" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), embedding.size()); i++) {
        std::cout << std::setw(10) << std::setprecision(6) << embedding[i] 
                  << std::setw(15) << std::setprecision(6) << normalized[i] 
                  << std::endl;
    }
    
    return 0;
}
