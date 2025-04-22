#include "tokenizer.h"
#include "config.h"
#include "ggml/include/ggml.h"
#include "ggml/include/ggml-cpu.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

// Structure to hold our model and context
struct rms_model {
    struct ggml_tensor * input;
    struct ggml_tensor * weights;
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
        ctx_size += 5 * input_data.size() * ggml_type_size(GGML_TYPE_F32);
        // Tensor overheads
        ctx_size += 7 * ggml_tensor_overhead();
        // Graph overhead
        ctx_size += ggml_graph_overhead();
        // Extra space
        ctx_size += 1024 * 1024;
    }

    struct ggml_init_params params {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    model.ctx = ggml_init(params);
    model.input = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, input_data.size());
    memcpy(model.input->data, input_data.data(), ggml_nbytes(model.input));

    model.weights = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, input_data.size());
    std::vector<float> ones(input_data.size(), 1.0f);
    memcpy(model.weights->data, ones.data(), ggml_nbytes(model.weights));
}

// Build the compute graph for RMS normalization
struct ggml_cgraph * build_graph(const rms_model& model, float eps = 1e-6) {
    struct ggml_cgraph * gf = ggml_new_graph(model.ctx);

    struct ggml_tensor * squared = ggml_sqr(model.ctx, model.input);
    struct ggml_tensor * sum_sq = ggml_sum(model.ctx, squared);
    struct ggml_tensor * mean_sq = ggml_scale(model.ctx, sum_sq, 1.0f / model.input->ne[0]);
    struct ggml_tensor * eps_tensor = ggml_new_f32(model.ctx, eps);
    struct ggml_tensor * rms = ggml_sqrt(model.ctx, 
        ggml_add(model.ctx, mean_sq, eps_tensor));
    struct ggml_tensor * normalized = ggml_div(model.ctx, model.input, rms);
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

    struct ggml_tensor * result = ggml_graph_node(gf, -1);
    
    std::vector<float> output(ggml_nelements(result));
    memcpy(output.data(), result->data, ggml_nbytes(result));

    ggml_free(model.ctx);
    
    return output;
}

void print_embedding_stats(const std::vector<float>& embedding, const std::string& label) {
    std::cout << "\n" << label << " statistics:" << std::endl;
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
}

int main() {
    ggml_time_init();
    
    // Load model and tokenizer
    std::cout << "Loading model and tokenizer..." << std::endl;
    Tokenizer tokenizer("./gguf/1b-q8_0.gguf");
    
    // Get input prompt
    std::string prompt;
    std::cout << "Enter your prompt: ";
    std::getline(std::cin, prompt);
    
    // Tokenize the prompt
    std::cout << "\nTokenizing prompt: '" << prompt << "'" << std::endl;
    std::vector<int> tokens = tokenizer.tokenize(prompt);
    
    // Process each token
    for (size_t i = 0; i < tokens.size(); i++) {
        std::cout << "\nProcessing token " << i << " (ID: " << tokens[i] << ")" << std::endl;
        
        // Get token text
        auto it = tokenizer.id_to_token.find(tokens[i]);
        if (it != tokenizer.id_to_token.end()) {
            std::cout << "Token text: '" << it->second << "'" << std::endl;
        }
        
        // Get embeddings
        std::vector<float> embedding = tokenizer.get_embeddings(tokens[i]);
        if (embedding.empty()) {
            std::cerr << "Failed to get embeddings for token " << tokens[i] << std::endl;
            continue;
        }
        
        // Print original embedding stats
        print_embedding_stats(embedding, "Original embedding");
        
        // Perform RMS normalization
        std::vector<float> normalized = compute_rms_norm(embedding);
        
        // Print normalized embedding stats
        print_embedding_stats(normalized, "After RMS normalization");
        
        // Print first few values comparison
        std::cout << "\nFirst 5 values comparison:" << std::endl;
        std::cout << std::setw(10) << "Original" << std::setw(15) << "Normalized" << std::endl;
        for (size_t j = 0; j < std::min(size_t(5), embedding.size()); j++) {
            std::cout << std::setw(10) << std::setprecision(6) << embedding[j] 
                      << std::setw(15) << std::setprecision(6) << normalized[j] 
                      << std::endl;
        }
    }
    
    return 0;
} 