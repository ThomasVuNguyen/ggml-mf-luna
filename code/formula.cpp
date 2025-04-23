#include "tokenizer.h"
#include "config.h"
#include "../ggml/include/ggml.h"
#include "../ggml/include/ggml-cpu.h"
#include "../ggml/include/gguf.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <fstream> // For debug file output

// Helper function to check matrix multiplication compatibility
bool check_matrix_compatibility(struct ggml_tensor* a, struct ggml_tensor* b) {
    // For matrix multiplication a*b, the inner dimensions must match
    // a: [m, k], b: [k, n] -> result: [m, n]
    return a->ne[0] == b->ne[1];
}

// Safe matrix multiplication that checks compatibility first
struct ggml_tensor* safe_mul_mat(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b, const char* op_name) {
    std::ofstream debug_file("formula_debug.txt", std::ios_base::app);
    debug_file << "Attempting " << op_name << " matrix multiply:" << std::endl;
    debug_file << "  A dims: " << a->ne[0] << "x" << a->ne[1] << ", type=" << a->type << std::endl;
    debug_file << "  B dims: " << b->ne[0] << "x" << b->ne[1] << ", type=" << b->type << std::endl;
    
    // Log warning if tensors are quantized
    if (a->type != GGML_TYPE_F32 && a->type != GGML_TYPE_F16) {
        debug_file << "  Warning: A is quantized (type=" << a->type << ")" << std::endl;
    }
    
    if (b->type != GGML_TYPE_F32 && b->type != GGML_TYPE_F16) {
        debug_file << "  Warning: B is quantized (type=" << b->type << ")" << std::endl;
    } 
    
    // For GGML, matrix A should be of shape [m, k] and B of shape [n, k]
    // Then we get a result of shape [m, n]
    // But for matmul, we need A's columns to match B's rows
    // If incompatible, try transposing B
    if (a->ne[0] != b->ne[1]) {
        debug_file << "  Dimensions don't match directly, trying to transpose B..." << std::endl;
        struct ggml_tensor* b_transposed = ggml_transpose(ctx, b);
        debug_file << "  B transposed dims: " << b_transposed->ne[0] << "x" << b_transposed->ne[1] << std::endl;
        
        if (a->ne[0] == b_transposed->ne[1]) {
            debug_file << "  Matrix multiply is compatible after transposing B" << std::endl;
            debug_file.close();
            try {
                return ggml_mul_mat(ctx, a, b_transposed);
            } catch (...) {
                debug_file.open("formula_debug.txt", std::ios_base::app);
                debug_file << "ERROR: Exception in matrix multiplication after transposing!" << std::endl;
                debug_file.close();
                
                // Create placeholder tensor
                struct ggml_tensor* result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, b_transposed->ne[0], a->ne[1]);
                return result;
            }
        }
        
        debug_file << "ERROR: Incompatible dimensions for " << op_name << " matrix multiply!" << std::endl;
        debug_file << "  A inner dim (ne[0]): " << a->ne[0] << " != B inner dim (ne[1]): " << b->ne[1] << std::endl;
        debug_file.close();
        
        // Instead of crashing, we'll create a properly sized tensor with zeros
        // This allows debugging to continue and see other issues
        struct ggml_tensor* result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, b_transposed->ne[0], a->ne[1]);
        return result;
    }
    
    debug_file << "  Matrix multiply is compatible" << std::endl;
    debug_file.close();
    
    // Instead of try/catch, check for potential errors beforehand
    if (a->type > GGML_TYPE_COUNT || b->type > GGML_TYPE_COUNT) {
        std::ofstream debug_file("formula_debug.txt", std::ios_base::app);
        debug_file << "ERROR: Invalid tensor type!" << std::endl;
        debug_file.close();
        
        // Return a zero tensor with proper dimensions
        return ggml_new_tensor_2d(ctx, GGML_TYPE_F32, b->ne[0], a->ne[1]);
    }
    
    // Try the multiplication, but if it fails, at least we've logged the parameters
    return ggml_mul_mat(ctx, a, b);
}

// Structure to hold the model's computational graph
struct ModelFormula {
    // Tokenization and embedding structures
    struct ggml_tensor* input_tokens;
    struct ggml_tensor* token_embeddings;
    
    // Transformer layer structures
    struct ggml_tensor* pre_norm_weights;
    struct ggml_tensor* ffn_norm_weights;
    struct ggml_tensor* q_weights;
    struct ggml_tensor* k_weights;
    struct ggml_tensor* v_weights;
    struct ggml_tensor* o_weights;
    struct ggml_tensor* gate_weights;
    struct ggml_tensor* up_weights;
    struct ggml_tensor* down_weights;
    
    // Final normalization weights
    struct ggml_tensor* output_norm_weights;
    
    // Model dimensions
    int hidden_size;
    int n_heads;
    int head_size;
    int n_kv_heads;
    int kv_head_size;
    int intermediate_size;
    float rms_norm_eps;
    
    // GGML context
    struct ggml_context* ctx;
};

// Initialize the model formula with weights from GGUF model
void init_model_formula(ModelFormula& model, const Tokenizer& tokenizer) {
    // Get model dimensions from config
    struct gguf_context* gguf_ctx = tokenizer.get_gguf_context();
    if (!gguf_ctx) {
        std::cerr << "Failed to get GGUF context" << std::endl;
        return;
    }
    
    // Extract basic model configuration
    int key_idx;
    key_idx = gguf_find_key(gguf_ctx, "llama.embedding_length");
    model.hidden_size = (key_idx != -1) ? gguf_get_val_u32(gguf_ctx, key_idx) : 2048;
    
    key_idx = gguf_find_key(gguf_ctx, "llama.attention.head_count");
    model.n_heads = (key_idx != -1) ? gguf_get_val_u32(gguf_ctx, key_idx) : 32;
    
    key_idx = gguf_find_key(gguf_ctx, "llama.attention.head_count_kv");
    model.n_kv_heads = (key_idx != -1) ? gguf_get_val_u32(gguf_ctx, key_idx) : 8;
    
    key_idx = gguf_find_key(gguf_ctx, "llama.feed_forward_length");
    model.intermediate_size = (key_idx != -1) ? gguf_get_val_u32(gguf_ctx, key_idx) : 8192;
    
    key_idx = gguf_find_key(gguf_ctx, "llama.attention.layer_norm_rms_epsilon");
    model.rms_norm_eps = (key_idx != -1) ? gguf_get_val_f32(gguf_ctx, key_idx) : 1e-6f;
    
    model.head_size = model.hidden_size / model.n_heads;
    model.kv_head_size = model.hidden_size / model.n_heads;
    
    std::cout << "Model configuration:" << std::endl;
    std::cout << "  Hidden size: " << model.hidden_size << std::endl;
    std::cout << "  Number of heads: " << model.n_heads << std::endl;
    std::cout << "  Number of KV heads: " << model.n_kv_heads << std::endl;
    std::cout << "  Head size: " << model.head_size << std::endl;
    std::cout << "  Intermediate size: " << model.intermediate_size << std::endl;
    std::cout << "  RMS norm epsilon: " << model.rms_norm_eps << std::endl;
}

// Build the computational graph for one transformer layer
struct ggml_tensor* build_transformer_layer(
    ModelFormula& model,
    struct ggml_context* ctx,
    struct ggml_tensor* input,
    int layer_idx
) {
    // Start with fresh debug file for this layer
    if (layer_idx == 0) {
        std::ofstream debug_file("formula_debug.txt");
        debug_file << "====== Starting transformer inference ======" << std::endl;
        debug_file.close();
    }
    
    std::ofstream debug_file("formula_debug.txt", std::ios_base::app);
    debug_file << "\n====== Building transformer layer " << layer_idx << " ======" << std::endl;
    
    // Check that input is valid
    if (!input) {
        debug_file << "Error: Input tensor is NULL" << std::endl;
        debug_file.close();
        return nullptr;
    }
    
    debug_file << "STEP 0: Initial setup" << std::endl;
    // Print input dimensions immediately
    debug_file << "Input dimensions: ne0=" << input->ne[0] 
              << ", ne1=" << input->ne[1] 
              << ", type=" << input->type << std::endl;
    
    // Layer parameter names
    std::string attn_norm_name = "blk." + std::to_string(layer_idx) + ".attn_norm.weight";
    std::string ffn_norm_name = "blk." + std::to_string(layer_idx) + ".ffn_norm.weight";
    std::string q_name = "blk." + std::to_string(layer_idx) + ".attn_q.weight";
    std::string k_name = "blk." + std::to_string(layer_idx) + ".attn_k.weight";
    std::string v_name = "blk." + std::to_string(layer_idx) + ".attn_v.weight";
    std::string o_name = "blk." + std::to_string(layer_idx) + ".attn_output.weight";
    std::string gate_name = "blk." + std::to_string(layer_idx) + ".ffn_gate.weight";
    std::string up_name = "blk." + std::to_string(layer_idx) + ".ffn_up.weight";
    std::string down_name = "blk." + std::to_string(layer_idx) + ".ffn_down.weight";
    
    debug_file << "STEP 1: Loading model weights (inspection only)" << std::endl;
    
    // Just log the weight tensor sizes to understand the model structure
    model.q_weights = ggml_get_tensor(model.ctx, q_name.c_str());
    model.k_weights = ggml_get_tensor(model.ctx, k_name.c_str());
    model.v_weights = ggml_get_tensor(model.ctx, v_name.c_str());
    model.o_weights = ggml_get_tensor(model.ctx, o_name.c_str());
    model.pre_norm_weights = ggml_get_tensor(model.ctx, attn_norm_name.c_str());
    model.ffn_norm_weights = ggml_get_tensor(model.ctx, ffn_norm_name.c_str());
    model.gate_weights = ggml_get_tensor(model.ctx, gate_name.c_str());
    model.up_weights = ggml_get_tensor(model.ctx, up_name.c_str());
    model.down_weights = ggml_get_tensor(model.ctx, down_name.c_str());
    
    // Log weight tensor information
    if (model.q_weights) debug_file << "Q weights: ne0=" << model.q_weights->ne[0] << ", ne1=" << model.q_weights->ne[1] << ", type=" << model.q_weights->type << std::endl;
    if (model.k_weights) debug_file << "K weights: ne0=" << model.k_weights->ne[0] << ", ne1=" << model.k_weights->ne[1] << ", type=" << model.k_weights->type << std::endl;
    if (model.v_weights) debug_file << "V weights: ne0=" << model.v_weights->ne[0] << ", ne1=" << model.v_weights->ne[1] << ", type=" << model.v_weights->type << std::endl;
    if (model.o_weights) debug_file << "O weights: ne0=" << model.o_weights->ne[0] << ", ne1=" << model.o_weights->ne[1] << ", type=" << model.o_weights->type << std::endl;
    if (model.pre_norm_weights) debug_file << "Pre-norm weights: ne0=" << model.pre_norm_weights->ne[0] << ", ne1=" << model.pre_norm_weights->ne[1] << ", type=" << model.pre_norm_weights->type << std::endl;
    if (model.ffn_norm_weights) debug_file << "FFN norm weights: ne0=" << model.ffn_norm_weights->ne[0] << ", ne1=" << model.ffn_norm_weights->ne[1] << ", type=" << model.ffn_norm_weights->type << std::endl;
    if (model.gate_weights) debug_file << "Gate weights: ne0=" << model.gate_weights->ne[0] << ", ne1=" << model.gate_weights->ne[1] << ", type=" << model.gate_weights->type << std::endl;
    if (model.up_weights) debug_file << "Up weights: ne0=" << model.up_weights->ne[0] << ", ne1=" << model.up_weights->ne[1] << ", type=" << model.up_weights->type << std::endl;
    if (model.down_weights) debug_file << "Down weights: ne0=" << model.down_weights->ne[0] << ", ne1=" << model.down_weights->ne[1] << ", type=" << model.down_weights->type << std::endl;
    
    debug_file << "STEP 2: Creating pass-through implementation" << std::endl;
    debug_file << "Instead of actual computation, just returning a copy of the input" << std::endl;
    
    // Create a new tensor with the same shape as the input
    struct ggml_tensor* output = ggml_dup(ctx, input);
    
    debug_file << "Layer " << layer_idx << " completed (pass-through mode)" << std::endl;
    debug_file.close();
    
    return output;
}

// Build the full computational graph for inference
struct ggml_tensor* build_transformer_inference_graph(
    ModelFormula& model,
    struct ggml_context* ctx,
    struct ggml_tensor* input_embeddings,
    int n_layers
) {
    struct ggml_tensor* hidden_states = input_embeddings;
    
    // Forward through all layers
    for (int i = 0; i < n_layers; i++) {
        hidden_states = build_transformer_layer(model, ctx, hidden_states, i);
        if (!hidden_states) {
            std::cerr << "Layer " << i << " failed" << std::endl;
            return nullptr;
        }
    }
    
    // Step 4: Final Layer Normalization
    struct ggml_tensor* final_hidden_states_squared = ggml_sqr(ctx, hidden_states);
    struct ggml_tensor* final_hidden_states_mean = ggml_mean(ctx, final_hidden_states_squared);
    
    // Create epsilon tensor
    struct ggml_tensor* eps_tensor3 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    *(float*)eps_tensor3->data = model.rms_norm_eps;
    
    struct ggml_tensor* final_hidden_states_rms = ggml_sqrt(ctx, 
        ggml_add(ctx, final_hidden_states_mean, eps_tensor3));
    
    // Get output norm weights
    std::string output_norm_name = "output_norm.weight";
    model.output_norm_weights = ggml_get_tensor(model.ctx, output_norm_name.c_str());
    if (!model.output_norm_weights) {
        std::cerr << "Failed to find output norm weights: " << output_norm_name << std::endl;
        return nullptr;
    }
    
    struct ggml_tensor* final_normalized = ggml_div(ctx, hidden_states, final_hidden_states_rms);
    final_normalized = ggml_mul(ctx, final_normalized, model.output_norm_weights);
    
    // Step 5: Language Modeling Head (Token Prediction)
    // In a real implementation, we would multiply by the token embedding matrix transposed
    // For simplicity, we'll just return the normalized hidden states
    
    return final_normalized;
}

// Example of token embedding lookup
struct ggml_tensor* get_token_embeddings(
    ModelFormula& model,
    struct ggml_context* ctx,
    const std::vector<int>& tokens
) {
    int n_tokens = tokens.size();
    int hidden_size = model.hidden_size;
    
    // Create token embeddings tensor
    struct ggml_tensor* embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, n_tokens);
    
    // Get token embedding weights
    std::string embedding_name = "token_embd.weight";
    struct ggml_tensor* embedding_weights = ggml_get_tensor(model.ctx, embedding_name.c_str());
    if (!embedding_weights) {
        std::cerr << "Failed to find token embedding weights: " << embedding_name << std::endl;
        return nullptr;
    }
    
    // Create input tensor for token indices
    struct ggml_tensor* input_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    int32_t* token_data = (int32_t*)input_tokens->data;
    for (int i = 0; i < n_tokens; i++) {
        token_data[i] = tokens[i];
    }
    
    // Use embedding lookup operation
    embeddings = ggml_get_rows(ctx, embedding_weights, input_tokens);
    
    return embeddings;
}

// Perform inference on input tokens
void inference_tokens(const Tokenizer& tokenizer, const std::vector<int>& tokens) {
    std::cout << "Building computational graph for inference..." << std::endl;
    
    // Initialize model formula
    ModelFormula model;
    model.ctx = tokenizer.get_ggml_context();
    init_model_formula(model, tokenizer);
    
    // Get number of layers
    struct gguf_context* gguf_ctx = tokenizer.get_gguf_context();
    int key_idx = gguf_find_key(gguf_ctx, "llama.block_count");
    int n_layers = (key_idx != -1) ? gguf_get_val_u32(gguf_ctx, key_idx) : 16;
    
    std::cout << "Number of layers: " << n_layers << std::endl;
    
    // Create context for computation
    size_t compute_size = 256*1024*1024; // 256 MB should be sufficient for small examples
    void* compute_mem = malloc(compute_size);
    if (!compute_mem) {
        std::cerr << "Failed to allocate compute memory" << std::endl;
        return;
    }
    
    struct ggml_init_params params = {
        .mem_size   = compute_size,
        .mem_buffer = compute_mem,
        .no_alloc   = false
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        std::cerr << "Failed to initialize GGML context" << std::endl;
        free(compute_mem);
        return;
    }
    
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    
    // Get token embeddings
    struct ggml_tensor* token_embeddings = get_token_embeddings(model, ctx, tokens);
    if (!token_embeddings) {
        std::cerr << "Failed to get token embeddings" << std::endl;
        ggml_free(ctx);
        free(compute_mem);
        return;
    }
    
    // Forward through the transformer model
    struct ggml_tensor* output = build_transformer_inference_graph(model, ctx, token_embeddings, n_layers);
    if (!output) {
        std::cerr << "Failed to build transformer inference graph" << std::endl;
        ggml_free(ctx);
        free(compute_mem);
        return;
    }
    
    // Build the computation graph
    ggml_build_forward_expand(graph, output);
    
    // Compute the graph 
    int n_threads = 1;
    ggml_graph_compute_with_ctx(ctx, graph, n_threads);
    
    // Print output shape and first few values
    std::cout << "Output shape: [" << output->ne[0] << ", " << output->ne[1] << "]" << std::endl;
    std::cout << "First few output values:" << std::endl;
    
    float* output_data = (float*)output->data;
    int max_values = std::min(10, (int)output->ne[0]);
    
    for (int i = 0; i < max_values; i++) {
        std::cout << std::setw(12) << std::fixed << std::setprecision(6) << output_data[i];
        if ((i+1) % 5 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;
    
    // Clean up
    ggml_free(ctx);
    free(compute_mem);
}

int main() {
    ggml_time_init();
    
    // Load model and initialize tokenizer
    std::cout << "Loading model..." << std::endl;
    Tokenizer tokenizer("./gguf/1b-q8_0.gguf");
    
    // Tokenize input
    std::string text = "Hi Thomas";
    std::cout << "Input text: \"" << text << "\"" << std::endl;
    
    std::vector<int> tokens = tokenizer.tokenize(text);
    
    std::cout << "Tokenized to " << tokens.size() << " tokens:" << std::endl;
    for (size_t i = 0; i < tokens.size(); i++) {
        std::string token_text = "(unknown)";
        auto it = tokenizer.id_to_token.find(tokens[i]);
        if (it != tokenizer.id_to_token.end()) {
            token_text = it->second;
        }
        std::cout << "  " << tokens[i] << ": " << token_text << std::endl;
    }
    
    // Perform inference
    inference_tokens(tokenizer, tokens);
    
    return 0;
}
