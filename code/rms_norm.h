#pragma once

#include "tokenizer.h"
#include <vector>
#include <string>

// Get the epsilon value from model config
float get_rms_epsilon(const Tokenizer& tokenizer);

// Get RMS normalization weights for a specific layer
std::vector<float> get_layer_rms_weights(const Tokenizer& tokenizer, const std::string& layer_name);

// Apply RMS normalization using model weights
std::vector<float> compute_rms_norm(const Tokenizer& tokenizer, 
                                    const std::vector<float>& input, 
                                    const std::string& layer_name);

// Print the first n values of a vector in a tabular format
void print_vector_head(const std::vector<float>& vec, const std::string& label, int n = 5); 