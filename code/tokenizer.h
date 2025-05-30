#pragma once

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

// Tokenizer class that loads a GGUF model and provides tokenization functionality
class Tokenizer {
private:
    struct gguf_context* ctx;
    struct ggml_context* ggml_ctx;
    std::string model_path;

public:
    std::map<int, std::string> id_to_token;
    std::map<std::string, int> token_to_id;

    Tokenizer(const std::string& model_path);
    ~Tokenizer();

    bool load_model();
    bool load_vocabulary();
    void print_config();
    void print_vocabulary_sample(int n = 20);
    std::vector<float> get_embeddings(int token_id) const;
    std::vector<float> get_embeddings(const std::string& text) const;
    std::string find_closest_token(const std::vector<float>& embedding);
    std::vector<int> tokenize(const std::string& text) const;
    int64_t get_hidden_size();
    size_t get_vocab_size() const { return id_to_token.size(); }
    struct gguf_context* get_gguf_context() const { return ctx; }
    struct ggml_context* get_ggml_context() const { return ggml_ctx; }
};

// Helper function to print token embeddings
void print_token_embedding(const Tokenizer& tokenizer, int token_id); 