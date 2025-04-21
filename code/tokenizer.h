#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <vector>
#include <string>
#include <map>

// Forward declarations for GGML/GGUF types
struct gguf_context;
struct ggml_context;

// Special token constants - may vary by model
#define TOKEN_BOS 1  // Beginning of sequence token ID (might need to be adjusted for your model)
#define TOKEN_EOS 2  // End of sequence token ID (might need to be adjusted for your model)

// Simple tokenizer that loads a GGUF model and provides tokenization functionality
class Tokenizer {
private:
    struct gguf_context* ctx;
    struct ggml_context* ggml_ctx;
    std::string model_path;
    std::map<int, std::string> id_to_token;
    std::map<std::string, int> token_to_id;
    
    // Get the index of special tokens from the vocabulary if available
    void find_special_tokens();

public:
    int bos_token_id;  // Beginning of sequence token ID
    int eos_token_id;  // End of sequence token ID
    
    Tokenizer(const std::string& model_path);
    ~Tokenizer();
    
    bool load_model();
    bool load_vocabulary();
    void print_config();
    void print_vocabulary_sample(int n = 20);
    std::vector<float> get_embeddings(int token_id);
    std::vector<float> get_embeddings(const std::string& text);
    std::string find_closest_token(const std::vector<float>& embedding);
    
    // Tokenize without special tokens
    std::vector<int> tokenize(const std::string& text);
    
    // Tokenize with special tokens
    std::vector<int> tokenize_with_special_tokens(const std::string& text, bool add_bos = true, bool add_eos = false);
};

#endif // TOKENIZER_H 