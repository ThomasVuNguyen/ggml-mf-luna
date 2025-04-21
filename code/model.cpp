#include "tokenizer.h"
#include "embedding.h"
#include <iostream>
#include <iomanip>
#include <fstream>

// Default model path locations to try if no path is provided
const char* DEFAULT_MODEL_PATHS[] = {
    "./gguf/1b-q8_0.gguf",      // Current directory
    "../gguf/1b-q8_0.gguf",     // One level up
    "../../gguf/1b-q8_0.gguf",  // Two levels up
    "/home/models/1b-q8_0.gguf" // Common models location
};
const int NUM_DEFAULT_PATHS = sizeof(DEFAULT_MODEL_PATHS) / sizeof(DEFAULT_MODEL_PATHS[0]);

int main(int argc, char* argv[]) {
    // Model path handling
    const char* model_path = nullptr;
    
    if (argc < 2) {
        // Try default paths if no model path is provided
        bool found_model = false;
        for (int i = 0; i < NUM_DEFAULT_PATHS; i++) {
            std::ifstream file(DEFAULT_MODEL_PATHS[i]);
            if (file.good()) {
                model_path = DEFAULT_MODEL_PATHS[i];
                found_model = true;
                std::cout << "Using default model path: " << model_path << std::endl;
                break;
            }
        }
        
        if (!found_model) {
            std::cerr << "Error: No model path provided and no default model found." << std::endl;
            std::cerr << "Usage: " << argv[0] << " <model_path> [text]" << std::endl;
            std::cerr << "  model_path: Path to the GGUF model file" << std::endl;
            std::cerr << "  text: Text to tokenize and embed (default: 'Thomas the Maker')" << std::endl;
            return 1;
        }
    } else {
        model_path = argv[1];
    }
    
    std::string text = (argc > 2) ? argv[2] : "Thomas the Maker";

    try {
        std::cout << "Loading tokenizer..." << std::endl;
        Tokenizer tokenizer(model_path);
        
        // Load the tokenizer model
        if (!tokenizer.load_model()) {
            std::cerr << "Failed to load tokenizer model" << std::endl;
            return 1;
        }
        
        // Print model info
        tokenizer.print_config();
        
        // Tokenize text with special tokens (add BOS token)
        std::vector<int> tokens = tokenizer.tokenize_with_special_tokens(text, true, false);
        
        // Set up embedding configuration
        EmbeddingConfig config = {0};
        
        // Initialize embedding weights by loading from the GGUF file
        EmbeddingWeights weights = {0};
        init_embedding_system(&config, &weights, model_path);
        
        // Save raw embeddings to compare with Python output
        std::ofstream json_out("output_cpp.json");
        json_out << "[";
        for (size_t i = 0; i < tokens.size(); i++) {
            if (i > 0) json_out << ", ";
            json_out << tokens[i];
        }
        json_out << "]\n";
        
        // Process each token and show raw embeddings
        for (int token : tokens) {
            // Get raw embedding for this token
            float* raw_embedding = (float*)malloc(config.dim * sizeof(float));
            get_token_embedding(raw_embedding, &weights, token, config.dim);
            
            // Display first few values of the raw embedding
            std::cout << "Embedding for token " << token << " (first 5 values): ";
            for (int i = 0; i < 5; i++) {
                std::cout << raw_embedding[i] << " ";
            }
            std::cout << std::endl;
            
            // Save to JSON file (raw)
            json_out << "Raw embedding (first 5 values): [";
            for (int i = 0; i < 5; i++) {
                if (i > 0) json_out << ", ";
                json_out << raw_embedding[i];
            }
            json_out << "]\n";
            
            // Get normalized embedding (like Python does)
            float* normalized_embedding = process_token(token, &weights, &config);
            
            // Save to JSON file (normalized)
            json_out << "Normalized embedding (first 5 values): [";
            for (int i = 0; i < 5; i++) {
                if (i > 0) json_out << ", ";
                json_out << normalized_embedding[i];
            }
            json_out << "]\n";
            
            free(raw_embedding);
            free(normalized_embedding);
        }
        
        json_out.close();
        
        // Clean up
        free_embedding_weights(&weights);
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}