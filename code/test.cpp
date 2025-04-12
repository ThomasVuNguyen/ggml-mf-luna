#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include the main GGML headers
#include "ggml/include/ggml.h"
#include "ggml/include/ggml-alloc.h"

int main(int argc, char ** argv) {
    printf("GGML test program\n");
    
    // Initialize GGML
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,  // 16 MB
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize GGML context\n");
        return 1;
    }
    
    // Create a simple tensor
    struct ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);
    printf("Created tensor with shape [%d]\n", tensor->ne[0]);
    
    // Print GGML system information
    printf("GGML_MAX_DIMS: %d\n", GGML_MAX_DIMS);
    
    // Clean up
    ggml_free(ctx);
    
    printf("GGML test completed successfully\n");
    return 0;
}
