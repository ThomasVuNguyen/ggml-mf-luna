#pragma once

#include "tokenizer.h"

void extract_model_config(const Tokenizer& tokenizer);
void print_model_weights(const Tokenizer& tokenizer);
void print_token_embedding(const Tokenizer& tokenizer, int token_id); 