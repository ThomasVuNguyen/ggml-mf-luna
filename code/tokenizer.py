#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import argparse

def print_usage():
    print("Usage: python tokenizer.py [--model MODEL_NAME] [text]")
    print("  MODEL_NAME: Name or path of the model to use (default: meta-llama/Llama-3.2-1B)")
    # print("  text: Text to tokenize (default: \"Hello world thomas\")")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Tokenize text using a transformer model")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B", help="Model name or path")
    parser.add_argument("text", nargs="?", default="Make America great again, and again, never yield.", help="Text to tokenize")
    args = parser.parse_args()
    
    model_name = args.model
    sample_text = args.text
    
    print(f"Loading tokenizer for {model_name}...")
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # Minimal model info
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Print vocabulary sample
    print("\n--- Vocabulary Sample ---")
    for i in range(20):  # Show first 20 tokens
        try:
            token_repr = tokenizer.convert_ids_to_tokens(i)
            print(f"ID {i:5}: {token_repr}")
        except:
            print(f"ID {i:5}: <error decoding token>")
    print("------------------------")
    
    print(f"\nTokenizing: \"{sample_text}\"")
    
    # Tokenize the text
    tokens = tokenizer.encode(sample_text, return_tensors="pt")
    
    # Show token IDs
    print(f"Token IDs: {tokens[0].tolist()}")
    print(f"Total tokens: {len(tokens[0])}")
    
    # Add a note about tokenization
    print("\nNote on tokenization:")
    print("- In most modern tokenizers, spaces are represented with special tokens or prefixes")
    print("- For example, many tokenizers use 'Ġ' (Unicode 0x0120) to mark tokens that follow a space")
    print("- Capitalization affects tokenization - 'Hello' and 'hello' are often different tokens")
    print("- Different models use different tokenization schemes")
    
    # Show token details with special representation
    print("\n--- Token Details ---")
    for token_id in tokens[0].tolist():
        # Get the token representation (showing special characters)
        token_repr = tokenizer.convert_ids_to_tokens(token_id)
        # Convert token ID back to text for display
        token_text = tokenizer.decode([token_id])
        print(f"ID {token_id:5}: '{token_text}' (repr: '{token_repr}')")
        
    # Verify by decoding back to text
    decoded_text = tokenizer.decode(tokens[0])
    print(f"\nDecoded: \"{decoded_text}\"")

if __name__ == "__main__":
    main()
