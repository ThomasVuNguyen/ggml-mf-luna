import llama_cpp

llm = llama_cpp.Llama(model_path="../gguf/1b-q8_0.gguf", embedding=True, verbose=False)
prompt = "Thomas"
info = llm.create_embedding(prompt)

tokens = llm.tokenize(prompt.encode())
print(tokens)

embeddings = info['data'][0]['embedding']

# or create multiple embeddings at once

# embeddings = llm.create_embedding(["Thomas the Maker"])

for embedding in embeddings:
    print("Embedding (first 5 values):", embedding[:5])
    print("Embedding size:", len(embedding))


