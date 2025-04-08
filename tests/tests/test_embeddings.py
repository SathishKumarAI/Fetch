from model.sentence_encoder import SentenceEncoder

sentences = [
    "The weather is lovely today.",
    "Artificial Intelligence is transforming the world."
]

model = SentenceEncoder(pooling='mean')
embeddings = model.encode(sentences)
print("Embeddings shape:", embeddings.shape)
