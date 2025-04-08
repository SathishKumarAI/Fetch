# sentence_encoder.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List
import os

class SentenceEncoder(nn.Module):
    def __init__(self, model_name=None, pooling=None):
        super(SentenceEncoder, self).__init__()
        model_name = model_name or os.getenv("MODEL_NAME", "bert-base-uncased")
        self.pooling = pooling or os.getenv("POOLING_STRATEGY", "mean")
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling == 'cls':
            return outputs.last_hidden_state[:, 0]  # CLS token
        elif self.pooling == 'mean':
            return (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        elif self.pooling == 'max':
            return torch.max(outputs.last_hidden_state * attention_mask.unsqueeze(-1), dim=1).values
        else:
            raise ValueError("Invalid pooling type.")

    def encode(self, sentences: List[str], device='cpu'):
        self.eval()
        encoded_input = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoded_input['input_ids'].to(device)
        attention_mask = encoded_input['attention_mask'].to(device)
        with torch.no_grad():
            embeddings = self.forward(input_ids, attention_mask)
        return embeddings

# CLI entry point
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import numpy as np

    parser = argparse.ArgumentParser(description="Sentence Encoder CLI")
    parser.add_argument('--sentences', nargs='+', help='Sentences to encode', required=True)
    parser.add_argument('--pooling', type=str, default='mean', help='Pooling strategy: mean | cls | max')
    args = parser.parse_args()

    os.environ['POOLING_STRATEGY'] = args.pooling

    encoder = SentenceEncoder()
    embeddings = encoder.encode(args.sentences)

    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings.numpy())

    plt.figure(figsize=(8, 6))
    for i, sentence in enumerate(args.sentences):
        plt.scatter(reduced[i, 0], reduced[i, 1])
        plt.annotate(sentence, (reduced[i, 0], reduced[i, 1]))
    plt.title("t-SNE Visualization of Sentence Embeddings")
    plt.show()
