import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import pandas as pd

# Define a simple Transformer-based model
class TransformerDiffusion(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=128, num_layers=4, nhead=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, d_model)  # +1 for padding
        self.positional_encoding = nn.Parameter(torch.randn(max_len, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True), num_layers
        )

        self.output_layer = nn.Linear(d_model, vocab_size)  # Predict next token probabilities
    
    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[: x.shape[0], :].unsqueeze(0)

        x = self.transformer(x)
        return self.output_layer(x)

# Define diffusion forward and backward process
class DiffusionModel:
    def __init__(self, model, timesteps=100, beta_start=0.0001, beta_end=0.02):
        self.model = model
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def train(self, file, max_seq_length=50):
        scapy_data = pd.read_csv(file)
        sequences = []
        vocab = ['<END>']
        self.max_length = max_seq_length
        for _, row in enumerate(scapy_data.iterrows()):
            fields = []
            try: 
                i = 3
                while pd.notna(row[1].iloc[i]) and i < len(row[1])-1:
                    field = row[1].iloc[i].split('(')[0]
                    fields.append(field)
                    if field not in vocab:
                        vocab.append(field)
                    i += 1
                sequences.append(fields[:50] + ['<END>']*(50-len(fields)))
            except Exception as e:
                print(e, row)
        self.sequences = sequences
        self.vocab_size = len(vocab)
    
    def forward_process(self, x, t):
        noise = torch.randint(0, self.model.output_layer.out_features, x.shape, device=x.device)
        noisy_x = torch.where(torch.rand_like(x.float()) < self.alpha_bars[t].to(x.device), x, noise)
        return noisy_x
    
    def reverse_process(self, x, seq_len):
        for t in reversed(range(self.timesteps)):
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            x = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(x.shape)
        return x[:, :seq_len]
    
if __name__ == '__main__':

    # Example usage
    sequences = [['a'], ['a','b','a'], ['a', 'b','a','d'], ['c', 'd']]
    vocab = list(set(token for seq in sequences for token in seq))
    dataset = SequenceDataset(sequences, vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = TransformerDiffusion(len(vocab), dataset.max_len)
    diffusion = DiffusionModel(model)

    # Training loop (simplified)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 10
    for epoch in range(epochs):
        for batch, seq_len in dataloader:
            t = torch.randint(0, diffusion.timesteps, (batch.size(0),))
            noisy_x = diffusion.forward_process(batch, t)
            logits = model(noisy_x)
            loss = loss_fn(logits.view(-1, logits.shape[-1]), batch.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")
