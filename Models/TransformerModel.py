import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np

'''
Each model class should have the functions
.train(data_csv)
.generate(n=5000) 
.gen_to_csv()
'''

import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SequenceDataset(Dataset):
    def __init__(self, sequences, vocab):
        self.vocab = vocab
        self.data = [torch.tensor([self.vocab[char] for char in seq], dtype=torch.long) for seq in sequences]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx][:-1], self.data[idx][1:]  # Input and target

def collate_fn(batch):
    src, tgt = zip(*batch)
    src = pad_sequence(src, batch_first=True, padding_value=0)
    tgt = pad_sequence(tgt, batch_first=True, padding_value=0)
    return src, tgt

class TransformerModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'{self.device} enabled')

    def train(self, R, n_epochs=10):
        self.sequences =  [fields + ['<END>'] for fields in R.train_seq]
        self.init_dist, self.vocab = self.compute_state_information()

        dataset = SequenceDataset(self.sequences, self.vocab)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

        self.model = TransformerModule(len(self.vocab)).to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(n_epochs):
            total_loss = 0
            for src, tgt in dataloader:
                src, tgt = src.to(device), tgt.to(device)
                optimizer.zero_grad()
                output = self.model(src, src)
                loss = criterion(output.view(-1, len(self.vocab)), tgt.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

    def compute_state_information(self):
        """ Compute the empirical distribution of initial states. """
        first_states = [seq[0] for seq in self.sequences if seq[0] != "<END>"]
        unique_states, counts = np.unique(first_states, return_counts=True)
        probabilities = counts / counts.sum()
        unique_states = sorted(set(state for seq in self.sequences for state in seq), reverse=True)
        return dict(zip(unique_states, probabilities)), {state: i for i, state in enumerate(unique_states)}

    def generate(self, n=5000, max_len=40, temperature=1.5):
        idx_to_char = {i: ch for ch, i in self.vocab.items()}
        self.model.eval()
        self.generated_sequences = []
        for i in range(n):
            if i%100 == 0:
                print(f'{i}/{n}')
            start_seq = random.choices(list(self.init_dist.keys()), list(self.init_dist.values()))
            generated = [self.vocab[ch] for ch in start_seq]
            for _ in range(max_len - len(start_seq)):
                src = torch.tensor(generated).unsqueeze(0).to(device)
                output = self.model(src, src)
                probs = torch.softmax(output[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

                generated.append(next_token)
                if next_token == self.vocab["<END>"]:
                    break
            self.generated_sequences.append([idx_to_char[i] for i in generated])
    
    def gen_to_csv(self):
        rows = []
        for i,seq in enumerate(self.generated_sequences):
            if '<END>' in seq:
                rows.append(['TransformerGenerated', f'ID:{i}'] + seq[:seq.index('<END>')])
            else:
                rows.append(['TransformerGenerated', f'ID:{i}'] + seq)
        df = pd.DataFrame(rows)
        df.to_csv("CSVs/TransformerGeneratedHeaders.csv")

class TransformerModule(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=4, dim_feedforward=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers, dim_feedforward, batch_first=True)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
    
    def forward(self, src, tgt):
        src = self.embedding(src) * (self.d_model ** 0.5)
        tgt = self.embedding(tgt) * (self.d_model ** 0.5)
        output = self.transformer(src.permute(1, 0, 2), tgt.permute(1, 0, 2))
        return self.fc_out(output.permute(1, 0, 2))