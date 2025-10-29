import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.nn.functional as F

class GANModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'{self.device} enabled')

    def __str__(self):
        return f"Generator: {self.G}\nDiscriminator: {self.D}"

    def train(self, R, max_seq_length=60):
        self.max_length = max_seq_length
        sequences =  [fields + ['<END>']*(max_seq_length-len(fields)) for fields in R.train_seq]
        sizes = [len(s) for s in sequences]
        assert all([s == max_seq_length for s in sizes]), (min(sizes), max(sizes))

        vocab = list(R.count_dict.keys()) + ['<END>']
        self.vocab_size = len(vocab)

        # Mapping between tokens and integers
        self.char_to_idx = {char: i for i, char in enumerate(vocab)}
        self.idx_to_char = {i: char for char, i in self.char_to_idx.items()}

        # Prepare training data
        encoded_data = [self.encode_sequence(seq) for seq in sequences]
        train_data = torch.stack(encoded_data).to(self.device)

        # 🔹 Hyperparameters
        self.noise_dim = 16
        embed_dim = 16
        hidden_dim = 64
        batch_size = len(train_data)

        # Initialize networks
        self.G = Generator(self.noise_dim, embed_dim, hidden_dim, self.vocab_size, max_seq_length).to(self.device)
        self.D = Discriminator(embed_dim, hidden_dim, self.vocab_size, max_seq_length).to(self.device)

        # Loss & Optimizers
        criterion = nn.CrossEntropyLoss()
        optimizer_G = optim.Adam(self.G.parameters(), lr=0.005, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(self.D.parameters(), lr=0.002, betas=(0.5, 0.999))

        # 🔹 Training Loop
        num_epochs = 1000
        for epoch in range(num_epochs):
            for _ in range(3):  # Train Discriminator more frequently
                # Generate noise input for Generator
                noise = torch.randn(batch_size, max_seq_length, self.noise_dim).to(self.device)

                # 🔹 Train Discriminator
                optimizer_D.zero_grad()
                real_labels = (torch.rand(batch_size, 1) * 0.2 + 0.8).to(self.device)  # 0.8 - 1.0
                fake_labels = (torch.rand(batch_size, 1) * 0.2).to(self.device)  # 0.0 - 0.2


                real_preds = self.D(train_data)  
                real_loss = criterion(real_preds, real_labels)

                fake_sequences = self.G(noise).argmax(dim=-1).detach()  # Stop gradient flow
                fake_preds = self.D(fake_sequences)
                fake_loss = criterion(fake_preds, fake_labels)

                d_loss = real_loss + fake_loss
                d_loss.backward()
                optimizer_D.step()

            # 🔹 Train Generator
            optimizer_G.zero_grad()
            fake_preds = self.D(fake_sequences)
            real_features = self.D(train_data).mean()
            fake_features = self.D(fake_sequences).mean()
            g_loss = criterion(fake_preds, real_labels) + 0.1 * torch.abs(real_features - fake_features)

            g_loss.backward()
            optimizer_G.step()

            # Logging
            if epoch % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    def encode_sequence(self, sequence):
        """Encodes a sequence of chars into a fixed-length tensor."""
        encoded = [self.char_to_idx[char] for char in sequence]
        padded = encoded + [self.vocab_size] * (self.max_length - len(encoded))  
        return torch.tensor(padded, dtype=torch.long).to(self.device)
    
    def generate(self, N=5000, temperature=1):
        # 🔹 Generate New Sequences
        noise = torch.randn(N, self.max_length, self.noise_dim).to(self.device)
        #generated_sequences = self.G(noise).argmax(dim=-1).to(self.device)
        logits = self.G(noise)  # shape: [N, seq_len, vocab_size]
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)  # temperature > 1 = more random
        generated_sequences = torch.multinomial(probs.view(-1, probs.size(-1)), 1).squeeze(-1)
        generated_sequences = generated_sequences.view(N, self.max_length)

        result = []
        for seq in generated_sequences:
            decoded = [self.idx_to_char[idx.item()] for idx in seq]
            result.append(decoded)
        self.generated_sequences = result
        return result
    
    def gen_to_csv(self):
        rows = []
        for i,seq in enumerate(self.generated_sequences):
            if '<END>' in seq:
                rows.append(['GANGenerated', f'ID:{i}'] + seq[:seq.index('<END>')])
            else:
                rows.append(['GANGenerated', f'ID:{i}'] + seq)
        df = pd.DataFrame(rows)
        df.to_csv("CSVs/GANGeneratedHeaders.csv")

# 🔹 Generator Network (Improved)
class Generator(nn.Module):
    def __init__(self, noise_dim, embed_dim, hidden_dim, vocab_size, max_seq_length):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim)
        self.lstm = nn.LSTM(noise_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, noise):
        lstm_out, _ = self.lstm(noise)
        logits = self.fc(lstm_out)  # (batch, seq_length, vocab_size)
        return F.gumbel_softmax(logits, tau=0.5, hard=True)  # Use Gumbel-Softmax


# 🔹 Discriminator Network (Improved)
class Discriminator(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, max_seq_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3)  # Added extra layer + dropout
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.batch_norm(lstm_out[:, -1, :])  # Use last hidden state
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        return self.sigmoid(logits)
