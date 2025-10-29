import torch
import torch.nn as nn
import torch.nn.functional as F

class PacketEncoder(nn.Module):
    """Encodes a single packet using 1D CNN layers."""
    def __init__(self, packet_length, embedding_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(embedding_dim),  # fixed output size
        )
    
    def forward(self, x):
        # x: (batch, seq_len=1000, packet_length)
        batch, seq_len, packet_length = x.size()
        x = x.view(-1, 1, packet_length)  # (batch*seq_len, 1, packet_length)
        x = self.encoder(x)  # (batch*seq_len, 64, embedding_dim)
        x = x.view(batch, seq_len, -1)  # (batch, seq_len, embedding_dim*channels)
        return x

class TrafficClassifier(nn.Module):
    def __init__(self, packet_length, embedding_dim=16, num_heads=4, transformer_layers=2):
        super().__init__()
        self.packet_encoder = PacketEncoder(packet_length, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim*64,  # match CNN output dim
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim*64, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # real/fake
        )
    
    def forward(self, x):
        # x: (batch, seq_len=1000, packet_length)
        x = self.packet_encoder(x)  # (batch, seq_len, embedding_dim*channels)
        x = x.permute(1, 0, 2)  # (seq_len, batch, dim) for transformer
        x = self.transformer(x)
        x = x.mean(dim=0)  # (batch, dim): global average pooling over seq_len
        out = self.classifier(x)
        return out

# Example usage:
if __name__ == "__main__":
    batch_size = 8
    seq_len = 1000
    packet_length = 128  # example: depends on protocol

    model = TrafficClassifier(packet_length)
    dummy_input = torch.randn(batch_size, seq_len, packet_length)
    logits = model(dummy_input)
    print("Output shape:", logits.shape)  # (batch_size, 2)
