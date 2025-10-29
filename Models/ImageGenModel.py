import torch
import torch.nn as nn
import torch.nn.functional as F
from PY.PacketDataset import PacketDataset
from torch.nn.utils.rnn import pad_sequence

class IndexedPacketDataset(PacketDataset):
    def __init__(self, data, labels, FIELD_TYPE_VOCAB, protocols=None, pkt_len=25632, num_pkts=256, max_chunks=4):
        """
        Preprocess the data so each sample has at most 256 packets,
        each up to 32
        repeated if necessary, with a max of 4 samples per original input.
        """
        self.field_type_to_idx = {ftype: i+1 for i, ftype in enumerate(FIELD_TYPE_VOCAB)} | {"None" : 0}
        self.samples = []
        self.pkt_len = pkt_len
        self.num_pkts = num_pkts

        for i, pkt_list in enumerate(data):
            label = labels[i]
            protocol = protocols[i] if protocols else 'unknown'

            chunks = [pkt_list[j:j+num_pkts] for j in range(0, len(pkt_list), num_pkts)]
            # Limit to max_chunks
            chunks = chunks[:max_chunks]

            for chunk in chunks:
                # If fewer than seq_len, repeat packets
                repeated = (chunk * ((num_pkts // len(chunk)) + 1))[:num_pkts]
                self.samples.append((repeated, label, protocol))


    def pad_packet(self, pkt_bytes):
        pkt_list = list(pkt_bytes)
        pkt_list = pkt_list[:self.pkt_len]
        pad_size = self.pkt_len - len(pkt_list)
        if pad_size > 0:
            pkt_list += [-255] * pad_size
        return pkt_list

    def __getitem__(self, idx):
        sample, label, protocol = self.samples[idx]

        padded_packets = [self.pad_packet(pkt) for pkt in sample]
        x = torch.tensor(padded_packets, dtype=torch.float32)  # (num_pkts, pkt_len)
        field_ids = torch.tensor([self.field_type_to_idx[t] for t in label], dtype=torch.long)
        return field_ids, x.unsqueeze(0), protocol  # (L,), (1, 256, 256)

    def __len__(self):
        return len(self.samples)




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

class CNNPacketImageGenerator(nn.Module):
    def __init__(self, field_vocab_size, field_emb_dim=64, cond_dim=128):
        super().__init__()
        self.field_emb = nn.Embedding(field_vocab_size, field_emb_dim)
        self.pos_enc = PositionalEncoding(field_emb_dim)
        self.project = nn.Sequential(
            nn.Linear(field_emb_dim, cond_dim),
            nn.ReLU(),
            nn.Linear(cond_dim, 256 * 4 * 1),  # change to 4x1 feature map
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(4,1), stride=(2,1), padding=(1,0)),  # (8x1)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(4,1), stride=(2,1), padding=(1,0)),   # (16x1)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(4,1), stride=(2,1), padding=(1,0)),    # (32x1)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(8,32), stride=(8,32)),                  # (256x32)
        )

    def forward(self, field_ids):
        if torch.any(field_ids < 0) or torch.any(field_ids >= self.field_emb.num_embeddings):
            raise ValueError(f"field_ids contains out-of-range indices! Max allowed: {self.field_emb.num_embeddings - 1}")

        emb = self.field_emb(field_ids)  # (B, L, D)
        emb = self.pos_enc(emb)          # (B, L, D)
        pooled = emb.mean(dim=1)         # (B, D)

        x = self.project(pooled)         # (B, 256*4*1)
        x = x.view(-1, 256, 4, 1)       # (B, 256, 4, 1)
        out = self.decoder(x)            # (B, 1, 256, 32)

        out = torch.clamp(out, -255, 255) #need padding
        return out


def generate_packet_image(field_type_sequence, model, field_type_to_idx, device='cuda'):
    model.eval()
    with torch.no_grad():
        field_ids = torch.tensor(
            [[field_type_to_idx[t] for t in field_type_sequence]], dtype=torch.long
        ).to(device)
        gen_image = model(field_ids)
    return gen_image.squeeze(0).squeeze(0).cpu().numpy()  # (256, 256)
