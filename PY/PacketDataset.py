import torch
from torch.utils.data import Dataset, DataLoader

class PacketDataset(Dataset):
    def __init__(self, data, labels, protocols=None, pkt_len=32, num_pkts=256, max_chunks=4):
        """
        Preprocess the data so each sample has at most 256 packets,
        each up to 32bytes
        repeated if necessary, with a max of 4 samples per original input.
        """
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
        y = torch.tensor(label, dtype=torch.long)
        z = protocol
        return x, y, z

    def __len__(self):
        return len(self.samples)
