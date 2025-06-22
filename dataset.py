import torch
from torch.utils.data import Dataset

class DummySequenceDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=10, vocab_size=1000, num_classes=2):
        self.seq_len = seq_len
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]