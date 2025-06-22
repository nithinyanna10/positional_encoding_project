import torch
import torch.nn as nn

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, nhead, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = LearnablePositionalEncoding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)                 # (B, T, D)
        x = self.pos_encoder(x)                       # Add position
        x = x.permute(1, 0, 2)                        # (T, B, D) for transformer
        x = self.encoder(x)                           # (T, B, D)
        x = x.mean(dim=0)                             # (B, D)
        return self.classifier(x)                     # (B, num_classes)
