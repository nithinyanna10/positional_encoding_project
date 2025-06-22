from model import TransformerClassifier
from dataset import DummySequenceDataset
from train import train
from visualize import visualize_positional_encoding

# Hyperparameters
vocab_size = 1000
d_model = 64
max_len = 10
nhead = 4
num_layers = 2
num_classes = 2

# Create dataset and model
dataset = DummySequenceDataset(num_samples=5000, seq_len=max_len, vocab_size=vocab_size, num_classes=num_classes)
model = TransformerClassifier(vocab_size, d_model, max_len, nhead, num_layers, num_classes)

# Train model
train(model, dataset, epochs=5)

# Visualize learned positional encoding
visualize_positional_encoding(model)
