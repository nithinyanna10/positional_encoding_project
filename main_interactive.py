import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer

# ---- Load your learnable positional encoding ---- #
from src.model import LearnablePositionalEncoding

# ---- Settings ---- #
embedding_dim = 100
max_len = 20  # max number of tokens we will visualize
sentence = input("Enter a sentence: ").strip().lower()

# ---- Tokenization ---- #
tokenizer = get_tokenizer("basic_english")
tokens = tokenizer(sentence)[:max_len]  # truncate if too long
print("🔤 Tokens:", tokens)

# ---- Load GloVe ---- #
print("🔄 Loading GloVe...")
vocab = GloVe(name="6B", dim=embedding_dim)

# ---- Get embeddings ---- #
token_embeddings = []
for token in tokens:
    try:
        token_embeddings.append(vocab[token])
    except KeyError:
        print(f"⚠️  '{token}' not found in GloVe. Replacing with zeros.")
        token_embeddings.append(torch.zeros(embedding_dim))

token_embeddings = torch.stack(token_embeddings).unsqueeze(0)  # [1, T, D]

# ---- Apply learnable positional encoding ---- #
pos_encoder = LearnablePositionalEncoding(max_len=max_len, d_model=embedding_dim)
model_input = pos_encoder(token_embeddings).squeeze(0)  # [T, D]

# ---- Visualization ---- #
sns.set(style="whitegrid")
plt.figure(figsize=(14, 6))
sns.heatmap(model_input.detach().numpy(), cmap="viridis", xticklabels=False, yticklabels=tokens)
plt.title("💡 Token Embeddings After Adding Learnable Positional Encodings")
plt.xlabel("Embedding Dimension")
plt.ylabel("Input Tokens")
plt.tight_layout()
plt.show()
