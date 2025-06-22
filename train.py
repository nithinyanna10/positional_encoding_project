import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam

def train(model, dataset, epochs=10, batch_size=32, lr=1e-3):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - Loss: {total_loss / len(loader):.4f}")
