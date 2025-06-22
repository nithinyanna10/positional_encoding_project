import matplotlib.pyplot as plt
import seaborn as sns

def visualize_positional_encoding(model):
    pos_weights = model.pos_encoder.pos_embedding.squeeze(0).detach().cpu().numpy()
    plt.figure(figsize=(12, 6))
    sns.heatmap(pos_weights, cmap="viridis")
    plt.title("Learned Positional Embeddings")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position")
    plt.show()
