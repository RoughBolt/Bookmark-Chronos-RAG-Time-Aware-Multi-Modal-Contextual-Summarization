import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.vector_store import create_chroma_collection

def visualize_memories():
    print("Fetching memories from ChromaDB...")
    client, collection = create_chroma_collection()
    
    # Extract all data
    data = collection.get(include=["embeddings", "metadatas", "documents"])
    
    embeddings = data.get("embeddings", [])
    metadatas = data.get("metadatas", [])
    documents = data.get("documents", [])
    
    if embeddings is None or len(embeddings) == 0:
        print("No embeddings found in the database. Run the main processing pipeline first!")
        return

    # Convert to numpy array
    X = np.array(embeddings)
    
    print(f"Loaded {X.shape[0]} memories with {X.shape[1]} dimensions.")
    print("Applying t-SNE Dimensionality Reduction (768D -> 2D)... This may take a few seconds.")
    
    # Apply t-SNE
    # Perplexity controls local vs global geometry. Must be less than number of samples.
    perplexity = min(30, max(5, int(len(X) / 3)))
    
    # init="pca" stabilizes the starting map rather than taking a random guess
    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=42)
    X_2d = tsne.fit_transform(X)
    
    # Plotting setup
    plt.figure(figsize=(12, 8))
    
    # Group by tags for colors so we can see how EVENT vs COMBAT clusters
    tags = [meta.get("tag", "UNKNOWN") if meta else "UNKNOWN" for meta in metadatas]
    unique_tags = list(set(tags))
    
    # Fallback colormap logic
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in np.linspace(0, 1, len(unique_tags))]
    
    for i, tag in enumerate(unique_tags):
        idx = [j for j, t in enumerate(tags) if t == tag]
        plt.scatter(
            X_2d[idx, 0], X_2d[idx, 1], 
            color=colors[i], label=tag, alpha=0.7, edgecolors='w', s=100
        )
        
    plt.title("Semantic Brain Map: Time-Aware Multi-Modal Memory (t-SNE 2D Projection)")
    plt.xlabel("t-SNE Latent Dimension 1 (Abstract Semantic Trait)")
    plt.ylabel("t-SNE Latent Dimension 2 (Abstract Semantic Trait)")
    plt.legend(title="Memory Type")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    print("Mapping complete! Rendering the memory graph...")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_memories()
