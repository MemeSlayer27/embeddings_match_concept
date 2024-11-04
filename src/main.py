import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import ollama

def get_embedding(text):
    """Get embedding using Ollama Python library."""
    response = ollama.embeddings(
        model='mxbai-embed-large',
        prompt=text
    )
    return np.array(response['embedding'])

def normalize_vector(vector):
    """Normalize vector to unit length."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def plot_vectors(component_vectors, aggregated_vector, labels):
    """Plot vectors in 2D using t-SNE."""
    # Combine component vectors with aggregated vector for t-SNE
    if aggregated_vector is not None:
        all_vectors = np.vstack([component_vectors, aggregated_vector.reshape(1, -1)])
    else:
        all_vectors = component_vectors
    
    # Initialize and fit t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, max(2, len(all_vectors) - 1)),  # Adjust perplexity based on number of points
        n_iter=1000,
        random_state=42
    )
    
    # Transform all vectors
    all_points = tsne.fit_transform(all_vectors)
    
    # Split points back into components and aggregated
    if aggregated_vector is not None:
        components_2d = all_points[:-1]
        aggregated_2d = all_points[-1:]
    else:
        components_2d = all_points
        aggregated_2d = None
    
    plt.figure(figsize=(10, 8))
    
    # Plot component vectors
    plt.scatter(components_2d[:, 0], components_2d[:, 1], c='blue', label='Components')
    
    # Plot aggregated vector if it exists
    if aggregated_vector is not None:
        plt.scatter(aggregated_2d[:, 0], aggregated_2d[:, 1], c='red', label='Aggregated')
    
    # Add labels
    for i, label in enumerate(labels):
        plt.annotate(label[:20] + "..." if len(label) > 20 else label,
                    (all_points[i, 0], all_points[i, 1]))
    
    plt.title("Embedding Vector Space Visualization (t-SNE)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.draw()
    plt.pause(0.1)

def aggregate_vectors(vectors):
    """Aggregate vectors using mean and normalize."""
    if not vectors:
        return None
    # Calculate mean of all vectors
    aggregated = np.mean(vectors, axis=0)
    # Normalize the result
    return normalize_vector(aggregated)

def main():
    vectors = []
    prompts = []
    
    print("\nEnter text to embed (type 'quit' to exit):")
    print("Note: First embedding might take longer as the model loads.\n")
    
    while True:
        user_input = input("> ")
        if user_input.lower() == 'quit':
            break
            
        try:
            # Get embedding for current input
            current_embedding = get_embedding(user_input)
            # Normalize the current embedding
            current_embedding = normalize_vector(current_embedding)
            vectors.append(current_embedding)
            prompts.append(user_input)
            
            # Only proceed with visualization if we have at least 2 points
            if len(vectors) >= 2:
                # Calculate aggregated vector
                aggregated_vector = aggregate_vectors(vectors)
                
                # Plot - now passing component vectors and aggregated vector separately
                plot_vectors(np.array(vectors), aggregated_vector, prompts + ["Aggregated"])
                
                print(f"\nProcessed {len(vectors)} inputs. Vector dimension: {current_embedding.shape[0]}")
            else:
                print("\nNeed at least 2 points for t-SNE visualization. Please add more points.")
            
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    plt.ion()  # Enable interactive plotting
    main()