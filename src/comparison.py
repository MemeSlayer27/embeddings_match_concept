import numpy as np
import ollama

def get_embedding(text):
    """Get embedding using Ollama Python library."""
    response = ollama.embeddings(
        model='mxbai-embed-large',
        prompt=text
    )
    return np.array(response['embedding'])

def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Avoid division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    
    return dot_product / (norm_v1 * norm_v2)

def compare_texts(text1, text2):
    """Compare two texts using embeddings and cosine similarity."""
    # Get embeddings
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    
    # Calculate similarity
    similarity = cosine_similarity(embedding1, embedding2)
    
    # Print results
    print("\nText Comparison Results:")
    print("-" * 50)
    print(f"Text 1: {text1[:50]}...")
    print(f"Text 2: {text2[:50]}...")
    print(f"Cosine Similarity: {similarity:.4f}")
    
    # Provide a qualitative interpretation
    if similarity > 0.95:
        interpretation = "Nearly identical meaning"
    elif similarity > 0.8:
        interpretation = "Very similar meaning"
    elif similarity > 0.6:
        interpretation = "Moderately similar meaning"
    elif similarity > 0.4:
        interpretation = "Somewhat similar meaning"
    else:
        interpretation = "Different meanings"
    
    print(f"Interpretation: {interpretation}")
    
    return similarity

def main():
    print("\nEnter two texts to compare:")
    text1 = input("Text 1: ")
    text2 = input("Text 2: ")
    
    try:
        compare_texts(text1, text2)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def get_aligned_political_statements():
    """
    Returns political statements with clear ideological alignment patterns.
    Each category has statements ranging from strongly progressive to strongly conservative,
    plus some moderate/compromise positions.
    """
    statements = {
        # ECONOMIC POLICY
        # Progressive
        "econ_prog_strong": "We need high taxes on the wealthy, universal basic income, and strong market regulations",
        "econ_prog_mod": "Progressive taxation and reasonable market oversight protect working families",
        # Moderate
        "econ_mod_1": "A mixed economy balancing free markets with necessary regulations works best",
        "econ_mod_2": "We should combine market efficiency with basic social protections",
        # Conservative
        "econ_cons_mod": "Lower taxes and reduced regulation promote economic growth and job creation",
        "econ_cons_strong": "Free market capitalism with minimal government intervention maximizes prosperity",

        # HEALTHCARE
        # Progressive
        "health_prog_strong": "We need Medicare for All with no private insurance involvement",
        "health_prog_mod": "Universal healthcare should be a basic right with optional private insurance",
        # Moderate
        "health_mod_1": "A public option alongside private insurance gives people choice",
        "health_mod_2": "Healthcare reform should expand access while preserving private insurance",
        # Conservative
        "health_cons_mod": "Market-based healthcare with limited government involvement works best",
        "health_cons_strong": "Healthcare should be fully private with no government interference",

        # CLIMATE/ENVIRONMENT
        # Progressive
        "climate_prog_strong": "We need immediate radical action including carbon bans and strict regulations",
        "climate_prog_mod": "Strong carbon pricing and renewable energy investments are essential",
        # Moderate
        "climate_mod_1": "Market-based solutions like carbon taxes can fight climate change",
        "climate_mod_2": "Balance environmental protection with economic growth",
        # Conservative
        "climate_cons_mod": "Environmental protection should be handled by states and markets",
        "climate_cons_strong": "Climate regulations hurt business and aren't necessary",

        # IMMIGRATION
        # Progressive
        "immig_prog_strong": "Open borders with a path to citizenship for all immigrants",
        "immig_prog_mod": "Welcome immigrants while maintaining reasonable entry processes",
        # Moderate
        "immig_mod_1": "Support legal immigration while securing borders",
        "immig_mod_2": "Reform immigration with both compassion and control",
        # Conservative
        "immig_cons_mod": "Merit-based immigration with strong border security",
        "immig_cons_strong": "Strict immigration controls and border enforcement only",

        # SOCIAL ISSUES
        # Progressive
        "social_prog_strong": "Actively promote progressive social change and equity in all institutions",
        "social_prog_mod": "Support social progress while respecting diverse viewpoints",
        # Moderate
        "social_mod_1": "Balance traditional values with changing social norms",
        "social_mod_2": "Let social change happen gradually through consensus",
        # Conservative
        "social_cons_mod": "Preserve traditional values while accepting some changes",
        "social_cons_strong": "Strictly maintain traditional social and cultural values",

        # GUN POLICY
        # Progressive
        "guns_prog_strong": "Ban most firearms and implement strict gun control nationwide",
        "guns_prog_mod": "Stronger gun control including assault weapon restrictions",
        # Moderate
        "guns_mod_1": "Universal background checks while protecting basic gun rights",
        "guns_mod_2": "Balance public safety with constitutional rights",
        # Conservative
        "guns_cons_mod": "Protect gun rights with limited safety regulations",
        "guns_cons_strong": "No restrictions on Second Amendment rights whatsoever",

        # EDUCATION
        # Progressive
        "edu_prog_strong": "Free public education through college with federal standards",
        "edu_prog_mod": "Increase education funding and expand public programs",
        # Moderate
        "edu_mod_1": "Improve public schools while allowing school choice",
        "edu_mod_2": "Balance public education with alternative options",
        # Conservative
        "edu_cons_mod": "School choice and local control of education",
        "edu_cons_strong": "Privatize education through vouchers and choice",

        # FOREIGN POLICY
        # Progressive
        "foreign_prog_strong": "Focus on international cooperation and reduce military spending",
        "foreign_prog_mod": "Emphasize diplomacy while maintaining defensive capability",
        # Moderate
        "foreign_mod_1": "Balance diplomatic and military approaches",
        "foreign_mod_2": "Selective engagement based on clear national interests",
        # Conservative
        "foreign_cons_mod": "Peace through strength and strong military presence",
        "foreign_cons_strong": "Military strength and aggressive foreign policy"
    }
    return statements

def create_similarity_matrix(statements):
    """Create a similarity matrix for all statements."""
    n = len(statements)
    matrix = np.zeros((n, n))
    
    # Get all embeddings first to avoid recomputing
    embeddings = {}
    print("\nGetting embeddings for all statements...")
    for i, (key, text) in enumerate(statements.items()):
        print(f"Processing {i+1}/{n}: {key}")
        embeddings[key] = get_embedding(text)
    
    # Compute similarity matrix
    print("\nComputing similarity matrix...")
    keys = list(statements.keys())
    for i, key1 in enumerate(keys):
        for j, key2 in enumerate(keys):
            matrix[i, j] = cosine_similarity(embeddings[key1], embeddings[key2])
    
    return matrix, keys

def plot_similarity_matrix(matrix, labels):
    """Plot a heatmap of the similarity matrix with improved readability."""
    plt.figure(figsize=(20, 16))
    
    # Create more readable labels
    readable_labels = []
    for label in labels:
        # Split label into parts
        parts = label.split('_')
        # Format based on position
        if 'prog' in label:
            position = 'Progressive'
        elif 'cons' in label:
            position = 'Conservative'
        elif 'mod' in label:
            position = 'Moderate'
        
        # Add strength if specified
        if 'strong' in label:
            position += '+'
        
        # Get category
        category = parts[0].capitalize()
        
        readable_labels.append(f"{category}\n{position}")
    
    # Create heatmap
    sns.heatmap(matrix, 
                xticklabels=readable_labels,
                yticklabels=readable_labels,
                cmap='RdYlBu_r',
                vmin=0,
                vmax=1,
                center=0.5)
    
    plt.title("Political Statement Similarity Matrix", pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def analyze_ideological_patterns(matrix, labels):
    """Analyze patterns in the similarity matrix focusing on ideological alignment."""
    print("\nIdeological Pattern Analysis:")
    print("-" * 50)
    
    # Group indices by ideology
    prog_indices = [i for i, label in enumerate(labels) if 'prog' in label]
    cons_indices = [i for i, label in enumerate(labels) if 'cons' in label]
    mod_indices = [i for i, label in enumerate(labels) if 'mod' in label]
    
    # Calculate average similarities within and between groups
    def get_group_similarity(indices1, indices2):
        similarities = []
        for i in indices1:
            for j in indices2:
                if i != j:  # Exclude self-similarity
                    similarities.append(matrix[i,j])
        return np.mean(similarities) if similarities else 0
    
    print("\nAverage Similarities Between Ideological Groups:")
    print(f"Progressive <-> Progressive: {get_group_similarity(prog_indices, prog_indices):.3f}")
    print(f"Conservative <-> Conservative: {get_group_similarity(cons_indices, cons_indices):.3f}")
    print(f"Moderate <-> Moderate: {get_group_similarity(mod_indices, mod_indices):.3f}")
    print(f"Progressive <-> Conservative: {get_group_similarity(prog_indices, cons_indices):.3f}")
    print(f"Progressive <-> Moderate: {get_group_similarity(prog_indices, mod_indices):.3f}")
    print(f"Conservative <-> Moderate: {get_group_similarity(cons_indices, mod_indices):.3f}")
    
    # Find strongest cross-category alignments
    print("\nStrongest Cross-Category Ideological Alignments:")
    alignments = []
    for i, label1 in enumerate(labels):
        category1 = label1.split('_')[0]
        for j, label2 in enumerate(labels):
            category2 = label2.split('_')[0]
            if category1 != category2:
                alignments.append((label1, label2, matrix[i,j]))
    
    alignments.sort(key=lambda x: x[2], reverse=True)
    for label1, label2, sim in alignments[:5]:
        print(f"{label1} <-> {label2}: {sim:.3f}")

def main():
    # Get statements
    statements = get_aligned_political_statements()
    
    # Create and plot similarity matrix
    matrix, labels = create_similarity_matrix(statements)
    plot_similarity_matrix(matrix, labels)
    
    # Analyze patterns
    analyze_ideological_patterns(matrix, labels)

if __name__ == "__main__":
    main()