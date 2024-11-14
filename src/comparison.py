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
    "econ_prog_strong_2": "Nationalize key industries and implement wealth caps",
    "econ_prog_strong_3": "Break up all large corporations and redistribute wealth",
    "econ_prog_strong_4": "Mandate employee ownership in all major companies",
    "econ_prog_mod_2": "Expand social programs through corporate tax reform",
    "econ_prog_mod_3": "Strengthen unions and mandate profit sharing",
    "econ_prog_mod_4": "Implement guaranteed public sector jobs",
    # Moderate
    "econ_mod_3": "Support both small business and worker protections",
    "econ_mod_4": "Targeted incentives for economic development",
    "econ_mod_5": "Promote public-private partnerships for growth",
    # Conservative
    "econ_cons_mod_2": "Simplify tax code and reduce business regulations",
    "econ_cons_mod_3": "Promote free trade with limited protections",
    "econ_cons_mod_4": "Focus on debt reduction and fiscal restraint",
    "econ_cons_strong_2": "Eliminate most business regulations entirely",
    "econ_cons_strong_3": "Privatize all government services possible",
    "econ_cons_strong_4": "Flat tax rate for all income levels",

    # HEALTHCARE
    # Progressive
    "health_prog_strong_2": "Nationalize all hospitals and medical facilities",
    "health_prog_strong_3": "Free universal mental and dental coverage",
    "health_prog_strong_4": "Government control of pharmaceutical industry",
    "health_prog_mod_2": "Expand Medicare to age 50 and above",
    "health_prog_mod_3": "Universal catastrophic coverage with subsidies",
    "health_prog_mod_4": "Mandatory employer-provided health benefits",
    # Moderate
    "health_mod_3": "Reform drug pricing while preserving innovation",
    "health_mod_4": "Increase healthcare price transparency",
    "health_mod_5": "Promote preventive care and wellness programs",
    # Conservative
    "health_cons_mod_2": "Health savings accounts with tax benefits",
    "health_cons_mod_3": "Interstate insurance competition",
    "health_cons_mod_4": "Tort reform to reduce healthcare costs",
    "health_cons_strong_2": "Cash-only medical practice model",
    "health_cons_strong_3": "Eliminate all healthcare mandates",
    "health_cons_strong_4": "Fully privatize Medicare and Medicaid",

    # CLIMATE/ENVIRONMENT
    # Progressive
    "climate_prog_strong_2": "Ban all fossil fuel extraction immediately",
    "climate_prog_strong_3": "Mandatory transition to plant-based diet",
    "climate_prog_strong_4": "Zero-emission requirements for all industries",
    "climate_prog_mod_2": "Green infrastructure investment program",
    "climate_prog_mod_3": "Phase out gas vehicles by 2030",
    "climate_prog_mod_4": "Mandate solar panels on new construction",
    # Moderate
    "climate_mod_3": "Invest in nuclear and renewable energy",
    "climate_mod_4": "Incentivize corporate sustainability",
    "climate_mod_5": "Support clean energy research and development",
    # Conservative
    "climate_cons_mod_2": "Focus on conservation over regulation",
    "climate_cons_mod_3": "Promote voluntary emissions reduction",
    "climate_cons_mod_4": "Balance energy independence with environment",
    "climate_cons_strong_2": "Expand fossil fuel production",
    "climate_cons_strong_3": "Eliminate EPA regulations",
    "climate_cons_strong_4": "Withdraw from climate agreements",

    # IMMIGRATION
    # Progressive
    "immig_prog_strong_2": "Abolish ICE and border patrol",
    "immig_prog_strong_3": "Grant citizenship to all current residents",
    "immig_prog_strong_4": "Provide full benefits to all immigrants",
    "immig_prog_mod_2": "Expand refugee and asylum programs",
    "immig_prog_mod_3": "Create more legal immigration pathways",
    "immig_prog_mod_4": "Support sanctuary city policies",
    # Moderate
    "immig_mod_3": "Modernize visa system and border security",
    "immig_mod_4": "Guest worker programs with oversight",
    "immig_mod_5": "Skills-based immigration with family unity",
    # Conservative
    "immig_cons_mod_2": "Points-based immigration system",
    "immig_cons_mod_3": "Strengthen visa tracking system",
    "immig_cons_mod_4": "Reform chain migration policies",
    "immig_cons_strong_2": "End birthright citizenship",
    "immig_cons_strong_3": "Deport all undocumented immigrants",
    "immig_cons_strong_4": "Build physical barriers on all borders",

    # SOCIAL ISSUES
    # Progressive
    "social_prog_strong_2": "Mandate diversity quotas in all institutions",
    "social_prog_strong_3": "Reparations for historical injustices",
    "social_prog_strong_4": "Restructure all systems for equity",
    "social_prog_mod_2": "Reform police and justice systems",
    "social_prog_mod_3": "Expand civil rights protections",
    "social_prog_mod_4": "Increase funding for social programs",
    # Moderate
    "social_mod_3": "Promote dialogue across differences",
    "social_mod_4": "Evidence-based social policy reform",
    "social_mod_5": "Balance individual rights and community needs",
    # Conservative
    "social_cons_mod_2": "Protect religious freedom rights",
    "social_cons_mod_3": "Support faith-based initiatives",
    "social_cons_mod_4": "Maintain current social structures",
    "social_cons_strong_2": "Return to traditional family values",
    "social_cons_strong_3": "Limit social change through legislation",
    "social_cons_strong_4": "Promote religious values in policy",

    # GUN POLICY
    # Progressive
    "guns_prog_strong_2": "Mandatory gun buyback programs",
    "guns_prog_strong_3": "Ban private gun ownership",
    "guns_prog_strong_4": "Strict liability for gun manufacturers",
    "guns_prog_mod_2": "Create gun ownership database",
    "guns_prog_mod_3": "Require insurance for gun owners",
    "guns_prog_mod_4": "Ban high-capacity magazines",
    # Moderate
    "guns_mod_3": "Improve mental health screening",
    "guns_mod_4": "Register assault-style weapons",
    "guns_mod_5": "Support responsible gun ownership",
    # Conservative
    "guns_cons_mod_2": "State-level gun policy control",
    "guns_cons_mod_3": "Focus on mental health not guns",
    "guns_cons_mod_4": "Protect concealed carry rights",
    "guns_cons_strong_2": "Allow open carry everywhere",
    "guns_cons_strong_3": "Eliminate waiting periods",
    "guns_cons_strong_4": "Constitutional carry nationwide",

    # EDUCATION
    # Progressive
    "edu_prog_strong_2": "Free universal preschool through PhD",
    "edu_prog_strong_3": "Cancel all student debt",
    "edu_prog_strong_4": "Federalize all education systems",
    "edu_prog_mod_2": "Increase teacher pay significantly",
    "edu_prog_mod_3": "Expand early childhood programs",
    "edu_prog_mod_4": "Fund arts and enrichment programs",
    # Moderate
    "edu_mod_3": "Support vocational training options",
    "edu_mod_4": "Reform standardized testing",
    "edu_mod_5": "Modernize curriculum standards",
    # Conservative
    "edu_cons_mod_2": "Promote charter school expansion",
    "edu_cons_mod_3": "Focus on core academics",
    "edu_cons_mod_4": "Support homeschooling rights",
    "edu_cons_strong_2": "Eliminate Department of Education",
    "edu_cons_strong_3": "End public education funding",
    "edu_cons_strong_4": "Full privatization of schools",

    # FOREIGN POLICY
    # Progressive
    "foreign_prog_strong_2": "Close all foreign military bases",
    "foreign_prog_strong_3": "End all military alliances",
    "foreign_prog_strong_4": "Eliminate nuclear weapons",
    "foreign_prog_mod_2": "Expand international aid programs",
    "foreign_prog_mod_3": "Strengthen UN involvement",
    "foreign_prog_mod_4": "Focus on climate cooperation",
    # Moderate
    "foreign_mod_3": "Support strategic partnerships",
    "foreign_mod_4": "Maintain regional stability",
    "foreign_mod_5": "Promote democratic values abroad",
    # Conservative
    "foreign_cons_mod_2": "Strengthen military alliances",
    "foreign_cons_mod_3": "Increase defense readiness",
    "foreign_cons_mod_4": "Support strategic deterrence",
    "foreign_cons_strong_2": "Double military spending",
    "foreign_cons_strong_3": "Unilateral foreign policy",
    "foreign_cons_strong_4": "Expand nuclear arsenal"
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