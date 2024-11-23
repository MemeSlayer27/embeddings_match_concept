import numpy as np
import ollama
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances



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

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

def create_distance_matrices(statements):
    """Create both similarity and distance matrices for all statements."""
    n = len(statements)
    cosine_matrix = np.zeros((n, n))
    
    # Get all embeddings first to avoid recomputing
    embeddings = {}
    print("\nGetting embeddings for all statements...")
    for i, (key, text) in enumerate(statements.items()):
        print(f"Processing {i+1}/{n}: {key}")
        embeddings[key] = get_embedding(text)
    
    # Convert embeddings to a matrix for euclidean distance calculation
    embedding_matrix = np.array([embeddings[key] for key in statements.keys()])
    
    # Compute cosine similarity matrix
    print("\nComputing similarity matrices...")
    keys = list(statements.keys())
    for i, key1 in enumerate(keys):
        for j, key2 in enumerate(keys):
            cosine_matrix[i, j] = cosine_similarity(embeddings[key1], embeddings[key2])
    
    # Compute euclidean distance matrix
    euclidean_matrix = euclidean_distances(embedding_matrix)
    
    return cosine_matrix, euclidean_matrix, keys

def plot_distance_matrices(cosine_matrix, euclidean_matrix, labels):
    """Plot heatmaps of both the similarity and distance matrices side by side."""
    # Create more readable labels
    readable_labels = []
    for label in labels:
        parts = label.split('_')
        if 'prog' in label:
            position = 'Progressive'
        elif 'cons' in label:
            position = 'Conservative'
        elif 'mod' in label:
            position = 'Moderate'
        
        if 'strong' in label:
            position += '+'
            
        category = parts[0].capitalize()
        readable_labels.append(f"{category}\n{position}")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 16))
    
    # Plot cosine similarity matrix
    sns.heatmap(cosine_matrix,
                xticklabels=readable_labels,
                yticklabels=readable_labels,
                cmap='RdYlBu_r',
                vmin=-1,  # Full range for cosine similarity
                vmax=1,
                center=0,
                ax=ax1)
    ax1.set_title("Cosine Similarity Matrix (-1 to 1)", pad=20)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
    
    # Plot euclidean distance matrix
    # Get the actual range for better visualization
    vmax = np.max(euclidean_matrix)
    sns.heatmap(euclidean_matrix,
                xticklabels=readable_labels,
                yticklabels=readable_labels,
                cmap='RdYlBu',  # Note: Using RdYlBu since higher distance = more different
                vmin=0,  # Euclidean distance is always non-negative
                vmax=vmax,  # Use actual maximum
                center=vmax/2,
                ax=ax2)
    ax2.set_title(f"Euclidean Distance Matrix (0 to {vmax:.2f})", pad=20)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    
    plt.suptitle("Comparison of Distance Metrics for Political Statements", fontsize=16)
    plt.tight_layout()
    plt.show()

def analyze_distance_patterns(cosine_matrix, euclidean_matrix, labels):
    """Analyze patterns in both matrices focusing on ideological alignment."""
    print("\nComparative Distance Pattern Analysis:")
    print("-" * 50)
    
    # Group indices by ideology
    prog_indices = [i for i, label in enumerate(labels) if 'prog' in label]
    cons_indices = [i for i, label in enumerate(labels) if 'cons' in label]
    mod_indices = [i for i, label in enumerate(labels) if 'mod' in label]
    
    def get_group_metrics(indices1, indices2, cosine_mat, euclidean_mat):
        cosine_similarities = []
        euclidean_distances = []
        for i in indices1:
            for j in indices2:
                if i != j:  # Exclude self-comparison
                    cosine_similarities.append(cosine_mat[i,j])
                    euclidean_distances.append(euclidean_mat[i,j])
        return (np.mean(cosine_similarities) if cosine_similarities else 0,
                np.mean(euclidean_distances) if euclidean_distances else 0)
    
    print("\nAverage Metrics Between Ideological Groups:")
    print("Group Comparison            Cosine Similarity    Euclidean Distance")
    print("-" * 65)
    
    comparisons = [
        ("Progressive <-> Progressive", prog_indices, prog_indices),
        ("Conservative <-> Conservative", cons_indices, cons_indices),
        ("Moderate <-> Moderate", mod_indices, mod_indices),
        ("Progressive <-> Conservative", prog_indices, cons_indices),
        ("Progressive <-> Moderate", prog_indices, mod_indices),
        ("Conservative <-> Moderate", cons_indices, mod_indices)
    ]
    
    for name, indices1, indices2 in comparisons:
        cos_sim, euc_dist = get_group_metrics(indices1, indices2, cosine_matrix, euclidean_matrix)
        print(f"{name:<25} {cos_sim:>16.3f} {euc_dist:>19.3f}")
    
    # Add overall statistics
    print("\nOverall Statistics:")
    print(f"Cosine Similarity Range: [{np.min(cosine_matrix):.3f}, {np.max(cosine_matrix):.3f}]")
    print(f"Euclidean Distance Range: [{np.min(euclidean_matrix):.3f}, {np.max(euclidean_matrix):.3f}]")

def main():
    # Get statements
    statements = get_aligned_political_statements()
    
    # Create matrices
    cosine_matrix, euclidean_matrix, labels = create_distance_matrices(statements)
    
    # Plot matrices
    plot_distance_matrices(cosine_matrix, euclidean_matrix, labels)
    
    # Analyze patterns
    analyze_distance_patterns(cosine_matrix, euclidean_matrix, labels)

if __name__ == "__main__":
    main()