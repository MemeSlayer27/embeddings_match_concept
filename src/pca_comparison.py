import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import ollama

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
    
    return dot_product / (norm_v1 * norm_v2) if (norm_v1 != 0 and norm_v2 != 0) else 0

def create_similarity_matrix_with_pca(statements, n_components=1000):
    """Create a similarity matrix for all statements using PCA-reduced embeddings."""
    n = len(statements)
    
    # Get all embeddings first
    print("\nGetting embeddings for all statements...")
    embeddings_list = []
    keys = list(statements.keys())
    
    for i, (key, text) in enumerate(statements.items()):
        print(f"Processing {i+1}/{n}: {key}")
        embedding = get_embedding(text)
        embeddings_list.append(embedding)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings_list)
    
    # Apply PCA
    print(f"\nApplying PCA to reduce dimensions to {n_components} components...")
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings_array)
    
    # Print variance explained
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    print(f"Variance explained by {n_components} components: {cumulative_variance[-1]:.3f}")
    
    # Compute similarity matrix using reduced embeddings
    print("\nComputing similarity matrix...")
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = cosine_similarity(reduced_embeddings[i], reduced_embeddings[j])
    
    return matrix, keys, pca, reduced_embeddings

def plot_similarity_matrix(matrix, labels):
    """Plot a heatmap of the similarity matrix with improved readability."""
    plt.figure(figsize=(20, 16))
    
    # Create more readable labels
    readable_labels = []
    for label in labels:
        parts = label.split('_')
        position = 'Progressive' if 'prog' in label else 'Conservative' if 'cons' in label else 'Moderate'
        position += '+' if 'strong' in label else ''
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
    
    plt.title("Political Statement Similarity Matrix (PCA-reduced)", pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_pca_components(reduced_embeddings, labels, n_display=2):
    """Plot the first n_display principal components."""
    plt.figure(figsize=(12, 8))
    
    # Create color mapping for different categories
    categories = [label.split('_')[0] for label in labels]
    unique_categories = list(set(categories))
    color_map = dict(zip(unique_categories, plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))))
    colors = [color_map[cat] for cat in categories]
    
    # Create marker mapping for different positions
    position_markers = {
        'prog': 'o',  # Progressive
        'cons': 's',  # Conservative
        'mod': '^'    # Moderate
    }
    markers = [position_markers[label.split('_')[1]] for label in labels]
    
    # Plot points
    for i in range(len(reduced_embeddings)):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], 
                   c=[colors[i]], marker=markers[i], s=100)
    
    # Add legend
    legend_elements = []
    # Categories legend
    for cat in unique_categories:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color_map[cat], label=cat.capitalize(),
                                        markersize=10))
    # Position legend
    for pos, marker in position_markers.items():
        legend_elements.append(plt.Line2D([0], [0], marker=marker, color='gray', 
                                        label=pos.capitalize(), markersize=10))
    
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA Visualization of Political Statements')
    plt.tight_layout()
    plt.show()

def main():
    # Get statements
    statements = get_aligned_political_statements()
    
    # Create similarity matrix with PCA
    matrix, labels, pca, reduced_embeddings = create_similarity_matrix_with_pca(statements, n_components=100)
    
    # Plot similarity matrix
    plot_similarity_matrix(matrix, labels)
    
    # Plot PCA components
    plot_pca_components(reduced_embeddings, labels)


if __name__ == "__main__":
    main()