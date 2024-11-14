import numpy as np
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import ollama


class MetricLearner:
    def __init__(self, input_dim, embedding_dim=32):
        self.model = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    

    def convert_to_tensor(self, X):
        """Convert input to PyTorch tensor"""
        if isinstance(X, np.ndarray):
            return torch.FloatTensor(X)
        return X

    def compute_distance(self, x1, x2):
        """Compute similarity between transformed pairs"""
        x1 = self.convert_to_tensor(x1)
        x2 = self.convert_to_tensor(x2)
        
        x1_transformed = self.model(x1)
        x2_transformed = self.model(x2)
        
        x1_transformed = nn.functional.normalize(x1_transformed, p=2, dim=1)
        x2_transformed = nn.functional.normalize(x2_transformed, p=2, dim=1)
        
        return torch.sum(x1_transformed * x2_transformed, dim=1)
    
    def get_similarity(self, x1, x2, temperature=1.0):
        """Get similarity between two vectors with temperature scaling"""
        self.model.eval()
        with torch.no_grad():
            sim = self.compute_distance(x1, x2)
            return (sim / temperature).numpy()
    
    def similarity_loss(self, predicted_sim, target_sim):
        """MSE loss between predicted and target similarities"""
        return torch.mean((predicted_sim - target_sim) ** 2)
    
    def train_step(self, x1, x2, similarities, batch_size=32):
        self.model.train()
        
        x1 = self.convert_to_tensor(x1)
        x2 = self.convert_to_tensor(x2)
        similarities = self.convert_to_tensor(similarities)
        
        predicted_similarities = self.compute_distance(x1, x2)
        loss = self.similarity_loss(predicted_similarities, similarities)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    def train(self, pairs, similarities, n_epochs=100, batch_size=32):
        """
        Train the metric learner
        pairs: list of (x1, x2) tuples
        similarities: list of similarity scores between 0 and 1
        """
        x1 = np.array([p[0] for p in pairs])
        x2 = np.array([p[1] for p in pairs])
        similarities = np.array(similarities)
        
        n_samples = len(pairs)
        indices = np.arange(n_samples)
        
        for epoch in range(n_epochs):
            # Shuffle data
            np.random.shuffle(indices)
            x1_shuffled = x1[indices]
            x2_shuffled = x2[indices]
            similarities_shuffled = similarities[indices]
            
            # Train in batches
            total_loss = 0
            for i in range(0, n_samples, batch_size):
                batch_x1 = x1_shuffled[i:i+batch_size]
                batch_x2 = x2_shuffled[i:i+batch_size]
                batch_similarities = similarities_shuffled[i:i+batch_size]
                
                loss = self.train_step(batch_x1, batch_x2, batch_similarities, batch_size)
                total_loss += loss
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/((n_samples+batch_size-1)//batch_size):.4f}")

def prepare_training_data(embeddings, statement_pairs):
    """
    Prepare training data from statement pairs with similarity scores
    embeddings: dict of {statement_id: embedding}
    statement_pairs: list of (id1, id2, similarity_score) tuples
    """
    pairs = []
    similarities = []
    
    for id1, id2, sim_score in statement_pairs:
        pairs.append((embeddings[id1], embeddings[id2]))
        similarities.append(sim_score)
    
    return pairs, similarities



def get_embedding(text):
    """Get embedding using Ollama Python library."""
    response = ollama.embeddings(
        model='mxbai-embed-large',
        prompt=text
    )
    return np.array(response['embedding'])

def embed_statements(statements):
    return {k: get_embedding(v) for k, v in statements.items()}


statements = {
    # Economic Policy
    'e1': 'We should increase taxes on high-income earners to fund social programs',
    'e2': 'The wealthy should pay their fair share to support society',
    'e3': 'Tax cuts for the wealthy create jobs and economic growth',
    'e4': 'Government spending on social programs should be reduced',
    'e5': 'A universal basic income is necessary in modern society',
    'e6': 'Corporate tax loopholes should be eliminated',
    'e7': 'Free market capitalism works best with minimal regulation',
    'e8': 'Workers deserve a significantly higher minimum wage',
    'e9': 'Economic inequality is a major threat to society',
    'e10': 'Government should not interfere with market forces',
    'e11': 'Break up large corporations to increase competition',
    'e12': 'Reduce regulations to stimulate economic growth',
    
    # Healthcare
    'h1': 'Healthcare should be provided as a universal public service',
    'h2': 'Everyone deserves access to affordable healthcare',
    'h3': 'Private healthcare leads to better quality and innovation',
    'h4': 'Government should stay out of healthcare decisions',
    'h5': 'Mental healthcare should be included in basic coverage',
    'h6': 'Healthcare costs are best controlled by market competition',
    'h7': 'Medical decisions should be between doctor and patient only',
    'h8': 'Preventive care should be free for everyone',
    'h9': 'Healthcare is a fundamental human right',
    'h10': 'Private insurance provides the best healthcare options',
    'h11': 'Medicare should be expanded to cover all ages',
    'h12': 'Healthcare innovation requires market incentives',
    
    # Immigration
    'i1': 'Immigration strengthens our economy and enriches our culture',
    'i2': 'We should welcome skilled workers from other countries',
    'i3': 'Immigration levels should be strictly limited',
    'i4': 'Strong borders are essential for national security',
    'i5': 'Provide pathway to citizenship for all immigrants',
    'i6': 'Merit-based immigration system is most effective',
    'i7': 'Sanctuary cities protect community safety',
    'i8': 'Immigration policy should prioritize national interests',
    'i9': 'Cultural diversity improves our society',
    'i10': 'Illegal immigration threatens our sovereignty',
    'i11': 'Family reunification should guide immigration policy',
    'i12': 'Immigration enforcement needs to be stricter',
    
    # Environment
    'en1': 'Climate change requires immediate government action',
    'en2': 'We must transition to renewable energy sources',
    'en3': 'Environmental regulations hurt business growth',
    'en4': 'Climate policies should not compromise economic development',
    'en5': 'Carbon emissions should be heavily taxed',
    'en6': 'Nuclear power is essential for clean energy',
    'en7': 'Green technology creates economic opportunities',
    'en8': 'Environmental protection costs jobs',
    'en9': 'Fossil fuels are still necessary for economic growth',
    'en10': 'Renewable energy can power our entire economy',
    'en11': 'Individual action is key to environmental protection',
    'en12': 'Market solutions are best for environmental problems'
}

# Statement pairs with similarity scores (0-1)
statement_pairs = [
    # Very similar statements (0.8-1.0)
    ('e1', 'e2', 0.9),    # Similar progressive tax views
    ('h1', 'h2', 0.85),   # Similar healthcare views
    ('i1', 'i2', 0.8),    # Similar pro-immigration views
    ('en1', 'en2', 0.85), # Similar climate action views
    ('e8', 'e9', 0.85),   # Progressive economic views
    ('h9', 'h11', 0.9),   # Progressive healthcare views
    ('i5', 'i7', 0.8),    # Progressive immigration views
    ('en5', 'en10', 0.85), # Progressive environmental views
    
    # Moderately similar statements (0.5-0.7)
    ('en2', 'en4', 0.6),  # Both about climate but different priorities
    ('i2', 'i3', 0.5),    # Both about immigration control but different emphasis
    ('h2', 'h3', 0.55),   # Both about healthcare quality but different approaches
    ('e6', 'e7', 0.6),    # Mixed economic views
    ('h5', 'h7', 0.65),   # Mixed healthcare views
    ('i6', 'i8', 0.7),    # Mixed immigration views
    ('en6', 'en7', 0.6),  # Mixed environmental views
    
    # Opposing statements (0.0-0.3)
    ('e1', 'e3', 0.2),    # Opposing tax views
    ('h1', 'h4', 0.1),    # Opposing healthcare views
    ('i1', 'i4', 0.15),   # Opposing immigration views
    ('en1', 'en3', 0.1),  # Opposing environmental views
    ('e9', 'e10', 0.15),  # Opposing economic views
    ('h9', 'h10', 0.2),   # Opposing healthcare views
    ('i5', 'i12', 0.1),   # Opposing immigration views
    ('en5', 'en9', 0.15), # Opposing environmental views
    
    # Cross-topic pairs with ideological alignment
    ('e1', 'h1', 0.7),    # Progressive views on tax and healthcare
    ('e3', 'h3', 0.7),    # Conservative views on tax and healthcare
    ('i3', 'en3', 0.65),  # Conservative views on immigration and environment
    ('e9', 'h9', 0.75),   # Strong progressive alignment
    ('e7', 'h6', 0.7),    # Strong conservative alignment
    ('i10', 'en8', 0.65), # Conservative views across topics
    
    # Cross-topic pairs with little relation
    ('e1', 'i2', 0.4),    # Tax policy vs immigration - some ideological overlap
    ('h2', 'en1', 0.45),  # Healthcare access vs climate - less related
    ('e4', 'i1', 0.3),    # Government spending vs immigration - different topics
    ('e11', 'h5', 0.35),  # Corporate policy vs healthcare - different topics
    ('i8', 'en6', 0.4),   # Immigration vs energy policy
    ('h7', 'en11', 0.3)   # Healthcare autonomy vs environmental responsibility
]



def compute_similarity_matrix(learner, embeddings, statements, temperature=1.0):
    """
    Compute similarity matrix between all statements using the learned metric
    """
    statement_ids = list(statements.keys())
    n_statements = len(statement_ids)
    similarity_matrix = np.zeros((n_statements, n_statements))
    
    # Get embeddings as array
    embeddings_array = np.array([embeddings[sid] for sid in statement_ids])
    
    # Convert to tensor all at once
    embeddings_tensor = learner.convert_to_tensor(embeddings_array)
    
    # Compute all similarities
    with torch.no_grad():
        for i in range(n_statements):
            for j in range(i, n_statements):
                sim = learner.get_similarity(
                    embeddings_array[i:i+1], 
                    embeddings_array[j:j+1],
                    temperature=temperature
                )
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
    
    return similarity_matrix, statement_ids

def print_similarity_comparison(orig_sims, learned_sims, statement_ids, top_k=5):
    """
    Print comparison of original vs learned similarities
    """
    n = len(statement_ids)
    
    print("\nMost different similarity pairs (learned vs original):")
    differences = []
    for i in range(n):
        for j in range(i+1, n):
            diff = abs(learned_sims[i,j] - orig_sims[i,j])
            differences.append((diff, i, j))
    
    differences.sort(reverse=True)
    for diff, i, j in differences[:top_k]:
        print(f"\n{statement_ids[i]} <-> {statement_ids[j]}")
        print(f"Original similarity: {orig_sims[i,j]:.3f}")
        print(f"Learned similarity: {learned_sims[i,j]:.3f}")
        print(f"Difference: {diff:.3f}")

# Modified training code:
embeddings = embed_statements(statements)
pairs, similarities = prepare_training_data(embeddings, statement_pairs)

# Compute original similarity matrix
orig_matrix = np.zeros((len(statements), len(statements)))
statement_ids = list(statements.keys())
for i, id1 in enumerate(statement_ids):
    for j, id2 in enumerate(statement_ids):
        # Using cosine similarity
        sim = np.dot(embeddings[id1], embeddings[id2]) / (
            np.linalg.norm(embeddings[id1]) * np.linalg.norm(embeddings[id2])
        )
        orig_matrix[i,j] = sim

print("\nOriginal similarities on training pairs:")
for (id1, id2, target_sim) in statement_pairs:
    i, j = statement_ids.index(id1), statement_ids.index(id2)
    print(f"{id1} <-> {id2}: Target: {target_sim:.3f}, Original: {orig_matrix[i,j]:.3f}")

# Initialize and get initial loss
learner = MetricLearner(input_dim=1024)
with torch.no_grad():
    x1 = np.array([p[0] for p in pairs])
    x2 = np.array([p[1] for p in pairs])
    initial_sims = learner.compute_distance(x1, x2)
    initial_loss = learner.similarity_loss(initial_sims, torch.FloatTensor(similarities))
    print(f"\nInitial loss before training: {initial_loss.item():.4f}")

# Train
learner.train(pairs, similarities, n_epochs=50)

# Compute similarity matrix with learned metric
learned_matrix, _ = compute_similarity_matrix(learner, embeddings, statements)

# Compare similarities
print_similarity_comparison(orig_matrix, learned_matrix, statement_ids)

# Also print final similarities for training pairs
print("\nFinal similarities on training pairs:")
for (id1, id2, target_sim) in statement_pairs:
    i, j = statement_ids.index(id1), statement_ids.index(id2)
    print(f"{id1} <-> {id2}: Target: {target_sim:.3f}, Learned: {learned_matrix[i,j]:.3f}")



# Load and process the new statements
new_statements = {
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
    }  # Your pasted statements dictionary

# Get embeddings for new statements
new_embeddings = embed_statements(new_statements)

# Compute original cosine similarity matrix for new statements
orig_matrix_new = np.zeros((len(new_statements), len(new_statements)))
new_statement_ids = list(new_statements.keys())
for i, id1 in enumerate(new_statement_ids):
    for j, id2 in enumerate(new_statement_ids):
        sim = np.dot(new_embeddings[id1], new_embeddings[id2]) / (
            np.linalg.norm(new_embeddings[id1]) * np.linalg.norm(new_embeddings[id2])
        )
        orig_matrix_new[i,j] = sim

# Compute similarity matrix with learned metric for new statements
learned_matrix_new, _ = compute_similarity_matrix(learner, new_embeddings, new_statements)

# Print some interesting comparisons
def print_topic_analysis(matrix, statement_ids, topic_prefix):
    """Print average similarities within a topic group"""
    indices = [i for i, sid in enumerate(statement_ids) if sid.startswith(topic_prefix)]
    if not indices:
        return
    
    sims = matrix[indices][:, indices]
    avg_sim = (sims.sum() - len(indices)) / (len(indices) * (len(indices) - 1))  # exclude diagonal
    print(f"{topic_prefix}: Average similarity = {avg_sim:.3f}")

# Analyze similarities within topics
print("\nOriginal similarities within topics:")
for topic in ['econ_', 'health_', 'climate_', 'immig_', 'social_', 'guns_', 'edu_', 'foreign_']:
    print_topic_analysis(orig_matrix_new, new_statement_ids, topic)

print("\nLearned similarities within topics:")
for topic in ['econ_', 'health_', 'climate_', 'immig_', 'social_', 'guns_', 'edu_', 'foreign_']:
    print_topic_analysis(learned_matrix_new, new_statement_ids, topic)

# Print some cross-ideology comparisons
def print_ideology_analysis(matrix, statement_ids):
    """Print average similarities between progressive and conservative statements"""
    prog_indices = [i for i, sid in enumerate(statement_ids) if 'prog_' in sid]
    cons_indices = [i for i, sid in enumerate(statement_ids) if 'cons_' in sid]
    mod_indices = [i for i, sid in enumerate(statement_ids) if 'mod_' in sid]
    
    if prog_indices and cons_indices:
        prog_cons_sims = matrix[prog_indices][:, cons_indices]
        avg_sim = prog_cons_sims.mean()
        print(f"Progressive-Conservative similarity: {avg_sim:.3f}")
    
    if prog_indices and mod_indices:
        prog_mod_sims = matrix[prog_indices][:, mod_indices]
        avg_sim = prog_mod_sims.mean()
        print(f"Progressive-Moderate similarity: {avg_sim:.3f}")
    
    if cons_indices and mod_indices:
        cons_mod_sims = matrix[cons_indices][:, mod_indices]
        avg_sim = cons_mod_sims.mean()
        print(f"Conservative-Moderate similarity: {avg_sim:.3f}")

print("\nOriginal ideological similarities:")
print_ideology_analysis(orig_matrix_new, new_statement_ids)

print("\nLearned ideological similarities:")
print_ideology_analysis(learned_matrix_new, new_statement_ids)

# Print the most extreme differences
print("\nMost changed similarities after learning:")
print_similarity_comparison(orig_matrix_new, learned_matrix_new, new_statement_ids, top_k=10)


import matplotlib.pyplot as plt
import seaborn as sns

def plot_similarity_matrices(orig_matrix, learned_matrix, statement_ids):
    """
    Plot original and learned similarity matrices side by side
    """
    # Set up the figure
    plt.figure(figsize=(20, 8))
    
    # Create labels for topics and ideologies
    labels = [f"{id.split('_')[0]}_{id.split('_')[1]}" for id in statement_ids]
    # Create labels for topics and ideologies
    labels = [f"{id.split('_')[0]}_{id.split('_')[1]}" for id in statement_ids]
    
    # Create a custom colormap that goes from blue to white to red
    colors = ['darkblue', 'white', 'darkred']
    n_bins = 100  # Number of color gradations
    cmap = sns.blend_palette(colors, n_bins, as_cmap=True)
    # Plot original similarities
    plt.subplot(1, 2, 1)
    sns.heatmap(orig_matrix, 
                cmap=cmap,
                center=0,
                vmin=-1, 
                vmax=1,
                xticklabels=labels, 
                yticklabels=labels)
    plt.title('Original Cosine Similarities')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    # Plot learned similarities
    plt.subplot(1, 2, 2)
    sns.heatmap(learned_matrix, cmap='RdBu_r', center=0,
                xticklabels=labels, yticklabels=labels)
    plt.title('Learned Similarities')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()

    # Also plot the difference matrix
    plt.figure(figsize=(10, 8))
    diff_matrix = learned_matrix - orig_matrix
    sns.heatmap(diff_matrix, cmap='RdBu_r', center=0,
                xticklabels=labels, yticklabels=labels)
    plt.title('Difference (Learned - Original)')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Plot the matrices
plot_similarity_matrices(orig_matrix_new, learned_matrix_new, new_statement_ids)

# You might also want to plot averages by category
def plot_category_similarities(matrix, statement_ids):
    """Plot average similarities within and between categories"""
    categories = sorted(list(set([id.split('_')[0] for id in statement_ids])))
    n_categories = len(categories)
    avg_matrix = np.zeros((n_categories, n_categories))
    
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            # Get indices for both categories
            indices1 = [idx for idx, sid in enumerate(statement_ids) if sid.startswith(cat1)]
            indices2 = [idx for idx, sid in enumerate(statement_ids) if sid.startswith(cat2)]
            
            # Calculate average similarity between categories
            avg_matrix[i, j] = matrix[np.ix_(indices1, indices2)].mean()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_matrix, cmap='RdBu_r', center=0,
                xticklabels=categories, yticklabels=categories)
    plt.title('Average Category Similarities')
    plt.tight_layout()
    plt.show()
    
    return avg_matrix

# Plot category similarities for both original and learned matrices
print("\nCategory-level similarities:")
orig_cat_matrix = plot_category_similarities(orig_matrix_new, new_statement_ids)
print("\nLearned category-level similarities:")
learned_cat_matrix = plot_category_similarities(learned_matrix_new, new_statement_ids)