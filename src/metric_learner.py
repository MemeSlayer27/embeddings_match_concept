import numpy as np
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import ollama
import metric_data

class MetricLearner:
    def __init__(self, input_dim, embedding_dim=128):
        self.model = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def convert_to_tensor(self, X):
        """Convert input to PyTorch tensor"""
        if isinstance(X, np.ndarray):
            return torch.FloatTensor(X).requires_grad_(True)
        elif isinstance(X, torch.Tensor):
            return X.requires_grad_(True)
        return torch.FloatTensor(X).requires_grad_(True)

    def get_similarity(self, x1, x2, temperature=1.0):
        """Get similarity from Euclidean distance in transformed space"""
        # Don't use no_grad here during training since we need gradients
        if not self.model.training:
            with torch.no_grad():
                # Transform inputs through learned network
                x1_transformed = self.model(self.convert_to_tensor(x1))
                x2_transformed = self.model(self.convert_to_tensor(x2))
                
                # Compute Euclidean distance
                distances = torch.sqrt(torch.sum((x1_transformed - x2_transformed) ** 2, dim=1))
                
                # Convert to similarity using Gaussian kernel
                similarities = torch.exp(-distances ** 2 / (2 * temperature ** 2))
                return similarities.numpy()
        else:
            # During training, we need gradients
            x1_transformed = self.model(self.convert_to_tensor(x1))
            x2_transformed = self.model(self.convert_to_tensor(x2))
            
            # Compute Euclidean distance
            distances = torch.sqrt(torch.sum((x1_transformed - x2_transformed) ** 2, dim=1))
            
            # Convert to similarity using Gaussian kernel
            similarities = torch.exp(-distances ** 2 / (2 * temperature ** 2))
            return similarities

    def train_step(self, x1, x2, similarities, batch_size=32):
        self.model.train()
        
        x1 = self.convert_to_tensor(x1)
        x2 = self.convert_to_tensor(x2)
        similarities = self.convert_to_tensor(similarities)
        
        predicted_similarities = self.get_similarity(x1, x2)
        loss = self.similarity_loss(predicted_similarities, similarities)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def similarity_loss(self, predicted_sim, target_sim):
        """MSE loss between predicted and target similarities"""
        # Convert both inputs to tensors if they aren't already
        predicted_sim = self.convert_to_tensor(predicted_sim)
        target_sim = self.convert_to_tensor(target_sim)
        return torch.mean((predicted_sim - target_sim) ** 2)
    
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
    'en12': 'Market solutions are best for environmental problems',
        # Education Policy
    'ed1': 'Public schools need increased government funding',
    'ed2': 'Teachers deserve higher salaries and better benefits',
    'ed3': 'School choice and vouchers improve education quality',
    'ed4': 'Parents should have more control over curriculum',
    'ed5': 'Student loan debt should be forgiven',
    'ed6': 'Merit-based education produces the best outcomes',
    'ed7': 'Schools should focus on practical job skills',
    'ed8': 'Free college education is a public good',
    'ed9': 'Private schools provide superior education',
    'ed10': 'Educational standards should be nationally unified',
    'ed11': 'Charter schools drain public education resources',
    'ed12': 'Competition improves educational outcomes',

    # Social Issues
    's1': 'Marriage equality is a fundamental right',
    's2': 'Gender equality requires affirmative action',
    's3': 'Traditional values strengthen society',
    's4': 'Religious freedom must be protected absolutely',
    's5': 'Systemic racism requires institutional reform',
    's6': 'Merit alone should determine advancement',
    's7': 'Gender roles are naturally determined',
    's8': 'Diversity initiatives improve organizations',
    's9': 'Political correctness limits free expression',
    's10': 'Social justice movements unite communities',
    's11': 'Identity politics divides society',
    's12': 'Equal opportunity exists for everyone already',

    # Public Safety
    'p1': 'Police departments need major reform',
    'p2': 'Gun ownership is a fundamental right',
    'p3': 'Stricter gun control laws are necessary',
    'p4': 'Law enforcement needs more funding',
    'p5': 'Rehabilitation should replace punishment',
    'p6': 'Violent crime requires harsher sentences',
    'p7': 'Drug use should be decriminalized',
    'p8': 'Private prisons should be abolished',
    'p9': 'Stand your ground laws protect citizens',
    'p10': 'Community policing improves safety',
    'p11': 'Capital punishment deters crime',
    'p12': 'Public safety requires armed citizens',

    # Technology Policy
    't1': 'Big tech companies need more regulation',
    't2': 'Personal data privacy requires strict laws',
    't3': 'Government surveillance ensures security',
    't4': 'Innovation requires minimal regulation',
    't5': 'Social media companies should censor harmful content',
    't6': 'Free speech online should be absolute',
    't7': 'Digital privacy is a fundamental right',
    't8': 'Cryptocurrency needs strict regulation',
    't9': 'Tech self-regulation is most effective',
    't10': 'AI development needs government oversight',
    't11': 'Internet access is a human right',
    't12': 'Market forces best guide tech development'
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
    ('e1', 'e3', 0.1),    # Opposing tax views
    ('h1', 'h4', 0.1),    # Opposing healthcare views
    ('i1', 'i4', 0.1),   # Opposing immigration views
    ('en1', 'en3', 0.1),  # Opposing environmental views
    ('e9', 'e10', 0.1),  # Opposing economic views
    ('h9', 'h10', 0.1),   # Opposing healthcare views
    ('i5', 'i12', 0.1),   # Opposing immigration views
    ('en5', 'en9', 0.1), # Opposing environmental views
    
    # Cross-topic pairs with ideological alignment
    ('e1', 'h1', 0.7),    # Progressive views on tax and healthcare
    ('e3', 'h3', 0.7),    # Conservative views on tax and healthcare
    ('i3', 'en3', 0.65),  # Conservative views on immigration and environment
    ('e9', 'h9', 0.75),   # Strong progressive alignment
    ('e7', 'h6', 0.7),    # Strong conservative alignment
    ('i10', 'en8', 0.65), # Conservative views across topics
    
    # Cross-topic pairs with little relation
    ('e1', 'i2', 0.1),    # Tax policy vs immigration - some ideological overlap
    ('h2', 'en1', 0.1),  # Healthcare access vs climate - less related
    ('e4', 'i1', 0.1),    # Government spending vs immigration - different topics
    ('e11', 'h5', 0.1),  # Corporate policy vs healthcare - different topics
    ('i8', 'en6', 0.1),   # Immigration vs energy policy
    ('h7', 'en11', 0.1),   # Healthcare autonomy vs environmental responsibility

    # Very similar statements (0.8-1.0)
    ('ed1', 'ed2', 0.9),   # Progressive education views
    ('s1', 's2', 0.85),    # Progressive social views
    ('p1', 'p8', 0.85),    # Progressive criminal justice views
    ('t1', 't2', 0.85),    # Progressive tech regulation views
    ('ed5', 'ed8', 0.9),   # Progressive education funding views
    ('s5', 's8', 0.85),    # Progressive diversity views
    ('p3', 'p5', 0.8),     # Progressive criminal justice approach
    ('t7', 't11', 0.85),   # Progressive digital rights views

    # Moderately similar statements (0.5-0.7)
    ('ed6', 'ed7', 0.65),  # Mixed education approach
    ('s3', 's4', 0.7),     # Conservative social values but different focus
    ('p2', 'p4', 0.6),     # Mixed law enforcement views
    ('t3', 't5', 0.55),    # Mixed regulation views
    ('ed10', 'ed12', 0.6), # Mixed education standards views
    ('s6', 's9', 0.65),    # Mixed social perspective
    ('p9', 'p12', 0.7),    # Conservative safety views

    # Opposing statements (0.0-0.3)
    ('ed1', 'ed9', 0.1),   # Public vs private education
    ('s1', 's7', 0.1),     # Progressive vs traditional values
    ('p1', 'p4', 0.1),     # Police reform vs funding
    ('t1', 't9', 0.1),     # Regulation vs self-regulation
    ('ed5', 'ed6', 0.2),   # Debt forgiveness vs merit-based
    ('s5', 's12', 0.1),    # Systemic reform vs status quo
    ('p3', 'p2', 0.1),     # Gun control vs gun rights
    ('t5', 't6', 0.1),     # Content moderation vs absolute free speech

    # Cross-topic pairs with ideological alignment
    ('ed1', 's2', 0.7),    # Progressive education and social views
    ('p2', 's4', 0.7),     # Conservative rights perspective
    ('t1', 'p1', 0.65),    # Progressive regulation views
    ('ed3', 't4', 0.7),    # Conservative market approach
    ('s5', 'p5', 0.75),    # Progressive reform alignment
    ('t6', 's9', 0.7),     # Conservative freedom alignment

    # Cross-topic pairs with little relation
    ('ed7', 'p10', 0.1),   # Job skills vs policing
    ('s1', 't8', 0.1),     # Marriage equality vs crypto
    ('p7', 't11', 0.1),    # Drug policy vs internet access
    ('t2', 'ed4', 0.1),    # Data privacy vs curriculum control
    ('s8', 'p11', 0.1),    # Diversity vs capital punishment
    ('ed12', 't3', 0.1),    # Education competition vs surveillance

    # Economic Policy Additional Pairs
    ('e1', 'e4', 0.1),     # Progressive tax vs reduced spending
    ('e1', 'e5', 0.85),    # Progressive redistribution alignment
    ('e1', 'e6', 0.8),     # Progressive tax policy alignment
    ('e1', 'e7', 0.15),    # Government intervention vs free market
    ('e1', 'e8', 0.8),     # Progressive economic views
    ('e1', 'e10', 0.1),    # Government intervention vs market forces
    ('e1', 'e11', 0.75),   # Progressive market regulation
    ('e1', 'e12', 0.1),    # Regulation vs deregulation
    ('e2', 'e3', 0.1),     # Fair share vs tax cuts
    ('e2', 'e4', 0.1),     # Support society vs reduced spending
    ('e2', 'e5', 0.8),     # Progressive wealth distribution
    ('e2', 'e6', 0.85),    # Tax policy alignment
    ('e2', 'e7', 0.15),    # Regulation vs free market
    ('e2', 'e8', 0.8),     # Progressive economic views
    ('e2', 'e9', 0.85),    # Economic inequality views
    ('e2', 'e10', 0.1),    # Regulation vs market forces
    ('e2', 'e11', 0.75),   # Market regulation alignment
    ('e2', 'e12', 0.1),    # Regulation vs deregulation
    ('e3', 'e4', 0.8),     # Conservative economic alignment
    ('e3', 'e5', 0.1),     # Tax cuts vs UBI
    ('e3', 'e6', 0.15),    # Tax policy opposition
    ('e3', 'e7', 0.85),    # Free market alignment
    ('e3', 'e8', 0.1),     # Conservative vs progressive economics
    ('e3', 'e9', 0.1),     # Economic inequality views
    ('e3', 'e10', 0.85),   # Market-focused alignment
    ('e3', 'e11', 0.15),   # Corporate policy disagreement
    ('e3', 'e12', 0.9),    # Conservative economic alignment
    
    # Healthcare Additional Pairs
    ('h1', 'h3', 0.15),    # Public vs private healthcare
    ('h1', 'h5', 0.85),    # Universal coverage alignment
    ('h1', 'h6', 0.1),     # Public vs market-based healthcare
    ('h1', 'h7', 0.6),     # Healthcare access views
    ('h1', 'h8', 0.9),     # Universal coverage alignment
    ('h1', 'h10', 0.1),    # Public vs private insurance
    ('h1', 'h12', 0.15),   # Government vs market healthcare
    ('h2', 'h4', 0.15),    # Access vs government non-intervention
    ('h2', 'h5', 0.85),    # Healthcare access alignment
    ('h2', 'h6', 0.15),    # Access vs market competition
    ('h2', 'h7', 0.65),    # Healthcare autonomy views
    ('h2', 'h8', 0.85),    # Healthcare access alignment
    ('h2', 'h10', 0.15),   # Affordable vs private healthcare
    ('h2', 'h11', 0.85),   # Healthcare access expansion
    ('h2', 'h12', 0.2),    # Access vs market incentives

    # Immigration Additional Pairs
    ('i1', 'i3', 0.15),    # Open vs restricted immigration
    ('i1', 'i5', 0.85),    # Pro-immigration alignment
    ('i1', 'i6', 0.6),     # Immigration benefits vs merit system
    ('i1', 'i7', 0.8),     # Pro-immigration alignment
    ('i1', 'i8', 0.5),     # Cultural vs national interests
    ('i1', 'i9', 0.9),     # Cultural benefits alignment
    ('i1', 'i10', 0.1),    # Immigration benefits vs threats
    ('i1', 'i11', 0.8),    # Family-friendly immigration
    ('i1', 'i12', 0.1),    # Open vs strict immigration
    ('i2', 'i4', 0.4),     # Skilled immigration vs border security
    ('i2', 'i5', 0.7),     # Immigration pathway alignment
    ('i2', 'i6', 0.8),     # Merit-based alignment
    ('i2', 'i7', 0.6),     # Immigration policy views
    ('i2', 'i9', 0.75),    # Cultural benefits alignment
    ('i2', 'i10', 0.2),    # Legal vs illegal immigration
    ('i2', 'i11', 0.7),    # Family vs skill priority
    ('i2', 'i12', 0.3),    # Welcoming vs strict enforcement

    # Environment Additional Pairs
    ('en1', 'en4', 0.5),   # Climate action vs economic balance
    ('en1', 'en5', 0.9),   # Climate action alignment
    ('en1', 'en6', 0.7),   # Clean energy alignment
    ('en1', 'en7', 0.8),   # Green policy alignment
    ('en1', 'en8', 0.1),   # Environmental protection vs jobs
    ('en1', 'en9', 0.1),   # Renewables vs fossil fuels
    ('en1', 'en10', 0.9),  # Clean energy transition
    ('en1', 'en11', 0.6),  # Climate action approaches
    ('en1', 'en12', 0.2),  # Government vs market solutions
    ('en2', 'en3', 0.1),   # Renewable transition vs regulation concerns
    ('en2', 'en5', 0.85),  # Clean energy alignment
    ('en2', 'en6', 0.75),  # Clean energy approaches
    ('en2', 'en7', 0.85),  # Green technology alignment
    ('en2', 'en8', 0.1),   # Environmental protection vs jobs
    ('en2', 'en9', 0.1),   # Renewables vs fossil fuels
    ('en2', 'en11', 0.65), # Environmental action alignment
    ('en2', 'en12', 0.3)   # Government vs market solutions
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
    learner.model.train()  # Set to training mode to get tensor output
    initial_sims = learner.get_similarity(x1, x2)
    initial_loss = learner.similarity_loss(initial_sims, similarities)  # No need to convert here anymore
    learner.model.eval()  # Set back to eval mode
    print(f"\nInitial loss before training: {initial_loss.item():.4f}")

# Train
learner.train(pairs, similarities, n_epochs=40)

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
    "foreign_cons_strong_4": "Expand nuclear arsenal",


    # Nuclear Policy
    'n1': 'Nuclear power is essential for a sustainable future',
    'n2': 'Nuclear weapons are a necessary deterrent',
    'n3': 'Nuclear disarmament is a global imperative',
    'n4': 'Nuclear energy should be replaced by renewables'
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


# Print the most extreme differences
print("\nMost changed similarities after learning:")
print_similarity_comparison(orig_matrix_new, learned_matrix_new, new_statement_ids, top_k=10)


import matplotlib.pyplot as plt
import seaborn as sns

def plot_similarity_matrices(orig_matrix, learned_matrix, statement_ids):
    """
    Plot original, learned, and difference similarity matrices with consistent colormap and formatting.
    """
    # Prepare more readable labels
    readable_labels = []
    for label in statement_ids:
        parts = label.split('_')
        # Determine position
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
    
    # Define consistent colormap and range
    cmap = 'RdYlBu_r'  # Reverse colormap: Red (low) to Blue (high)
    vmin, vmax, center = 0, 1, 0.5

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Plot original similarities
    sns.heatmap(orig_matrix, 
                cmap=cmap, vmin=vmin, vmax=vmax, center=center,
                xticklabels=readable_labels, yticklabels=readable_labels,
                cbar_kws={'label': 'Cosine Similarity'}, ax=axes[0])
    axes[0].set_title('Original Cosine Similarities')
    axes[0].tick_params(axis='x', rotation=45, labelsize=10)
    axes[0].tick_params(axis='y', rotation=0, labelsize=10)

    # Plot learned similarities
    sns.heatmap(learned_matrix, 
                cmap=cmap, vmin=vmin, vmax=vmax, center=center,
                xticklabels=readable_labels, yticklabels=readable_labels,
                cbar_kws={'label': 'Cosine Similarity'}, ax=axes[1])
    axes[1].set_title('Learned Similarities')
    axes[1].tick_params(axis='x', rotation=45, labelsize=10)
    axes[1].tick_params(axis='y', rotation=0, labelsize=10)

    # Compute and plot difference matrix
    diff_matrix = learned_matrix - orig_matrix
    sns.heatmap(diff_matrix, 
                cmap='RdBu_r', center=0, 
                xticklabels=readable_labels, yticklabels=readable_labels,
                cbar_kws={'label': 'Difference (Learned - Original)'}, ax=axes[2])
    axes[2].set_title('Difference Matrix')
    axes[2].tick_params(axis='x', rotation=45, labelsize=10)
    axes[2].tick_params(axis='y', rotation=0, labelsize=10)

    # Add a tight layout and show the plots
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