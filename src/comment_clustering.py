import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ollama
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors



def get_embedding(text):
    """Get embedding using Ollama Python library."""
    response = ollama.embeddings(
        model='mxbai-embed-large',
        prompt=text
    )
    return np.array(response['embedding'])


politician_responses = [
    # Original responses
    "We must focus on successful integration first. Our social services are already strained, and we need to ensure current immigrants can access language courses and job training. Only after we've built robust integration systems should we consider expanding immigration.",
    
    "Integration is the key to social cohesion. I've seen in my constituency how unmanaged immigration without proper integration leads to parallel societies. We need to pause new arrivals and invest in making sure everyone already here can participate fully in our society.",
    
    "The numbers clearly show our integration programs are underfunded. As mayor of a diverse city, I've witnessed how proper integration leads to economic benefits. We must first strengthen these programs before accepting more immigrants.",
    
    "While immigration enriches our society, we need to balance this with successful integration. I support a measured approach where we temporarily reduce new arrivals while expanding our integration services and job training programs.",
    
    "Our current integration programs need improvement. However, this doesn't mean completely stopping new immigration - rather, we should scale back new arrivals while we strengthen our integration infrastructure.",
    
    "Both integration and controlled immigration are important for our economy. We should enhance our integration programs while maintaining current immigration levels, focusing on skilled workers who can contribute immediately.",
    
    "This is a complex issue that requires nuance. We need to simultaneously improve integration services while keeping our doors open to those who can contribute to our society. The two aspects aren't mutually exclusive.",
    
    "Our country's strength comes from continuous renewal through immigration. While integration is important, we can handle both simultaneously. Our economy needs new workers, and our integration programs are continuously improving.",
    
    "Integration and new immigration work hand in hand. My experience in the private sector shows that new immigrants often help integrate existing ones through community networks and job opportunities. We shouldn't create artificial barriers.",
    
    "This is a false choice. Our country was built by immigrants, and we have the capacity to both welcome newcomers and support integration. We need more workers in healthcare, agriculture, and tech - we can't afford to close our doors.",
    
    "Integration happens naturally when we provide opportunities. Having worked with immigrant communities for 20 years, I've seen how new arrivals actually strengthen integration by building vibrant communities and creating jobs. We must keep our doors open.",
    
    "We face a demographic crisis with an aging population. Rather than limiting immigration, we need both - better integration programs AND continued immigration. The data shows that second-generation immigrants are already well-integrated.",
    
    "The question oversimplifies a complex issue. In my region, some sectors desperately need workers, while some immigrant communities struggle with integration. We need targeted approaches for different situations.",
    
    "Based on my experience as an integration counselor, successful integration often depends more on economic opportunity than on specific programs. We should focus on job creation while maintaining steady immigration levels.",
    
    "The real issue isn't choosing between integration and new immigration - it's about resource allocation. We need to increase funding for both integration programs and border management while maintaining controlled immigration flows.",
    
    # New additional responses
    "Our rural communities are dying out. From my experience as a local councilor, controlled immigration combined with strong integration programs could revitalize these areas. We need both new arrivals and integration support.",
    
    "The cultural heritage of our nation must be preserved. We should completely halt new immigration until we can ensure all current immigrants have fully adapted to our values and way of life.",
    
    "As someone who immigrated here 30 years ago, I know firsthand that integration takes time and resources. But that doesn't mean stopping new arrivals - it means improving our systems to handle both effectively.",
    
    "Integration is failing because we're accepting too many people too quickly. I've seen my neighborhood completely change in 5 years. We need a total pause on immigration until we sort out the current situation.",
    
    "The statistics are clear - areas with higher immigration actually show better economic growth. Instead of limiting new arrivals, we should be learning from successful integration models in these regions.",
    
    "Having worked in education for 25 years, I've seen how children of immigrants thrive when given proper support. We need to maintain immigration while doubling down on educational integration programs.",
    
    "Our healthcare system depends on immigrant workers. Rather than choosing between integration and new immigration, we should focus on streamlining professional qualification recognition and language training.",
    
    "Small businesses in my district are crying out for workers. Yes, integration is important, but we can't solve our labor shortage without new immigration. We need to work on both simultaneously.",
    
    "Security must come first. Until we can properly screen and track everyone who's already here, we shouldn't accept any new immigrants. Integration includes ensuring public safety.",
    
    "The housing crisis is already severe - we need to pause immigration until we can ensure adequate housing for both current residents and immigrants. Integration must include access to affordable housing."

    # Variations on existing themes
    "As a former integration minister, I can tell you that our current programs are at breaking point. We need a complete pause on immigration for at least two years to rebuild our integration infrastructure.",
    
    "My research in urban sociology shows that successful integration happens organically in communities with balanced immigration rates. We should maintain current levels while improving support services.",
    
    "The housing shortage is just an excuse. Our construction sector needs immigrant workers to build more homes. The solution is more immigration, not less, combined with strategic urban planning.",
    
    "Integration isn't just about language and jobs. From my experience running community centers, we need to focus on cultural exchange and social mixing. Current programs are too focused on economics.",
    
    # New perspectives
    "Climate refugees are coming whether we like it or not. Instead of debating current immigration, we need to prepare our integration systems for the massive displacement that climate change will cause.",
    
    "Technology is the key. My startup has developed AI-powered language learning and job matching systems that could revolutionize integration. We need to modernize our approach, not limit numbers.",
    
    "The whole debate ignores the temporary worker perspective. We should separate permanent immigration from seasonal work programs. Different types of immigration need different integration approaches.",
    
    "Religious differences are the elephant in the room. Until we honestly address the challenges of integrating people with fundamentally different value systems, we're just avoiding the real issue.",
    
    "As a police chief, I've seen how poor integration leads to crime. But it's poverty, not culture, that's the problem. We need better economic integration programs, not immigration restrictions.",
    
    "The EU freedom of movement complicates this discussion. We can't talk about integration without addressing how EU migration affects our capacity to integrate non-EU immigrants.",
    
    "Look at the success of our sports programs - when young immigrants join local sports clubs, integration happens naturally. We need more funding for sports and cultural programs, not bureaucratic systems.",
    
    "The mental health of immigrants is completely ignored in this debate. As a psychiatrist working with refugee trauma, I can tell you that without mental health support, integration is impossible.",
    
    "Our schools are the front line of integration, but teachers are overwhelmed. We need smaller class sizes and more support staff before accepting more immigrant children.",
    
    "Integration starts before arrival. We should expand pre-immigration orientation programs in source countries, teaching language and culture before people arrive.",
    
    "The digital divide is a major barrier to integration. Many immigrants lack basic digital skills needed for modern jobs. We need to include digital literacy in all integration programs.",
    
    "Remote work has changed everything. We can now attract high-skilled immigrants to rural areas, solving both our brain drain and integration issues through geographic distribution.",
    
    "The underground economy undermines integration. We need stricter employment enforcement combined with easier work permit processes to prevent exploitation.",
    
    "Gender equality must be central to integration policy. Some immigrant women are isolated from society due to cultural barriers. We need specialized programs targeting this issue.",
    
    "Integration metrics are outdated. We measure language skills and employment but ignore social connections and civic participation. We need new ways to evaluate integration success.",
    
    "Local governments need more power over integration. A one-size-fits-all national policy ignores regional differences in labor markets and housing availability.",

        "Our tech industry's competitiveness is at stake. We're losing ground to other countries because our immigration policies are too restrictive. We need streamlined visas for tech talent and better integration support for their families.",
    
    "Small towns are dying while big cities are overcrowded. We should incentivize immigrants to settle in smaller communities through targeted integration programs and job guarantees.",
    
    "The informal economy is growing because our work permit system is too rigid. We need flexible immigration policies that match our labor market needs while ensuring proper worker protections.",

    # Education and Youth
    "Second-generation immigrant youth are caught between two worlds. Our integration programs need to specifically address identity and belonging, not just language and employment.",
    
    "University campuses should be integration hubs. We need to better connect international students with local communities and create pathways for them to stay after graduation.",
    
    "Early childhood education is key to integration success. We need universal pre-school programs that bring immigrant and local families together from the start.",

    # Security and Social Cohesion
    "Border control and integration are two sides of the same coin. Without knowing who's entering our country, we can't plan proper integration services.",
    
    "Social media creates parallel information worlds that hinder integration. We need digital literacy programs that help immigrants navigate our media landscape.",
    
    "Gang recruitment of immigrant youth is a real problem. But the solution isn't less immigration - it's better youth programs and community policing.",

    # Infrastructure and Planning
    "Smart city technology could revolutionize integration. We should use data analytics to better match immigrants with communities where they're most likely to succeed.",
    
    "Public transportation is an overlooked integration issue. Many immigrants can't access language classes or job opportunities because of poor transit connections.",
    
    "The suburbanization of immigration has caught us off guard. We need to help suburban communities develop integration infrastructure that cities built over decades.",

    # Healthcare and Social Services
    "The pandemic exposed gaps in our healthcare access for immigrants. We need universal health coverage regardless of immigration status to ensure public health.",
    
    "Social workers are overwhelmed by complex immigration cases. We need specialized training and more resources to handle intercultural family support.",
    
    "Traditional medicine and modern healthcare often clash in immigrant communities. We need cultural mediators in our healthcare system.",

    # Cultural and Religious Aspects
    "Food brings people together. We should support immigrant food entrepreneurs as a way to create cultural exchange and economic opportunity.",
    
    "Religious organizations are untapped integration partners. Instead of seeing religion as a barrier, we should work with faith communities to build bridges.",
    
    "Cultural festivals aren't enough. We need year-round intercultural dialogue programs that address real issues in our communities.",

    # Specific Solutions
    "Integration mentorship works. We should pair every new immigrant family with a local family for their first year.",
    
    "Language learning needs to happen in real-world settings. We should combine language classes with vocational training and community service.",
    
    "The private sector needs incentives to hire immigrants. Tax breaks for companies that provide language training and cultural integration support could help.",

    # Critical Perspectives
    "This whole debate ignores indigenous perspectives. We need to consider how immigration and integration policies affect First Nations communities.",
    
    "The integration industry has become a business. We're spending too much on consultants and not enough on direct support to immigrants.",
    
    "Integration programs often enforce conformity rather than celebrate diversity. We need to rethink what successful integration really means.",

    # Long-term Views
    "Climate change will force us to rethink everything. We need flexible integration systems that can handle large population movements.",
    
    "Artificial intelligence will eliminate many jobs that immigrants currently do. Our integration programs need to prepare for this technological shift.",
    
    "Virtual reality could transform integration training. Immigrants could experience local cultural situations before they even arrive.",

    # Practical Concerns
    "Housing discrimination undermines all other integration efforts. We need stronger enforcement of fair housing laws and support for immigrant homeownership.",
    
    "Immigrant entrepreneurs face too many barriers. Simplified business regulations and dedicated support services could help them create jobs for others.",
    
    "Financial literacy is crucial for integration. We need programs that help immigrants understand our banking system and build credit history.",

        # Similar to integration first/pause immigration views
    "As a city council member, I can see our integration services are at capacity. We must pause new immigration temporarily while we strengthen our support systems for those already here.",
    
    "The data from my district shows we're overwhelming our integration resources. Let's focus on helping current immigrants succeed before accepting more newcomers.",
    
    "Having worked in social services for 15 years, I see how stretched our integration programs are. We need to pause immigration briefly to rebuild our support infrastructure.",

    # Similar to economic benefit/labor shortage views
    "My chamber of commerce research shows critical worker shortages across industries. We need more immigration AND better integration programs to address this crisis.",
    
    "As a factory owner, I can tell you we're desperate for workers. Good integration programs combined with increased immigration is the only solution to our labor shortage.",
    
    "The agricultural sector is struggling to find workers. We need to maintain strong immigration flows while improving integration support - our food security depends on it.",

    # Similar to cultural/values perspectives
    "We need to protect our national identity and values. Immigration should be paused until we can ensure proper cultural integration of those already here.",
    
    "Cultural integration must come first. Our society is changing too rapidly - we need to slow immigration until we can properly integrate existing communities.",
    
    "Our traditional values are being eroded by too-rapid immigration. We should focus on cultural integration before accepting more newcomers.",

    # Similar to both/balanced approach views
    "Integration and immigration work together - we need both. My experience in community development shows that new arrivals often help integrate existing immigrants.",
    
    "We shouldn't choose between integration and immigration - both are essential. Our economy needs workers, and our communities need proper integration support.",
    
    "The solution is balance: maintain steady immigration while strengthening integration programs. One supports the other.",

    # Similar to housing crisis views
    "The rental market is overwhelmed in immigrant neighborhoods. We need to solve our housing shortage before increasing immigration levels.",
    
    "Housing affordability is the key integration challenge. Until we build more affordable homes, we should limit new immigration.",
    
    "My real estate studies show we lack housing for current residents. Immigration must be slowed until we can house people properly.",

    # Similar to education/schools perspectives
    "Our classrooms are overcrowded with students needing language support. We must limit immigration until schools can properly support integration.",
    
    "As another teacher, I've seen how essential school-based integration is. We need more resources for education before accepting more immigrant families.",
    
    "The education system is the foundation of integration. Let's strengthen our schools' integration programs before expanding immigration.",

    # Similar to healthcare system views
    "Our hospitals are struggling with language and cultural barriers. We need better healthcare integration programs before increasing immigration.",
    
    "The medical system needs more cultural competency training. Until we can provide proper healthcare to current immigrants, we should limit new arrivals.",
    
    "Healthcare integration is fundamental. We must improve medical services for current immigrants before accepting more.",

    # Similar to rural revival views
    "Like other rural mayors, I see immigration as key to revitalizing our small towns. We need targeted programs to attract immigrants to declining areas.",
    
    "Rural development depends on immigration. With proper integration support, immigrants can help save our dying small towns.",
    
    "The solution to rural decline is controlled immigration plus strong integration programs. Our small towns need new residents to survive.",

        # Very similar to "integration first" statements
    "Speaking as a social worker, I see daily how our integration services are overwhelmed. We need a temporary immigration pause to strengthen these vital support systems.",
    
    "The data is clear: our integration programs are at breaking point. As a policy researcher, I recommend pausing new arrivals until we can better serve those already here.",
    
    "My 20 years in immigrant services shows we're stretched too thin. Let's temporarily reduce immigration while we build stronger integration infrastructure.",
    
    # Very similar to pro-immigration economic views
    "The labor shortage is crippling our economy. As an economist, I can tell you we need more immigration, not less, along with robust integration support.",
    
    "Our manufacturing sector is desperate for workers. Immigration plus good integration programs is the only way to solve this growing economic crisis.",
    
    "Business growth is stalling due to worker shortages. We need to increase immigration while strengthening integration services - our economy depends on it.",
    
    # Very similar to cultural preservation views
    "As a cultural heritage officer, I believe we must pause immigration until current immigrants have fully embraced our societal values and traditions.",
    
    "Our national identity is at risk. We should halt new immigration temporarily while we focus on cultural integration of existing immigrant communities.",
    
    "The pace of cultural change is too rapid. Let's pause immigration until we can ensure proper cultural integration of those already here.",
    
    # Very similar to balanced approach statements
    "Based on my research, immigration and integration are complementary processes. We need both - new arrivals often help existing immigrants integrate better.",
    
    "As a community organizer, I see how immigration and integration support each other. We shouldn't reduce one to strengthen the other.",
    
    "My experience in immigrant services shows that steady immigration actually helps integration. We need to maintain both processes simultaneously.",
    
    # Very similar to rural development views
    "Our rural economy needs immigrants. With proper integration support, new arrivals could revitalize our declining small towns and villages.",
    
    "Small-town revitalization depends on immigration. Let's create targeted programs to attract and integrate immigrants in rural areas.",
    
    "As a rural business owner, I can tell you we need immigrants to survive. Good integration programs could help save our dying communities.",
    
    # Very similar to education-focused views
    "From my classroom experience, school-based integration is key. We need more educational resources before we can accept more immigrant students.",
    
    "The education system is at capacity with integration needs. Let's strengthen our schools' support systems before increasing immigration.",
    
    "As a school administrator, I see how integration strains our resources. We must improve educational support before accepting more immigrant families.",
    
    # Very similar to healthcare system views
    "Healthcare integration is at a breaking point. We need to improve our medical support systems before accepting more immigrants.",
    
    "Our hospital's integration services are overwhelmed. Let's focus on improving healthcare access for current immigrants before adding more.",
    
    "Medical integration must come first. As a healthcare worker, I say we need better systems before increasing immigration.",
    
    # Very similar to tech/modernization views
    "Digital solutions can transform integration. My tech company is developing AI tools to revolutionize how we support immigrants.",
    
    "Technology is the answer to our integration challenges. We need to embrace digital tools rather than limiting immigration.",
    
    "Smart technology can solve our integration problems. Let's use AI and digital platforms to support both current and new immigrants.",
        "We absolutely must prioritize integration first. Our current systems are overwhelmed and need strengthening before we accept more immigrants.",
    "Integration services are at their limit. We need to pause immigration while we improve support for existing immigrants.",
    "Let's focus on integration now. Only after our support systems are stronger should we consider accepting more immigrants.",
    "Our integration infrastructure needs work first. We should temporarily reduce immigration while we improve our support systems.",
    "Integration must be our priority. Current programs are stretched thin - let's strengthen them before expanding immigration.",

    # Strong economic/labor needs cluster
    "Worker shortages are hurting our economy. We need more immigration now, with integration support alongside.",
    "Our businesses can't grow without immigrants. We need more workers and can manage integration simultaneously.",
    "The labor crisis demands more immigration. We can handle integration while bringing in needed workers.",
    "Economic growth requires more immigrants. We can manage both integration and new arrivals effectively.",
    "Our economy needs immigrant workers now. Integration can happen alongside continued immigration.",

    # Cultural preservation cluster
    "Our cultural identity needs protection. Let's pause immigration until current immigrants are fully integrated.",
    "Cultural integration must come first. We need to slow immigration until existing communities adapt.",
    "Preserving our values requires a pause in immigration. Let's focus on integrating current residents.",
    "Cultural cohesion is essential. We should limit immigration until we achieve better integration.",
    "Our society needs time to integrate. Immigration should wait until cultural adaptation is successful.",

    # Balanced approach cluster
    "Integration and immigration work together perfectly. We need both processes to continue simultaneously.",
    "Both aspects strengthen each other. Immigration and integration should proceed hand in hand.",
    "We can manage both successfully. Immigration and integration support each other naturally.",
    "These processes are complementary. We need both immigration and integration working together.",
    "Neither should be sacrificed. Immigration and integration are equally important and mutually supportive.",

    # Rural development cluster
    "Rural areas desperately need immigrants. With good integration support, they can revive our small towns.",
    "Small communities need new residents. Immigration with proper integration can save rural areas.",
    "Rural revitalization requires immigrants. Good integration programs will help them settle successfully.",
    "Our small towns need immigration. Strong integration support will help revive rural communities.",
    "Rural survival depends on immigration. Proper integration programs will ensure successful settlement."

]

# Get embeddings for each response
embeddings = np.array([get_embedding(response) for response in politician_responses])


# 1. Scale the embeddings (recommended for DBSCAN)
scaler = StandardScaler()
scaled_embeddings = scaler.fit_transform(embeddings)

# Find optimal eps value
neighbors = NearestNeighbors(n_neighbors=2)
neighbors.fit(scaled_embeddings)
distances, _ = neighbors.kneighbors()

# Sort and plot distances to find the "elbow"
sorted_distances = np.sort(distances[:, 1])  # Distance to 5th nearest neighbor
plt.plot(sorted_distances)
plt.xlabel('Points')
plt.ylabel('Distance to 5th nearest neighbor')
plt.show()


# 2. Apply DBSCAN
dbscan = DBSCAN(
    eps=28,          # Maximum distance between samples to be considered neighbors
    min_samples=2,    # Minimum number of samples in a neighborhood to form a core point
    n_jobs=-1         # Use all CPU cores
)

# 3. Get cluster labels
labels = dbscan.fit_predict(scaled_embeddings)

# 4. Basic analysis
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Number of clusters
n_noise = list(labels).count(-1)                            # Number of noise points

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")

# 5. Get samples for each cluster
for cluster_id in set(labels):
    if cluster_id == -1:
        print(f"Noise points: {np.sum(labels == -1)}")
    else:
        print(f"Cluster {cluster_id}: {np.sum(labels == cluster_id)} points")

# Assuming you have:
# embeddings: your original embeddings
# labels: cluster labels from DBSCAN
def visualize_clusters(embeddings, labels, method='tsne'):
    """
    Visualize high-dimensional clusters using dimensionality reduction
    
    Parameters:
    - embeddings: numpy array of shape (n_samples, n_dimensions)
    - labels: cluster labels from DBSCAN
    - method: 'tsne' or 'umap'
    """
    # Convert labels to string for better plotting
    label_str = np.array(['Noise' if l == -1 else f'Cluster {l}' for l in labels])
    
    # Reduce to 2D
    if method == 'tsne':
        # Set perplexity to n_samples/3 (or minimum 5)
        n_samples = len(embeddings)
        perplexity = min(n_samples - 1, max(5, n_samples/3))
        reducer = TSNE(n_components=2, 
                      random_state=42,
                      perplexity=perplexity)  # Modified this line
    else:  # umap
        reducer = UMAP(n_components=2, 
                      random_state=42,
                      n_neighbors=min(15, len(embeddings)-1))  # Added this parameter
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with a different color for each cluster
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='tab20', alpha=0.6)
    
    # Add labels and title
    plt.title(f'Cluster Visualization using {method.upper()}')
    plt.xlabel(f'{method.upper()} Dimension 1')
    plt.ylabel(f'{method.upper()} Dimension 2')
    
    # Add legend
    legend1 = plt.legend(*scatter.legend_elements(),
                        title="Clusters",
                        loc="center left",
                        bbox_to_anchor=(1, 0.5))
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.show()

    # Print cluster sizes
    unique_labels = np.unique(labels)
    print("\nCluster Sizes:")
    for label in unique_labels:
        if label == -1:
            print(f"Noise points: {np.sum(labels == label)}")
        else:
            print(f"Cluster {label}: {np.sum(labels == label)} points")

# For interactive exploration, you can create a function to visualize specific clusters
def explore_cluster(embeddings, labels, cluster_id):
    """
    Detailed visualization of a specific cluster
    """
    mask = labels == cluster_id
    cluster_embeddings = embeddings[mask]
    
    # Calculate cluster statistics
    center = np.mean(cluster_embeddings, axis=0)
    distances = np.linalg.norm(cluster_embeddings - center, axis=1)
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Distance distribution
    plt.subplot(121)
    sns.histplot(distances, bins=30)
    plt.title(f'Distance Distribution in Cluster {cluster_id}')
    plt.xlabel('Distance from Cluster Center')
    
    # Plot 2: Points colored by distance from center
    plt.subplot(122)
    n_samples = len(cluster_embeddings)
    if n_samples > 1:  # Only do t-SNE if we have more than one point
        perplexity = min(n_samples - 1, max(5, n_samples/3))
        tsne = TSNE(n_components=2, 
                    random_state=42,
                    perplexity=perplexity)
        cluster_2d = tsne.fit_transform(cluster_embeddings)
        
        plt.scatter(cluster_2d[:, 0], cluster_2d[:, 1], 
                   c=distances, cmap='viridis')
        plt.colorbar(label='Distance from Center')
        plt.title(f'Cluster {cluster_id} Structure')
    else:
        plt.text(0.5, 0.5, 'Not enough points for visualization',
                ha='center', va='center')
    
    plt.tight_layout()
    plt.show()

# Example usage:
# Basic visualization
visualize_clusters(embeddings, labels, method='tsne')

# Alternative using UMAP (often better for large datasets)
visualize_clusters(embeddings, labels, method='umap')

# Print statements in each cluster
print("\nDetailed Cluster Contents:")
for cluster_id in sorted(set(labels)):
    print(f"\n{'='*50}")
    if cluster_id == -1:
        print(f"NOISE POINTS:")
    else:
        print(f"CLUSTER {cluster_id}:")
    print(f"{'='*50}")
    
    # Get indices of responses in this cluster
    cluster_indices = np.where(labels == cluster_id)[0]
    
    # Print each response with its index
    for idx in cluster_indices:
        print(f"\nResponse {idx}:")
        print(politician_responses[idx])
        print("-" * 30)

# Explore all clusters automatically
print("\nExploring all clusters...")
for cluster_id in sorted(set(labels)):
    if cluster_id != -1:  # Skip noise points
        print(f"\n{'#'*50}")
        print(f"Exploring Cluster {cluster_id}")
        print(f"{'#'*50}")
        explore_cluster(embeddings, labels, cluster_id)