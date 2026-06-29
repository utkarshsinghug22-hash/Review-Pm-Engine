import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

print("Loading reviews.json...")
with open('reviews.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Fallback filtering to find Feature Requests using keywords
feature_keywords = ['add', 'feature', 'want', 'wish', 'please make', 'need', 'option', 'suggest', 'would be great', 'implement']
feature_df = df[df['text'].str.contains('|'.join(feature_keywords), case=False, na=False)].copy()

print(f"Found {len(feature_df)} feature requests (keyword matched).")
if len(feature_df) == 0:
    print("Not enough data to cluster.")
    exit(1)

# Embed
print("Loading sentence-transformer (MiniLM)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = feature_df['text'].fillna("").tolist()

print("Encoding texts...")
embeddings = model.encode(texts, show_progress_bar=False)

# KMeans Clustering
n_clusters = 5 # Top 5 features
print(f"Clustering into {n_clusters} clusters...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
feature_df['cluster'] = kmeans.fit_predict(embeddings)

# Output cluster info
print("=== CLUSTER RESULTS ===")
for i in range(n_clusters):
    cluster_data = feature_df[feature_df['cluster'] == i]
    size = len(cluster_data)
    
    # Calculate Impact proxy (e.g. proxy from ratings since sentiment analysis pipeline isn't complete)
    # Average rating could be a reverse proxy for impact: 1-star means high impact (pain).
    avg_rating = cluster_data['rating'].mean()
    
    # Get representative quotes (closest to cluster center)
    center = kmeans.cluster_centers_[i]
    cluster_embeddings = embeddings[feature_df['cluster'] == i]
    # Calculate distances to center
    distances = np.linalg.norm(cluster_embeddings - center, axis=1)
    # Get top 3 closest
    top_indices = np.argsort(distances)[:3]
    
    # Get indices in original df
    global_indices = cluster_data.iloc[top_indices].index
    quotes = cluster_data.loc[global_indices, 'text'].tolist()
    
    print(f"\nCluster {i} | Volume (Confidence Proxy): {size} | Avg Rating (Impact Proxy): {avg_rating:.1f}")
    for q in quotes:
        print(f"  - {q.strip()}")
