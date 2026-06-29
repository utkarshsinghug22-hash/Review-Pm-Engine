import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

print("Loading reviews_classified.csv...")
try:
    df = pd.read_csv('reviews_classified.csv')
except FileNotFoundError:
    print("reviews_classified.csv not found. Did you run classify_reviews.py?")
    exit(1)

# Filter feature requests. Sometimes classification isn't perfect, so we might want to also include 'Other' 
# or just stick to 'Feature request'
# For the sake of having enough data, we might also filter by keywords. Let's stick to the classified label.
feature_df = df[df['category'] == 'Feature request'].copy()
if feature_df.empty:
    print("No feature requests found in classification. Falling back to simple keyword matching.")
    feature_keywords = ['add', 'feature', 'want', 'wish', 'please make', 'need', 'option']
    feature_df = df[df['text'].str.contains('|'.join(feature_keywords), case=False, na=False)].copy()

print(f"Found {len(feature_df)} feature requests.")
if len(feature_df) == 0:
    print("Not enough data to cluster.")
    exit(1)

# Embed
print("Loading sentence-transformer...")
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = feature_df['text'].fillna("").tolist()

print("Encoding texts...")
embeddings = model.encode(texts, show_progress_bar=True)

# KMeans Clustering
n_clusters = min(10, len(texts))
print(f"Clustering into {n_clusters} clusters...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
feature_df['cluster'] = kmeans.fit_predict(embeddings)

# Output cluster info
print("=== CLUSTER RESULTS ===")
for i in range(n_clusters):
    cluster_data = feature_df[feature_df['cluster'] == i]
    size = len(cluster_data)
    
    # Calculate Impact proxy (e.g. percentage of negative sentiment)
    # If the feature is highly requested and people are negative about it missing, impact is high.
    sentiments = cluster_data['sentiment'].value_counts(normalize=True)
    neg_pct = sentiments.get('negative', 0) * 100
    
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
    
    print(f"\nCluster {i} | Size: {size} | % Negative: {neg_pct:.1f}%")
    for q in quotes:
        print(f"  - {q.strip()}")
