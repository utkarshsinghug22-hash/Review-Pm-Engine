import json
import pandas as pd
import torch
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from tqdm import tqdm
import re

# Load data
with open('reviews.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# High-signal tagging
df['is_high_signal'] = df['thumbsUp'] >= 10

# Initialize Pipelines
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {device}")

print("Loading zero-shot classifier...")
classifier = pipeline("zero-shot-classification", 
                      model="facebook/bart-large-mnli", 
                      device=device)
candidate_labels = ["Bug report", "Feature request", "Praise", "Complaint", "Other"]

print("Loading sentiment analyzer...")
# cardiffnlp/twitter-roberta-base-sentiment has labels: LABEL_0 (negative), LABEL_1 (neutral), LABEL_2 (positive)
sentiment_analyzer = pipeline("sentiment-analysis", 
                              model="cardiffnlp/twitter-roberta-base-sentiment", 
                              device=device)
sentiment_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}

# Helper function to clean text slightly
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove emojis and special chars for TF-IDF
    return re.sub(r'[^\w\s]', '', text.lower())

texts = df['text'].fillna("").tolist()

# To save time in this demonstration, if the dataset is > 1000, we'll take a sample
# to ensure it completes within a reasonable timeframe (BART is heavy on CPU).
if len(texts) > 500:
    print(f"Dataset has {len(texts)} reviews. Sampling 500 for demonstration to avoid excessive compute time.")
    df = df.sample(500, random_state=42).reset_index(drop=True)
    texts = df['text'].fillna("").tolist()

print(f"Processing {len(texts)} reviews...")
categories = []
sentiments = []

for text in tqdm(texts, desc="Classifying & Sentiment"):
    if not text.strip():
        categories.append("Other")
        sentiments.append("neutral")
        continue
        
    try:
        # Category
        res_cat = classifier(text, candidate_labels)
        categories.append(res_cat['labels'][0])
        
        # Sentiment
        # Truncate text for sentiment model to avoid length errors
        res_sent = sentiment_analyzer(text[:512])
        label = res_sent[0]['label']
        sentiments.append(sentiment_map.get(label, "neutral"))
    except Exception as e:
        print(f"Error on text: {text[:50]}... -> {e}")
        categories.append("Other")
        sentiments.append("neutral")

df['category'] = categories
df['sentiment'] = sentiments

# TF-IDF Keyword Extraction per category
print("Extracting keywords using TF-IDF...")
df['clean_text'] = df['text'].apply(clean_text)

weighted_corpus = []
for idx, row in df.iterrows():
    text = row['clean_text']
    weight = 3 if row['is_high_signal'] else 1
    weighted_corpus.append((text + " ") * weight)

df['weighted_text'] = weighted_corpus

# Extract top keywords per category
category_keywords = {}
for cat in candidate_labels:
    cat_df = df[df['category'] == cat]
    if len(cat_df) == 0:
        category_keywords[cat] = ""
        continue
        
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
    try:
        tfidf_matrix = vectorizer.fit_transform(cat_df['weighted_text'])
        sum_words = tfidf_matrix.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        top_words = [word for word, count in words_freq[:10]]
        category_keywords[cat] = ", ".join(top_words)
    except ValueError:
        category_keywords[cat] = "" 

df['category_top_keywords'] = df['category'].map(category_keywords)

print("Keyword mapping per category:")
for cat, kw in category_keywords.items():
    print(f"  {cat}: {kw}")

df = df.drop(columns=['clean_text', 'weighted_text'])

output_file = 'reviews_classified.csv'
df.to_csv(output_file, index=False)
print(f"Saved results to {output_file}")
