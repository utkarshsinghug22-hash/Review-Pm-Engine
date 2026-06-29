import json
import random

with open('reviews.json', encoding='utf-8') as f:
    data = json.load(f)

valid = [d for d in data if d.get('text') and len(d['text']) > 30]
random.seed(42)
sample = random.sample(valid, 20)

for i, s in enumerate(sample):
    print(f"[{i}] Rating {s.get('rating')} (Thumbs: {s.get('thumbsUp', 0)}): {s.get('text')}")
