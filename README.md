# App Review Scraper & Product Strategy Engine

**"What should we build next?"** 

This project is a complete Product Management pipeline designed to answer that question using real data. It acts as an end-to-end case study on how to translate raw user feedback into actionable product strategy.

## 🚀 The Product Management Loop

This repository automates the core phases of a data-driven product workflow:

1. **User Research (Data Extraction):** 
   - A Node.js scraper that extracts thousands of recent user reviews from the Google Play Store and Apple App Store.
2. **Data Analysis (NLP & Machine Learning):** 
   - Uses HuggingFace's zero-shot classification (`facebook/bart-large-mnli`) and sentiment analysis to categorize reviews into Bug Reports, Feature Requests, and Complaints. 
   - Utilizes `sentence-transformers` and KMeans clustering to semantically group similar feature requests into overarching themes.
3. **Prioritization (The RICE Framework):** 
   - Applies the mathematical RICE scoring model (Reach × Impact × Confidence ÷ Effort) to rank the feature clusters based on user pain points and review volume. 
4. **The Deliverable (Dashboard & PRD):** 
   - An interactive **Streamlit Dashboard** that visualizes these insights.
   - The dashboard includes a definitive PM recommendation on what to ship next, complete with a 1-page mini-PRD to solve the #1 user pain point.

## 📊 View the Dashboard Live

You can explore the interactive dashboard here: **[Live Demo: App Review Strategy Engine](https://review-pm-engine-kbcbjurr9ihfup6akgffdd.streamlit.app/)**

*Note: The dashboard features a dynamic Review Explorer, category breakdowns, rating trends, and the final RICE-ranked feature cards.*

## 💻 Running it Locally

**1. Scrape the Data**
```bash
npm install
npm start
```

**2. Run the Machine Learning Pipeline**
```bash
pip install -r requirements.txt
python classify_reviews.py
python cluster_features.py
```

**3. Launch the PM Dashboard**
```bash
python -m streamlit run app.py
```

## 👋 About the Author
I am a curious, analytical, and impact-driven builder deeply passionate about consumer technology and product management. I built this engine to showcase my ability to dive into unstructured data, find the signal in the noise, and translate insights into clear product requirements.
