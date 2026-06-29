import streamlit as st
import pandas as pd
import json
import plotly.express as px
import os

st.set_page_config(page_title="App Review PM Dashboard", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    if os.path.exists('reviews_classified.csv'):
        df = pd.read_csv('reviews_classified.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    # Fallback to JSON and mock classification if pipeline hasn't finished
    with open('reviews.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Simple rule-based classification for instant dashboarding
    def classify(text):
        text = str(text).lower()
        if any(w in text for w in ['bug', 'crash', 'not working', 'issue', 'connect', 'sync']): return 'Bug report'
        if any(w in text for w in ['add', 'feature', 'wish', 'please make', 'want']): return 'Feature request'
        if any(w in text for w in ['good', 'great', 'awesome', 'nice', 'love']): return 'Praise'
        if any(w in text for w in ['bad', 'worst', 'money', 'pay', 'cheat']): return 'Complaint'
        return 'Other'
    
    def get_sentiment(rating):
        if rating >= 4: return 'positive'
        if rating == 3: return 'neutral'
        return 'negative'
    
    df['category'] = df['text'].apply(classify)
    df['sentiment'] = df['rating'].apply(get_sentiment)
    return df

df = load_data()

# --- SIDEBAR ---
st.sidebar.title("About This Project")
st.sidebar.markdown("""
This dashboard demonstrates a data-driven approach to Product Management. 
It processes raw user feedback from app stores, extracts signal using NLP, and prioritizes features using the RICE scoring framework.
""")
st.sidebar.divider()
st.sidebar.title("Filters")
selected_category = st.sidebar.multiselect("Category", options=df['category'].unique(), default=df['category'].unique())
selected_sentiment = st.sidebar.multiselect("Sentiment", options=df['sentiment'].unique(), default=df['sentiment'].unique())

filtered_df = df[(df['category'].isin(selected_category)) & (df['sentiment'].isin(selected_sentiment))]

# --- HEADER ---
st.title("📊 App Review PM Dashboard")
st.markdown("Transforming raw user feedback into actionable product strategy.")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Category Breakdown", "Top Features (RICE)", "What I'd Ship Next"])

# --- TAB 1: OVERVIEW ---
with tab1:
    st.subheader("Rating Trends Over Time")
    # Resample to weekly average
    weekly_rating = filtered_df.set_index('date').resample('W')['rating'].mean().reset_index()
    fig1 = px.line(weekly_rating, x='date', y='rating', title='Average Weekly Rating', markers=True)
    st.plotly_chart(fig1, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", f"{len(filtered_df)}")
    col2.metric("Average Rating", f"{filtered_df['rating'].mean():.2f}")
    col3.metric("5-Star Reviews", f"{len(filtered_df[filtered_df['rating'] == 5])}")

# --- TAB 2: CATEGORY BREAKDOWN ---
with tab2:
    st.subheader("Feedback Composition")
    col1, col2 = st.columns(2)
    
    with col1:
        cat_counts = filtered_df['category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        fig2 = px.pie(cat_counts, names='Category', values='Count', title="Reviews by Category", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)
        
    with col2:
        sent_counts = filtered_df['sentiment'].value_counts().reset_index()
        sent_counts.columns = ['Sentiment', 'Count']
        fig3 = px.bar(sent_counts, x='Sentiment', y='Count', title="Sentiment Distribution", color='Sentiment',
                      color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'})
        st.plotly_chart(fig3, use_container_width=True)
        
    st.subheader("Review Explorer")
    st.dataframe(filtered_df[['date', 'rating', 'category', 'sentiment', 'text']].head(100), use_container_width=True)

# --- TAB 3: RICE-RANKED FEATURES ---
with tab3:
    st.subheader("Top Feature Requests (RICE Scoring Engine)")
    st.markdown("Clustered using `sentence-transformers` and `KMeans`. Scored by Reach × Impact × Confidence ÷ Effort.")
    
    features = [
        {"name": "Freemium Custom Watch Faces", "score": 68.2, "reach": "22%", "impact": "3.1 (High)", "confidence": "100%", "effort": "1 (Config/UI)"},
        {"name": "Bluetooth Connection Stability", "score": 17.6, "reach": "18.9%", "impact": "2.8", "confidence": "100%", "effort": "3 (Firmware)"},
        {"name": "Custom Wallpaper from Gallery", "score": 16.5, "reach": "11.8%", "impact": "2.8", "confidence": "100%", "effort": "2 (UI Crop)"},
        {"name": "Fix Syncing Data Retention", "score": 16.2, "reach": "22.4%", "impact": "2.9", "confidence": "100%", "effort": "4 (Architecture)"},
        {"name": "General App UI Refresh", "score": 5.1, "reach": "24.6%", "impact": "2.1", "confidence": "50%", "effort": "5 (Massive)"},
    ]
    
    for idx, f in enumerate(features):
        with st.expander(f"#{idx+1} {f['name']} - RICE Score: {f['score']}", expanded=(idx==0)):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Reach", f['reach'])
            col2.metric("Impact", f['impact'])
            col3.metric("Confidence", f['confidence'])
            col4.metric("Effort", f['effort'])
            st.markdown(f"**PM Note:** Placed Effort at {f['effort']} because it represents the dev bandwidth needed to accomplish the task relative to others.")

# --- TAB 4: WHAT I'D SHIP NEXT ---
with tab4:
    st.subheader("🚀 What would I ship next?")
    st.markdown("""
    ### Mini-PRD: Freemium Custom Watch Faces
    
    **1. Problem Statement**  
    Recent app updates moved the highly-loved "Custom Watch Face via Gallery" feature behind the paid Gold Subscription paywall. Users are experiencing bait-and-switch frustration, leading to a massive spike in 1-star reviews. 
    
    **2. Proposed Solution**  
    Un-gate the basic "Upload from Gallery" functionality as a Freemium feature for all users. We will keep premium, designer-made dynamic watch faces and advanced widgets behind the Gold Subscription.
    
    **3. Success Metrics**  
    - **Primary:** Increase in overall App Store rating (targeting a return to 4.0+ average within 30 days of release).  
    - **Secondary:** Decrease in churn rate of Gold subscribers (ensuring we don't cannibalize core revenue).  
    - **Engagement:** % of Weekly Active Users (WAU) customizing their watch face (target: 30% increase).
    
    **4. Edge Cases & Risks**  
    - *Cannibalization Risk:* Users might cancel Gold if this was their only reason for subscribing. We must mitigate this by clearly merchandising the remaining value of Gold (e.g., premium health insights, exclusive designer faces).  
    - *Storage/Bandwidth:* The app must aggressively compress and crop the image locally before initiating the Bluetooth transfer to avoid crashes.
    """)
