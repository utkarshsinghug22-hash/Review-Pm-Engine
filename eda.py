import pandas as pd
import matplotlib.pyplot as plt
import os

def run_eda():
    data_path = 'reviews.json'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_json(data_path)
    
    print(f"Loaded {len(df)} reviews.")
    
    # 1. Distribution of ratings
    print("\n--- Distribution of Ratings ---")
    rating_dist = df['rating'].value_counts().sort_index(ascending=False)
    print(rating_dist)
    
    plt.figure(figsize=(8, 5))
    rating_dist.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.savefig('rating_distribution.png')
    plt.close()
    
    # 2. Review volume over time (monthly)
    print("\n--- Review Volume Over Time ---")
    df['date'] = pd.to_datetime(df['date'])
    df['month_year'] = df['date'].dt.to_period('M')
    volume_over_time = df['month_year'].value_counts().sort_index()
    print("Latest 5 months of review volume:")
    print(volume_over_time.tail(5))
    
    plt.figure(figsize=(12, 6))
    # Convert period index to string for plotting if necessary, though pandas handles PeriodIndex well
    volume_over_time.plot(kind='line', marker='o', color='coral')
    plt.title('Review Volume Over Time')
    plt.xlabel('Month')
    plt.ylabel('Number of Reviews')
    plt.grid(True)
    plt.savefig('volume_over_time.png')
    plt.close()
    
    # 3. Average review length
    print("\n--- Average Review Length ---")
    df['review_length'] = df['text'].fillna('').apply(len)
    
    avg_length_overall = df['review_length'].mean()
    print(f"Overall Average Review Length: {avg_length_overall:.2f} characters")
    
    avg_length_by_rating = df.groupby('rating')['review_length'].mean()
    print("\nAverage Review Length by Rating:")
    print(avg_length_by_rating)
    
    plt.figure(figsize=(8, 5))
    avg_length_by_rating.plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.title('Average Review Length by Rating')
    plt.xlabel('Rating')
    plt.ylabel('Average Length (characters)')
    plt.xticks(rotation=0)
    plt.savefig('avg_length_by_rating.png')
    plt.close()
    
    print("\nPlots saved as rating_distribution.png, volume_over_time.png, and avg_length_by_rating.png.")

if __name__ == '__main__':
    run_eda()
