import pandas as pd

# Load the CSVs from the data folder
fake_df = pd.read_csv('data/Fake.csv', low_memory=False)
true_df = pd.read_csv('data/True.csv', low_memory=False)

# Add labels to both datasets
fake_df['label'] = 'FAKE'
true_df['label'] = 'REAL'

# Combine both datasets
news_df = pd.concat([fake_df, true_df], ignore_index=True)

# Save combined dataset
news_df.to_csv('data/news.csv', index=False)

print("news.csv created successfully in the 'data' folder.")
