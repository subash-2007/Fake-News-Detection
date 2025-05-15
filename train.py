import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from main import FakeNewsDetector

# Load dataset
df = pd.read_csv('data/news.csv')  # Adjust path to your dataset
X = df['text']
y = df['label'].map({'REAL': 1, 'FAKE': 0})  # Ensure labels are 0 and 1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize components
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Save model and vectorizer
os.makedirs('models', exist_ok=True)
joblib.dump({'model': model, 'vectorizer': vectorizer}, 'models/fake_news_model.joblib')

print("✅ Model and vectorizer saved to 'models/fake_news_model.joblib'")
