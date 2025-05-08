import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import joblib
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """Clean and preprocess text data."""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def load_data(self, path):
        """Load and preprocess the dataset."""
        print("Loading data...")
        df = pd.read_csv(path)
        # Assuming columns: 'text', 'label' (1=real, 0=fake)
        df = df[['text', 'label']].dropna()
        df['text'] = df['text'].apply(self.preprocess_text)
        return df

    def split_data(self, df, test_size=0.2, random_state=42):
        """Split data into train and test sets."""
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['label'], 
            test_size=test_size, 
            random_state=random_state, 
            stratify=df['label']
        )
        return X_train, X_test, y_train, y_test

    def vectorize_text(self, X_train, X_test):
        """Convert text to TF-IDF features."""
        print("Vectorizing text...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        return X_train_vec, X_test_vec

    def train_model(self, X_train, y_train, model_type='logreg'):
        """Train a classifier with hyperparameter tuning."""
        print(f"Training {model_type} model...")
        
        if model_type == 'logreg':
            param_grid = {
                'C': [0.1, 1, 10],
                'max_iter': [1000]
            }
            base_model = LogisticRegression()
        elif model_type == 'nb':
            param_grid = {
                'alpha': [0.1, 0.5, 1.0]
            }
            base_model = MultinomialNB()
        elif model_type == 'svm':
            param_grid = {
                'C': [0.1, 1, 10],
                'max_iter': [1000]
            }
            base_model = LinearSVC()
        else:
            raise ValueError("model_type must be 'logreg', 'nb', or 'svm'")

        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=5, 
            scoring='f1',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        return self.model

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance."""
        print("\nEvaluating model...")
        y_pred = self.model.predict(X_test)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        print("\nDetailed Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        return metrics

    def save_model(self, model_path='models/fake_news_model.joblib'):
        """Save the trained model and vectorizer."""
        print("Saving model...")
        joblib.dump({
            'model': self.model,
            'vectorizer': self.vectorizer
        }, model_path)

    def load_saved_model(self, model_path='models/fake_news_model.joblib'):
        """Load a saved model and vectorizer."""
        print("Loading saved model...")
        saved_data = joblib.load(model_path)
        self.model = saved_data['model']
        self.vectorizer = saved_data['vectorizer']

def main():
    # Create necessary directories
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Initialize detector
    detector = FakeNewsDetector()
    
    # Load and process data
    df = detector.load_data('data/news.csv')
    X_train, X_test, y_train, y_test = detector.split_data(df)
    X_train_vec, X_test_vec = detector.vectorize_text(X_train, X_test)
    
    # Train model (try different models: 'logreg', 'nb', 'svm')
    detector.train_model(X_train_vec, y_train, model_type='logreg')
    
    # Evaluate model
    detector.evaluate_model(X_test_vec, y_test)
    
    # Save model
    detector.save_model()

if __name__ == "__main__":
    main() 