import pandas as pd
import numpy as np
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class FakeNewsDetector:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.model = None
        self.vectorizer = None
    
    def preprocess_text(self, text):
        """Clean and preprocess text data with advanced techniques"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and apply lemmatization
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def load_data(self, fake_path, true_path):
        """Load and preprocess the fake and true news datasets"""
        # Load datasets
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)
        
        # Add labels
        fake_df['label'] = 0
        true_df['label'] = 1
        
        # Check and select required columns
        if 'text' in fake_df.columns and 'text' in true_df.columns:
            # Both have 'text' column, good to go
            text_col = 'text'
        elif 'title' in fake_df.columns and 'text' in fake_df.columns and 'title' in true_df.columns and 'text' in true_df.columns:
            # Combine title and text for richer features
            fake_df['text'] = fake_df['title'] + ". " + fake_df['text']
            true_df['text'] = true_df['title'] + ". " + true_df['text']
            text_col = 'text'
        else:
            raise ValueError("Datasets must contain either 'text' column or both 'title' and 'text' columns")
        
        # Combine datasets
        combined_df = pd.concat([fake_df[[text_col, 'label']], true_df[[text_col, 'label']]], ignore_index=True)
        
        # Shuffle the data
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Preprocess text data
        print("Preprocessing text data...")
        combined_df['processed_text'] = combined_df[text_col].apply(self.preprocess_text)
        
        return combined_df
    
    def build_pipeline(self):
        """Build an optimized machine learning pipeline"""
        # TF-IDF Vectorizer with optimized parameters
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.85,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        
        # Logistic Regression with optimized parameters
        model = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            solver='liblinear',
            random_state=42
        )
        
        return model
    
    def train_model(self, data_df, test_size=0.2, use_grid_search=False, use_smote=True):
        """Train an optimized model with options for hyperparameter tuning and SMOTE"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            data_df['processed_text'], 
            data_df['label'], 
            test_size=test_size, 
            random_state=42, 
            stratify=data_df['label']
        )
        
        print(f"Training data size: {len(X_train)}")
        print(f"Testing data size: {len(X_test)}")
        
        # Vectorize the text data
        print("Vectorizing text data...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Apply SMOTE to handle class imbalance
        if use_smote:
            print("Applying SMOTE to balance classes...")
            smote = SMOTE(random_state=42)
            X_train_vec, y_train = smote.fit_resample(X_train_vec, y_train)
        
        # Create and train the model
        if use_grid_search:
            print("Performing grid search for hyperparameter tuning...")
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear', 'saga'],
                'class_weight': [None, 'balanced']
            }
            
            grid_search = GridSearchCV(
                LogisticRegression(max_iter=1000, random_state=42),
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_vec, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            self.model = self.build_pipeline()
            self.model.fit(X_train_vec, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_vec)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Print evaluation results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix
        }
    
    def save_model(self, model_path='models/fake_news_model.joblib'):
        """Save the trained model and vectorizer"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model and vectorizer must be trained before saving")
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and vectorizer
        joblib.dump({'model': self.model, 'vectorizer': self.vectorizer}, model_path)
        print(f"Model and vectorizer saved to {model_path}")
    
    def load_saved_model(self, model_path='models/fake_news_model.joblib'):
        """Load a saved model and vectorizer"""
        saved_data = joblib.load(model_path)
        self.model = saved_data['model']
        self.vectorizer = saved_data['vectorizer']
    
    def predict(self, text):
        """Predict whether a given news article is fake or real with improved confidence scoring"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model is not trained yet. Please train the model first.")
        
        # Preprocess the input text
        processed_text = self.preprocess_text(text)
        
        # Vectorize the input
        text_vec = self.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.model.predict(text_vec)[0]
        
        # Get prediction probabilities for better confidence scoring
        probabilities = self.model.predict_proba(text_vec)[0]
        confidence = probabilities[1] if prediction == 1 else probabilities[0]
        
        # Calculate confidence margin (difference between two class probabilities)
        confidence_margin = abs(probabilities[1] - probabilities[0])
        
        # Return result with enhanced confidence information
        result = "Real" if prediction == 1 else "Fake"
        
        return {
            'prediction': result,
            'confidence': float(confidence),
            'confidence_margin': float(confidence_margin),
            'raw_probabilities': probabilities.tolist(),
            'processed_text': processed_text
        }

def main():
    """Main function to train and save a model"""
    detector = FakeNewsDetector()
    
    try:
        # Load data
        fake_path = 'data/Fake.csv'
        true_path = 'data/True.csv'
        df = detector.load_data(fake_path, true_path)
        
        # Train model with optimizations
        metrics = detector.train_model(df, use_grid_search=True, use_smote=True)
        
        # Save the model
        detector.save_model()
        
        # Example prediction
        example_text = "Breaking: Scientists discover that climate change is a hoax perpetrated by the government"
        prediction = detector.predict(example_text)
        print(f"\nExample prediction for: '{example_text}'")
        print(f"Prediction: {prediction['prediction']} (Confidence: {prediction['confidence']:.2f})")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 