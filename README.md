<<<<<<< HEAD
# Fake News Detection Using Machine Learning

This project implements a machine learning pipeline to classify news articles as real or fake based on their content. It uses text preprocessing, TF-IDF vectorization, and various classification models to achieve this goal.

## Features

- Text preprocessing (lowercase, remove special characters, lemmatization)
- TF-IDF vectorization with n-grams
- Multiple model support (Logistic Regression, Naive Bayes, SVM)
- Hyperparameter tuning using GridSearchCV
- Model evaluation metrics (accuracy, precision, recall, F1)
- REST API for predictions
- Model persistence

## Project Structure

```
fake-news-detection/
│
├── data/                # For your dataset (CSV)
├── models/              # For saving trained models
├── app.py              # Flask API
├── main.py             # Main ML pipeline
├── requirements.txt    # Dependencies
└── README.md
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your dataset in the `data` folder as `news.csv` with columns:
   - `text`: The news article text
   - `label`: 1 for real news, 0 for fake news

## Usage

### Training the Model

Run the main script to train and evaluate the model:
```bash
python main.py
```

This will:
- Load and preprocess the data
- Split into train/test sets
- Train the model (default: Logistic Regression)
- Evaluate performance
- Save the trained model

### Using the API

Start the Flask server:
```bash
python app.py
```

Make predictions using the API:
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Your news article text here"}'
```

## Model Selection

The code supports three models:
- Logistic Regression (`logreg`)
- Naive Bayes (`nb`)
- Support Vector Machine (`svm`)

To use a different model, modify the `model_type` parameter in `main.py`.

## Evaluation Metrics

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score

## Contributing

Feel free to submit issues and enhancement requests! 
=======
# Fake-News-Detection
>>>>>>> 850ee786c73fdc7a1d243eb8d5f6dd1bf084ba7b
