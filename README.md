# 🎬 IMDB Movie Sentiment Analysis

A Natural Language Processing (NLP) Machine Learning project that classifies IMDB movie reviews into Positive or Negative sentiment.

This project implements a complete NLP pipeline including:

- Text preprocessing and cleaning
- Feature extraction using TF-IDF
- Machine Learning model training
- Hyperparameter tuning
- Model evaluation
- Saving trained models
- Prediction system
- FastAPI deployment


# 📌 Project Overview

Sentiment Analysis is one of the most important Natural Language Processing tasks that focuses on understanding the emotional meaning behind text.

In this project, an automated sentiment classification system was developed to analyze movie reviews and predict whether the review expresses a positive or negative sentiment.

The project was developed in two stages:

1. Experimentation & Research
- Performed on Kaggle
- Data exploration
- NLP experiments
- Model training and evaluation

2. Production Implementation
- Clean Python project structure
- Saved trained models
- Prediction pipeline
- FastAPI API


# 📊 Dataset

Dataset:

IMDB Dataset of 50K Movie Reviews

The dataset contains:

- 50,000 movie reviews
- Two sentiment classes:
  - Positive
  - Negative


Dataset Columns:

| Column | Description |
|---|---|
| review | Movie review text |
| sentiment | Sentiment label |


# 📓 Kaggle Notebook

The complete experimentation, analysis, and model development process was performed on Kaggle.

The notebook includes:

- Exploratory Data Analysis (EDA)
- Sentiment distribution analysis
- Text preprocessing
- Feature engineering
- TF-IDF vectorization
- Model training
- Model evaluation


Kaggle Notebook:

https://www.kaggle.com/code/mostafa312/sentiment-analysis-nlp


The GitHub repository contains the production-ready implementation:

- Clean Python scripts
- Saved ML models
- Prediction pipeline
- FastAPI deployment


# 🧹 Text Preprocessing Pipeline

The following NLP preprocessing steps were applied:

- Removing HTML tags
- Converting text to lowercase
- Removing numbers
- Removing punctuation
- Removing English stopwords


Example:

Before:

This movie was AMAZING!!! <br /> 10/10


After:

movie amazing


# 🔤 Feature Extraction

## TF-IDF Vectorization

TF-IDF was used to convert text data into numerical vectors.

Final feature matrix:

50000 Reviews

162079 Features


TF-IDF helps the machine learning model understand the importance of words inside reviews.


# 🤖 Machine Learning Model

## Logistic Regression

The main classification model used:

Logistic Regression


Reasons for choosing Logistic Regression:

- Excellent performance on text classification problems
- Efficient with high-dimensional sparse data
- Fast training
- Strong baseline for NLP tasks


# ⚙️ Hyperparameter Optimization

Used:

- GridSearchCV
- Cross Validation


Best Parameters:

C = 10

solver = liblinear


# 📈 Model Performance


## Accuracy

89.61%


## Cross Validation Score

Average Accuracy: 89.55%


## Classification Report

Negative:

Precision: 0.90

Recall: 0.90

F1-score: 0.90


Positive:

Precision: 0.90

Recall: 0.91

F1-score: 0.90


## Confusion Matrix

[[4492 508]

[475 4525]]


# 📁 Project Structure

IMDB-Movie-Sentiment-Analysis/

│

├── data/

│   └── IMDB Dataset.csv

│

├── models/

│   ├── logistic_regression_model.pkl

│   └── tfidf_vectorizer.pkl

│

├── src/

│   ├── train.py

│   ├── preprocessing.py

│   ├── predict.py

│   └── api.py

│

├── requirements.txt

├── .gitignore

└── README.md



# ⚙️ Installation

Clone the repository:

git clone <repository-url>


Install dependencies:

pip install -r requirements.txt



# 🚀 Run Prediction

Run:

python src/predict.py


Example review:

This movie was amazing and very enjoyable


Output:

Prediction: positive



# 🌐 FastAPI Deployment

The project includes a REST API using FastAPI.


Run API:

uvicorn src.api:app --reload


API Endpoint:

POST /predict


Example Request:

{
 "review": "This movie was amazing"
}


Example Response:

{
 "prediction": "positive"
}



# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TF-IDF
- Logistic Regression
- GridSearchCV
- FastAPI
- Joblib
- Kaggle



# 🔮 Future Improvements

- Word2Vec embeddings
- GloVe embeddings
- LSTM / GRU models
- Transformer models
- BERT fine-tuning
- Docker deployment
- Web application interface



# 👨‍💻 Author

Mostafa Mohamed

Artificial Intelligence 


Focused on:

- Machine Learning
- Natural Language Processing
- Data Science
- AI Engineering
