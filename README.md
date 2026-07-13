# 🎬 IMDB Movie Review Sentiment Analysis

A Machine Learning project that predicts whether an IMDB movie review is **Positive** or **Negative** using Natural Language Processing (NLP).

---

## 📌 Project Overview

This project uses:

- Python
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- FastAPI

The model is trained on the IMDB Movie Reviews dataset and can predict the sentiment of new reviews through a REST API.

---

## 📂 Project Structure

```
.
├── api.py
├── train.py
├── preprocessing.py
├── predict.py
├── requirements.txt
├── README.md
├── logistic_regression_model.pkl
├── tfidf_vectorizer.pkl
```

---

## 🚀 Features

- Text preprocessing
- TF-IDF Vectorization
- Logistic Regression model
- FastAPI REST API
- Real-time prediction

---

## 📊 Model Performance

Accuracy:

```
89.3%
```

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Run API

```bash
uvicorn api:app --reload
```

Then open:

```
http://127.0.0.1:8000/docs
```

---

## 📝 Example Request

```json
{
  "review": "This movie was amazing!"
}
```

Example Response

```json
{
  "prediction": "positive"
}
```

---

## 📚 Dataset

IMDB Movie Reviews Dataset

Download from Kaggle:

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

---

## 👨‍💻 Author

Mostafa Mohamed

Faculty of Computers and Information

Artificial Intelligence Department
