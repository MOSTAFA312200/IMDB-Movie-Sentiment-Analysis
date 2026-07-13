
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
model = joblib.load(MODELS_DIR / "logistic_regression_model.pkl")




from preprocessing import clean_text





review = input("Enter your movie review: ")

cleaned_review = clean_text(review)

print("Original Review:")
print(review)

print("\nCleaned Review:")
print(cleaned_review)

review_vector = vectorizer.transform([cleaned_review])

print(review_vector.shape)

prediction = model.predict(review_vector)

print(prediction)




