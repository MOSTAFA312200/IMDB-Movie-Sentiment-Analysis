from fastapi import  FastAPI
import joblib
from pathlib import Path
from pydantic import BaseModel
from src.preprocessing import clean_text

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
model = joblib.load(MODELS_DIR / "logistic_regression_model.pkl")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "IMDB Sentiment Analysis API is running!"}

class ReviewRequest(BaseModel):
    review: str


@app.post("/predict")
def predict(request: ReviewRequest):

    cleaned_review = clean_text(request.review)

    review_vector = vectorizer.transform([cleaned_review])

    prediction = model.predict(review_vector)[0]

    return {
        "original_review": request.review,
        "cleaned_review": cleaned_review,
        "prediction": prediction
    }

