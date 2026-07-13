import joblib
import pandas as pd

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from preprocessing import clean_text


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATA_PATH = DATA_DIR / "IMDB Dataset.csv"

df = pd.read_csv(DATA_PATH)

print(df.head())
print(df.shape)

df["clean_review"] = df["review"].apply(clean_text)

print(df[["review", "clean_review"]].head())


vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["clean_review"])

print(X.shape)


y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)


model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

print("Training Finished Successfully!")


from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)




print("Models Saved Successfully!")

from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(
    LogisticRegression(max_iter=1000, random_state=42),
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("Cross Validation Scores:", cv_scores)
print("Average Accuracy:", cv_scores.mean())


from sklearn.model_selection import GridSearchCV

param_grid = {
    "C": [0.1, 1, 10],
    "solver": ["liblinear", "lbfgs"]
}




grid_search = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000, random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)


grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross Validation Score:", grid_search.best_score_)


from sklearn.metrics import classification_report


best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print(cm)

joblib.dump(best_model, MODELS_DIR / "logistic_regression_model.pkl")

print("Best model saved successfully!")
