import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

from config import (
    DATA_PATH,
    TEXT_COLUMN,
    LABEL_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    LOGISTIC_MODEL_PATH,
    SVM_MODEL_PATH,
    RESULTS_DIR
)

from preprocessing import clean_arabic_text


# =========================
# LOAD AND CLEAN DATA
# =========================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)
df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

df["clean_text"] = df[TEXT_COLUMN].apply(clean_arabic_text)
df = df[df["clean_text"].str.strip() != ""]
df = df.drop_duplicates(subset=["clean_text"])

X = df["clean_text"]
y = df[LABEL_COLUMN]

print("Final dataset shape:", df.shape)
print("\nClass distribution:")
print(y.value_counts())


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# =========================
# LOGISTIC REGRESSION
# =========================

print("\nTraining Logistic Regression...")

logistic_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=50000
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

logistic_model.fit(X_train, y_train)
logistic_pred = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(y_test, logistic_pred)

print("\nLogistic Regression Accuracy:", logistic_accuracy)
print(classification_report(y_test, logistic_pred))


# =========================
# SVM MODEL
# =========================

print("\nTraining SVM...")

svm_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=50000
    )),
    ("classifier", LinearSVC(
        class_weight="balanced"
    ))
])

svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_pred)

print("\nSVM Accuracy:", svm_accuracy)
print(classification_report(y_test, svm_pred))


# =========================
# SAVE MODELS
# =========================

os.makedirs(os.path.dirname(LOGISTIC_MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(SVM_MODEL_PATH), exist_ok=True)

joblib.dump(logistic_model, LOGISTIC_MODEL_PATH)
joblib.dump(svm_model, SVM_MODEL_PATH)

print("\nModels saved successfully.")


# =========================
# SAVE RESULTS
# =========================

results = pd.DataFrame({
    "model": ["Logistic Regression", "SVM"],
    "accuracy": [logistic_accuracy, svm_accuracy]
})

results_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
results.to_csv(results_path, index=False)

print("Results saved to:", results_path)
print("\nBaseline training completed successfully.")
