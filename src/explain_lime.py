import sys
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from lime.lime_text import LimeTextExplainer

from config import (
    DATA_PATH,
    TEXT_COLUMN,
    LABEL_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    LOGISTIC_MODEL_PATH
)

from preprocessing import clean_arabic_text

sys.stdout.reconfigure(encoding="utf-8")


# =========================
# LOAD MODEL
# =========================

print("Loading Logistic Regression model...")

model = joblib.load(LOGISTIC_MODEL_PATH)

print("Model loaded successfully.")


# =========================
# LOAD DATASET
# =========================

df = pd.read_csv(DATA_PATH)
df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

df["clean_text"] = df[TEXT_COLUMN].apply(clean_arabic_text)
df = df[df["clean_text"].str.strip() != ""]
df = df.drop_duplicates(subset=["clean_text"])


# =========================
# TRAIN / TEST SPLIT
# =========================

X = df["clean_text"]
y = df[LABEL_COLUMN]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

test_df = pd.DataFrame({
    "text": X_test,
    "true_label": y_test
}).reset_index(drop=True)


# =========================
# SELECT SAMPLE
# =========================
sample = test_df.sample(1).iloc[0]

text = sample["text"]
true_label = sample["true_label"]
predicted_label = model.predict([text])[0]

print("\nTweet:")
print(text)

print("\nTrue Label:")
print(true_label)

print("\nPredicted Label:")
print(predicted_label)


# =========================
# LIME EXPLANATION
# =========================

class_names = list(model.classes_)

explainer = LimeTextExplainer(class_names=class_names)

explanation = explainer.explain_instance(
    text,
    model.predict_proba,
    num_features=10
)

print("\nLIME Explanation:")
for word, weight in explanation.as_list():
    print(word, ":", weight)


# =========================
# SAVE HTML EXPLANATION
# =========================


import os
from config import RESULTS_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)

output_path = os.path.join(RESULTS_DIR, "lime_explanation.html")

explanation.save_to_file(output_path)

print("\nLIME explanation saved to:")
print(output_path)