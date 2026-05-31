import os
import sys
import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from lime.lime_text import LimeTextExplainer

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import (
    DATA_PATH,
    TEXT_COLUMN,
    LABEL_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    MARBERT_MODEL_PATH,
    RESULTS_DIR
)

from preprocessing import clean_arabic_text

sys.stdout.reconfigure(encoding="utf-8")


# =========================
# DEVICE
# =========================

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


# =========================
# LABELS
# =========================

id_to_label = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

class_names = ["negative", "neutral", "positive"]


# =========================
# LOAD MARBERT MODEL
# =========================

print("Loading saved MARBERT model...")

tokenizer = AutoTokenizer.from_pretrained(MARBERT_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MARBERT_MODEL_PATH)

model.to(device)
model.eval()

print("MARBERT loaded successfully.")


# =========================
# LOAD DATASET
# =========================

df = pd.read_csv(DATA_PATH)
df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

# Keep original tweet for display
df["original_text"] = df[TEXT_COLUMN].astype(str)

# Clean version for model processing
df["clean_text"] = df[TEXT_COLUMN].apply(clean_arabic_text)

df = df[df["clean_text"].str.strip() != ""]
df = df.drop_duplicates(subset=["clean_text"])

# IMPORTANT:
# We use original_text for LIME display,
# but clean inside the prediction function.
X = df["original_text"]
y = df[LABEL_COLUMN]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

test_df = pd.DataFrame({
    "original_text": X_test,
    "true_label": y_test
}).reset_index(drop=True)


# =========================
# MARBERT PREDICT PROBA
# Required by LIME
# =========================

def marbert_predict_proba(texts):
    cleaned_texts = [clean_arabic_text(text) for text in texts]

    inputs = tokenizer(
        cleaned_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)

    return probabilities.cpu().numpy()


# =========================
# RANDOM SAMPLE
# =========================

sample = test_df.sample(1).iloc[0]

original_text = sample["original_text"]
cleaned_text = clean_arabic_text(original_text)
true_label = sample["true_label"]

probabilities = marbert_predict_proba([original_text])[0]
predicted_index = int(np.argmax(probabilities))
predicted_label = id_to_label[predicted_index]


# =========================
# PRINT RESULT
# =========================

print("\nOriginal Tweet:")
print(original_text)

print("\nCleaned Tweet Used by Model:")
print(cleaned_text)

print("\nTrue Label:")
print(true_label)

print("\nPredicted Label:")
print(predicted_label)

print("\nPrediction Probabilities:")
for label, prob in zip(class_names, probabilities):
    print(label, ":", round(float(prob), 4))


# =========================
# LIME EXPLANATION
# =========================

print("\nGenerating LIME explanation for MARBERT...")

explainer = LimeTextExplainer(class_names=class_names)

explanation = explainer.explain_instance(
    original_text,
    marbert_predict_proba,
    num_features=10,
    num_samples=500
)

print("\nLIME Explanation:")
for word, weight in explanation.as_list():
    print(word, ":", weight)


# =========================
# SAVE HTML EXPLANATION
# =========================

os.makedirs(RESULTS_DIR, exist_ok=True)

output_path = os.path.join(
    RESULTS_DIR,
    "marbert_lime_explanation.html"
)

explanation.save_to_file(output_path)

print("\nMARBERT + LIME explanation saved to:")
print(output_path)