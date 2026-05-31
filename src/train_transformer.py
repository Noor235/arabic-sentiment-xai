import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from config import (
    DATA_PATH,
    TEXT_COLUMN,
    LABEL_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_NAME,
    MARBERT_MODEL_PATH
)

from preprocessing import clean_arabic_text

sys.stdout.reconfigure(encoding="utf-8")


# =========================
# LOAD DATASET
# =========================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

df["clean_text"] = df[TEXT_COLUMN].apply(clean_arabic_text)

df = df[df["clean_text"].str.strip() != ""]
df = df.drop_duplicates(subset=["clean_text"])

print("Dataset shape:", df.shape)

print("\nClass Distribution:")
print(df[LABEL_COLUMN].value_counts())


# =========================
# LABEL ENCODING
# =========================

label_mapping = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

df["label"] = df[LABEL_COLUMN].map(label_mapping)

print("\nLabel mapping completed.")


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)

train_df = pd.DataFrame({
    "text": X_train,
    "label": y_train
})

test_df = pd.DataFrame({
    "text": X_test,
    "label": y_test
})


# =========================
# CONVERT TO HF DATASET
# =========================

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


# =========================
# LOAD TOKENIZER
# =========================

print("\nLoading MARBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# =========================
# TOKENIZATION
# =========================

def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

print("\nTokenizing dataset...")

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)


# =========================
# LOAD MODEL
# =========================

print("\nLoading MARBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)


# =========================
# METRICS
# =========================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy
    }


# =========================
# TRAINING ARGUMENTS
# =========================

training_args = TrainingArguments(
    output_dir="../results/marbert_results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="../results/logs",
    logging_steps=50
)

# =========================
# TRAINER
# =========================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


# =========================
# TRAIN MODEL
# =========================

print("\nStarting MARBERT training...")

trainer.train()


# =========================
# EVALUATION
# =========================

print("\nEvaluating model...")

predictions = trainer.predict(test_dataset)

preds = np.argmax(predictions.predictions, axis=-1)

accuracy = accuracy_score(y_test, preds)

print("\nMARBERT Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, preds))


# =========================
# SAVE MODEL
# =========================

print("\nSaving model...")

os.makedirs(MARBERT_MODEL_PATH, exist_ok=True)

trainer.save_model(MARBERT_MODEL_PATH)
tokenizer.save_pretrained(MARBERT_MODEL_PATH)

print("\nMARBERT model saved successfully.")