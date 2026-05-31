import os
import sys
import joblib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import (
    DATA_PATH,
    TEXT_COLUMN,
    LABEL_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    LOGISTIC_MODEL_PATH,
    SVM_MODEL_PATH,
    MARBERT_MODEL_PATH,
    RESULTS_DIR
)

from preprocessing import clean_arabic_text

sys.stdout.reconfigure(encoding="utf-8")

os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATA_PATH)
df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

df["clean_text"] = df[TEXT_COLUMN].apply(clean_arabic_text)
df = df[df["clean_text"].str.strip() != ""]
df = df.drop_duplicates(subset=["clean_text"])

X = df["clean_text"]
y = df[LABEL_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

labels = ["negative", "neutral", "positive"]


# =========================
# HELPER FUNCTIONS
# =========================

def calculate_metrics(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    return {
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-score": round(f1, 4)
    }


def save_confusion_matrix(model_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"{model_name} confusion matrix saved to:", output_path)


# =========================
# LOAD BASELINE MODELS
# =========================

print("Loading baseline models...")

logistic_model = joblib.load(LOGISTIC_MODEL_PATH)
svm_model = joblib.load(SVM_MODEL_PATH)


# =========================
# BASELINE PREDICTIONS
# =========================

print("Predicting Logistic Regression...")
logistic_pred = logistic_model.predict(X_test)

print("Predicting SVM...")
svm_pred = svm_model.predict(X_test)


# =========================
# LOAD MARBERT
# =========================

print("Loading MARBERT...")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MARBERT_MODEL_PATH)
marbert_model = AutoModelForSequenceClassification.from_pretrained(MARBERT_MODEL_PATH)

marbert_model.to(device)
marbert_model.eval()

id_to_label = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


def predict_marbert(texts, batch_size=32):
    all_predictions = []

    for i in range(0, len(texts), batch_size):
        batch_texts = list(texts[i:i + batch_size])

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = marbert_model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_predictions.extend([id_to_label[int(p)] for p in preds])

    return all_predictions


print("Predicting MARBERT...")
marbert_pred = predict_marbert(X_test)


# =========================
# SAVE METRICS TABLE
# =========================

metrics = []

metrics.append(calculate_metrics("Logistic Regression", y_test, logistic_pred))
metrics.append(calculate_metrics("SVM", y_test, svm_pred))
metrics.append(calculate_metrics("MARBERT", y_test, marbert_pred))

metrics_df = pd.DataFrame(metrics)

metrics_path = os.path.join(RESULTS_DIR, "model_comparison_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

print("\nModel Comparison Metrics:")
print(metrics_df)

print("\nMetrics saved to:", metrics_path)


# =========================
# SAVE CLASSIFICATION REPORTS
# =========================

reports = {
    "logistic_regression": logistic_pred,
    "svm": svm_pred,
    "marbert": marbert_pred
}

for model_name, predictions in reports.items():
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        zero_division=0
    )

    report_path = os.path.join(
        RESULTS_DIR,
        f"{model_name}_classification_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    print(f"{model_name} classification report saved to:", report_path)


# =========================
# SAVE CONFUSION MATRICES
# =========================

save_confusion_matrix("Logistic Regression", y_test, logistic_pred)
save_confusion_matrix("SVM", y_test, svm_pred)
save_confusion_matrix("MARBERT", y_test, marbert_pred)


# =========================
# SAVE MODEL COMPARISON BAR CHART
# =========================

plt.figure(figsize=(8, 5))
plt.bar(metrics_df["Model"], metrics_df["Accuracy"])
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.tight_layout()

chart_path = os.path.join(RESULTS_DIR, "model_accuracy_comparison.png")
plt.savefig(chart_path, dpi=300)
plt.close()

print("Accuracy comparison chart saved to:", chart_path)

print("\nEvaluation completed successfully.")