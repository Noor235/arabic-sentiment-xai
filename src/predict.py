import random
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH,
    TEXT_COLUMN,
    LABEL_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    LOGISTIC_MODEL_PATH,
    SVM_MODEL_PATH
)

from preprocessing import clean_arabic_text


# =========================
# LOAD MODELS
# =========================

print("Loading trained models...")

logistic_model = joblib.load(LOGISTIC_MODEL_PATH)
svm_model = joblib.load(SVM_MODEL_PATH)

print("Models loaded successfully.")


# =========================
# LOAD DATASET
# =========================

df = pd.read_csv(DATA_PATH)

df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

df["clean_text"] = df[TEXT_COLUMN].apply(clean_arabic_text)

df = df[df["clean_text"].str.strip() != ""]
df = df.drop_duplicates(subset=["clean_text"])


# =========================
# TRAIN TEST SPLIT
# =========================

X = df["clean_text"]
y = df[LABEL_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

test_df = pd.DataFrame({
    "text": X_test,
    "true_label": y_test
})


# =========================
# RANDOM SAMPLE TESTING
# =========================

print("\nRandom Prediction Samples\n")

random_samples = test_df.sample(5)

for _, row in random_samples.iterrows():

    text = row["text"]
    true_label = row["true_label"]

    logistic_prediction = logistic_model.predict([text])[0]
    svm_prediction = svm_model.predict([text])[0]

    print("=" * 60)

    print("\nTweet:")
    print(text)

    print("\nTrue Label:")
    print(true_label)

    print("\nLogistic Regression Prediction:")
    print(logistic_prediction)

    print("\nSVM Prediction:")
    print(svm_prediction)

    print("\nMatch Results:")
    print("Logistic Correct:", logistic_prediction == true_label)
    print("SVM Correct:", svm_prediction == true_label)

print("\nPrediction testing completed successfully.")