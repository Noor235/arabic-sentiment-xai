import re
import pandas as pd

from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH,
    TEXT_COLUMN,
    LABEL_COLUMN,
    TEST_SIZE,
    RANDOM_STATE
)


# =========================
# CLEANING FUNCTION
# =========================

def clean_arabic_text(text):

    text = str(text)

    # remove links
    text = re.sub(r"http\S+|www\S+", " ", text)

    # remove mentions
    text = re.sub(r"@\w+", " ", text)

    # remove hashtag symbol only
    text = text.replace("#", " ")

    # normalize Arabic letters
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)

    # remove tashkeel
    text = re.sub(r"[\u064B-\u065F]", "", text)

    # remove non-Arabic chars
    text = re.sub(r"[^ء-ي\s]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================
# LOAD DATASET
# =========================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print("\nSentiment Distribution:")
print(df[LABEL_COLUMN].value_counts())


# =========================
# KEEP NEEDED COLUMNS
# =========================

df = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()

df.dropna(inplace=True)

# =========================
# CLEAN TEXT
# =========================

print("\nCleaning Arabic text...")

df["clean_text"] = df[TEXT_COLUMN].apply(clean_arabic_text)

# remove empty rows
df = df[df["clean_text"].str.strip() != ""]

# remove duplicates
df.drop_duplicates(subset=["clean_text"], inplace=True)

print("\nDataset shape after cleaning:", df.shape)

# =========================
# TRAIN / TEST SPLIT
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

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nPreprocessing completed successfully.")
