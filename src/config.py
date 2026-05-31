import os

# =========================
# BASE PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# =========================
# DATASET
# =========================

DATA_PATH = os.path.join(DATA_DIR, "arabic_sentiment.csv")

TEXT_COLUMN = "Text"
LABEL_COLUMN = "sentiment"

# =========================
# TRAINING
# =========================

TEST_SIZE = 0.2
RANDOM_STATE = 42

# =========================
# MODEL PATHS
# =========================

LOGISTIC_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "logistic_regression.pkl"
)

SVM_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "svm_model.pkl"
)

MARBERT_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "marbert_model"
)

# =========================
# TRANSFORMER MODEL
# =========================

MODEL_NAME = "UBC-NLP/MARBERT"

#This file controls:

#dataset path
#model paths
#column names
#transformer model name
#training settings

#Professional ML projects always centralize settings like this.