"""Configuration parameters for the XGBoost model training."""

# Model parameters
MODEL_PARAMS = {
    'objective': 'multi:softmax',
    'eta': 0.1,
    'max_depth': 6,
    'silent': 1,
    'nthread': 4,
    'num_class': 6
}

# Training parameters
NUM_ROUNDS = 5
TRAIN_SPLIT = 0.7

# Dataset configuration
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data"
DATASET_FILE = "./dermatology.data"

# Wandb configuration
WANDB_PROJECT = "Lab1-visualize-models"
WANDB_RUN_NAME = "xgboost"

