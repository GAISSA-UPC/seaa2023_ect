import os

from tensorflow.keras.metrics import AUC, Precision, Recall

from src.environment import ROOT, read_config

cfg = read_config(os.path.join(ROOT, "config", "experiment.yaml"))
INITIAL_EPOCHS = cfg["INITIAL_EPOCHS"]
FT_EPOCHS = cfg["FT_EPOCHS"]

TEST_SPLIT = 0.3

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
OPTIMIZER = "adam"
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
FINE_TUNE_AT = 0.0  # Percentage of the model to freeze
FT_OPTIMIZER = "sgd"
FT_LEARNING_RATE = 1e-5
FT_MOMENTUM = 0.9
METRICS = {"acc": "binary_accuracy", "precision": Precision(), "recall": Recall(), "auc": AUC()}
