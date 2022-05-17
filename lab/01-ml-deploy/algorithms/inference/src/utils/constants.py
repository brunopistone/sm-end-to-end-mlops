from datetime import datetime
import os

DATETIME_NOW = datetime.now().strftime("%Y-%m-%d%H:%M:%S")

MODEL_NAME = "model"

BASE_PATH = os.path.join("/", "opt", "ml")
INPUT_PATH = os.path.join(BASE_PATH, "input", "data", "train")
INPUT_MODELS_PATH = os.path.join(BASE_PATH, "input", "data", "models")
CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints", DATETIME_NOW)
MODEL_PATH = os.path.join(BASE_PATH, "model")
OUTPUT_PATH = os.path.join(BASE_PATH, "output")
OUTPUT_PATH_DATA = os.path.join(BASE_PATH, "output", "data")
PARAM_FILE = os.path.join(BASE_PATH, "input", "config", "hyperparameters.json")
PROCESSING_PATH = os.path.join(BASE_PATH, "processing")
PROCESSING_PATH_INPUT = os.path.join(PROCESSING_PATH, "input")
PROCESSING_PATH_OUTPUT = os.path.join(PROCESSING_PATH, "output")