import os
from pathlib import Path

import yaml


def read_config(file):
    with open(file, "r") as cfg_file:
        try:
            return yaml.safe_load(cfg_file)
        except yaml.YAMLError as exc:
            print(exc)


ROOT = os.path.dirname(Path(__file__).parent)

resources_cfg = read_config(os.path.join(ROOT, "config", "resources.yaml"))
GPU_MEM_LIMIT = resources_cfg["GPU"]["MEM_LIMIT"]
USE_CACHE = resources_cfg["USE_CACHE"]

base_cfg = read_config(os.path.join(ROOT, "config", "base.yaml"))
OUTPUT_DIR = os.path.join(ROOT, base_cfg["OUTPUT_DIR"])
os.makedirs(OUTPUT_DIR, exist_ok=True)
FIGURES_DIR = os.path.join(ROOT, base_cfg["FIGURES_DIR"])
os.makedirs(FIGURES_DIR, exist_ok=True)
METRICS_DIR = os.path.join(ROOT, base_cfg["METRICS_DIR"])
os.makedirs(METRICS_DIR, exist_ok=True)
DATASET_DIR = os.path.join(ROOT, base_cfg["DATASET_DIR"])
os.makedirs(DATASET_DIR, exist_ok=True)
MODELS_DIR = os.path.join(ROOT, base_cfg["MODELS_DIR"])
os.makedirs(MODELS_DIR, exist_ok=True)
PRETRAINED_MODELS_DIR = os.path.join(MODELS_DIR, "pre-trained")
os.makedirs(PRETRAINED_MODELS_DIR, exist_ok=True)

INFERENCE_MODEL = os.path.join(MODELS_DIR, base_cfg["INFERENCE_MODEL"])

DEBUG = base_cfg["DEBUG"]
