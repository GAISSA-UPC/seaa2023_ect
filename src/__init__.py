import logging
import os
from typing import Union

import tensorflow as tf

import mlflow
from src.__version__ import __version__
from src.environment import DEBUG, GPU_MEM_LIMIT, ROOT, read_config


def limit_gpu_memory(max_size: Union[int, None]):
    """
    Put a hard limit on the available GPU memory for the training.

    Parameters
    ----------
    max_size : Union[int, None]
        The maximum available GPU memory. If None the default TensorFlow configuration will be used.

    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise Exception("No GPU found.")
    if max_size is not None:
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                tf.config.set_logical_device_configuration(
                    gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=max_size)]
                )
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)


limit_gpu_memory(GPU_MEM_LIMIT)

cfg = read_config(os.path.join(ROOT, "config", "experiment.yaml"))
REPETITIONS = cfg["REPETITIONS"]
WARMUP_TIME = cfg["WARMUP_TIME"]
COOLDOWN = cfg["COOLDOWN"]
COOLDOWN_EVERY = cfg["COOLDOWN_EVERY"]
MLFLOW_ENABLED = cfg["MLFLOW_ENABLED"]

cfg = read_config(os.path.join(ROOT, "secrets.yaml"))
MLFLOW_URL = cfg["MLFLOW"]["URL"]

mlflow.set_tracking_uri(MLFLOW_URL)

MLFLOW_TRACKING_USERNAME = cfg["MLFLOW"]["USERNAME"]
MLFLOW_TRACKING_PASSWORD = cfg["MLFLOW"]["PASSWORD"]
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD


def setup_custom_logger(name):
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    if DEBUG:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARN)
    logger.addHandler(handler)
    return logger


logger = setup_custom_logger("profiler")

SEED = 2022
