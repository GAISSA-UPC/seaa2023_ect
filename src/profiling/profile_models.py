import logging
import os
import subprocess
import time
from argparse import ArgumentParser
from datetime import datetime

import tensorflow_addons as tfa

import mlflow
from src import COOLDOWN, COOLDOWN_EVERY, MLFLOW_ENABLED, REPETITIONS, WARMUP_TIME
from src.data.utils import load_data
from src.environment import DATASET_DIR, METRICS_DIR, ROOT
from src.mlflow.utils import search_best_f1score
from src.models import BATCH_SIZE
from src.models.train_model import create_model
from src.profiling.monitoring import monitor_training
from src.utils import create_folder_if_not_exists

MINUTES_TO_SECONDS = 60

logger = logging.getLogger("profiler")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "environment",
        help="The type of training environment.",
        choices=["local", "cloud"],
        type=str,
    )
    parser.add_argument(
        "--experiment_name",
        help="The name of the MLflow experiment.",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--data",
        help="Path to the dataset folder.",
        default=DATASET_DIR,
        type=str,
    )
    args = parser.parse_args()
    experiment_name = args.experiment_name
    environment = args.environment.lower()
    data_folder = args.data
    return experiment_name, environment, data_folder


def warmup():
    print("Starting warmup...")
    warmup_model, _ = create_model("xception")
    warmup_model.fit(
        train_ds,
        epochs=100,
        validation_data=val_ds,
        steps_per_epoch=train_size // BATCH_SIZE,
        validation_steps=val_size // BATCH_SIZE,
        callbacks=tfa.callbacks.TimeStopping(seconds=WARMUP_TIME * MINUTES_TO_SECONDS, verbose=1),
    )
    print(f"Waiting {COOLDOWN} minutes to cooldown before starting.")
    time.sleep(COOLDOWN * MINUTES_TO_SECONDS)


if __name__ == "__main__":
    experiment_name, environment, data_folder = parse_args()
    train_ds, train_size, val_ds, val_size = load_data(data_folder)

    warmup()

    runs = 0
    gpu_id = os.getenv("GPU_DEVICE_ORDINAL", 0)
    for arch in ["mobilenet_v2", "nasnet_mobile", "xception", "resnet50", "vgg16"]:
        if MLFLOW_ENABLED:
            if experiment_name is None:
                raise ValueError("MLflow is enabled but no experiment name was provided.")

            mlflow.set_experiment(experiment_name)
            best_f1 = search_best_f1score(arch)

        OUT_DIR = os.path.join(METRICS_DIR, "raw", environment, arch)

        create_folder_if_not_exists(OUT_DIR)
        for i in range(REPETITIONS):
            print(f"Start run {i} for architecture {arch}")
            if COOLDOWN_EVERY > 0 and runs != 0 and runs % COOLDOWN_EVERY == 0:
                print(f"Waiting {COOLDOWN} minutes to cooldown")
                time.sleep(COOLDOWN * MINUTES_TO_SECONDS)

            creation_time = datetime.now().strftime("%Y%m%dT%H%M%S")
            gpu_metrics = os.path.join(OUT_DIR, f"gpu-power-{creation_time}.csv")
            cpu_metrics = os.path.join(OUT_DIR, f"cpu-mem-usage-{creation_time}.csv")
            command = f"nvidia-smi -i {gpu_id} --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,power.max_limit,temperature.gpu --format=csv -l 1 -f {gpu_metrics}"
            nvidiaProfiler = subprocess.Popen(command.split())

            time.sleep(3)
            run_id, acc_plot, model = monitor_training(arch, train_ds, train_size, val_ds, val_size, cpu_metrics)
            time.sleep(3)

            nvidiaProfiler.terminate()

            if run_id is not None:
                with mlflow.start_run(run_id=run_id) as run:
                    mlflow.set_tag("environment", environment)
                    mlflow.log_artifact(acc_plot, "figures")
                    mlflow.log_artifact(cpu_metrics, "metrics")
                    mlflow.log_artifact(gpu_metrics, "metrics")

                    precision = run.data.metrics["val_precision"]
                    recall = run.data.metrics["val_recall"]
                    f1 = 2 * (precision * recall) / (precision + recall)
                    if f1 >= best_f1:
                        mlflow.tensorflow.log_model(model, "model", registered_model_name=arch)
                        best_f1 = f1

            runs += 1
