import os
from argparse import ArgumentParser

import mlflow
from src import MLFLOW_ENABLED
from src.data.utils import load_data
from src.environment import DATASET_DIR, ROOT
from src.mlflow.utils import search_best_f1score
from src.models.train_model import run_main

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "environment",
        help="Whether is to be executed locally or in the cloud.",
        choices=["local", "cloud"],
    )
    parser.add_argument(
        "arch",
        help="Architecture of the DNN",
        choices=["vgg16", "resnet50", "xception", "mobilenet_v2", "nasnet_mobile"],
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
        help="Path to the dataset folder",
        default=DATASET_DIR,
        type=str,
    )

    args = parser.parse_args()
    experiment_name = args.experiment_name
    architecture = args.arch.lower()
    environment = args.environment.lower()
    data_folder = args.data

    if MLFLOW_ENABLED:
        if experiment_name is None:
            raise ValueError("MLflow is enabled but no experiment name was provided.")
        best_f1 = search_best_f1score(architecture)
        experiment = mlflow.set_experiment(experiment_name)
    else:
        mlflow.tensorflow.autolog(disable=True)

    train_ds, train_steps_per_epoch, val_ds, val_steps_per_epoch = load_data(data_folder)
    run_id, figure, model = run_main(architecture, train_ds, train_steps_per_epoch, val_ds, val_steps_per_epoch)
    if MLFLOW_ENABLED:
        with mlflow.start_run(run_id=run_id) as run:
            mlflow.set_tag("environment", environment)
            mlflow.log_artifact(figure, "figures")

            f1 = (
                2
                * run.data.metrics["val_precision"]
                * run.data.metrics["val_recall"]
                / (run.data.metrics["val_precision"] * run.data.metrics["val_recall"])
            )
            if f1 >= best_f1:
                mlflow.tensorflow.log_model(
                    model=model,
                    artifact_path="model",
                    registered_model_name=architecture,
                    await_registration_for=None,
                )
