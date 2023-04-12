import os
import shutil
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import mlflow
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from src.environment import METRICS_DIR


def log_model(run_id, model_path, model_name):
    with mlflow.start_run(run_id=run_id):
        model = tf.keras.models.load_model(model_path)
        mlflow.tensorflow.log_model(
            model=model, artifact_path="model", registered_model_name=model_name, await_registration_for=None
        )


def upload_artifact(run_id, path, dst_folder):
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(path, dst_folder)


def upload_artifacts(run_id, path, dst_folder):
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifacts(path, dst_folder)


def compute_runs_f1score(runs):
    """
    Computes the validation F1-score for all the `runs`.
    Parameters
    ----------
    runs : `pandas.DataFrame`
        A dataframe with the precision and recall of the runs.

    Returns
    -------
    f1-score : `pandas.Series`
        The F1-score of the runs.
    """
    precision = runs["metrics.val_precision"]
    recall = runs["metrics.val_recall"]
    return (2 * (precision * recall) / (precision + recall)).fillna(0)


def search_runs_with_models(architecture):
    client = mlflow.client.MlflowClient()
    models = client.search_model_versions(filter_string=f"name = '{architecture}' AND attribute.status = 'FINISHED'")
    run_ids = [f"'{model.run_id}'" for model in models]
    selection = ",".join(run_ids)
    runs = mlflow.search_runs(search_all_experiments=True, filter_string=f"run_id IN ({selection})")
    return runs


def search_best_f1score(architecture):
    """
    Search the best F1-score for a given architecture.

    Parameters
    ----------
    architecture : str
        The architecture to search for the F1-score.

    Returns
    -------
    f1-score : float
        The best F1-score for the architecture.

    """
    runs = search_runs_with_models(architecture)
    return np.max(compute_runs_f1score(runs))


def download_best_model(architecture, out_path=None):
    runs = search_runs_with_models(architecture)
    runs["metrics.val_f1score"] = compute_runs_f1score(runs)
    best_run = runs.iloc[runs["metrics.val_f1score"].idxmax()].run_id
    print(f"Best run ID: {best_run}")
    mlflow.artifacts.download_artifacts(run_id=best_run, artifact_path="model/data/model", dst_path=out_path)
    shutil.move(os.path.join(out_path, "model/data/model"), os.path.join(out_path, f"best-{architecture}"))
    shutil.rmtree(os.path.join(out_path, "model"))


def download_energy_metrics(environment, architecture):
    runs = mlflow.search_runs(
        experiment_names=[f"{environment}-{architecture}"], filter_string="attribute.status = 'FINISHED'"
    )

    out_path = os.path.join(METRICS_DIR, "raw", strategy)
    for run in tqdm(runs.run_id):
        try:
            mlflow.artifacts.download_artifacts(run_id=run, artifact_path="metrics", dst_path=out_path)
        except ConnectionResetError:
            # Suspend download for 5 seconds if connection is closed by the server and try again.
            print("Waiting 5 seconds")
            time.sleep(5)
            mlflow.artifacts.download_artifacts(run_id=run, artifact_path="metrics", dst_path=out_path)

    os.rename(os.path.join(out_path, "metrics"), os.path.join(out_path, architecture))


def remove_obsolete_models(keep_n_best: int = 10):
    """
    Removes all the models from the configured mlflow tracking server except for the top-n for each architecture and task.

    Parameters
    ----------
    keep_n_best : int, default 10
        Keep the top _n_ best models according to the F1-score.
    """

    client = mlflow.client.MlflowClient()

    runs = mlflow.search_runs(search_all_experiments=True, filter_string="attribute.status = 'FINISHED'")
    runs["metrics.val_f1score"] = compute_runs_f1score(runs)
    architectures = runs["tags.model"].unique()
    for architecture in architectures:
        runs_ids = (
            runs.query(f"`tags.model` == @architecture")
            .sort_values(by="metrics.val_f1score", ascending=False)
            .reset_index()
            .loc[keep_n_best:, "run_id"]
            .values
        )

        for run_id in runs_ids:
            models = client.search_model_versions(filter_string=f"run_id = '{run_id}'")
            if models:
                model = models[0]
                client.delete_model_version(name=model.name, version=model.version)
                repository = get_artifact_repository(runs[runs.run_id == run_id].artifact_uri.iloc[0])
                repository.delete_artifacts("model")
