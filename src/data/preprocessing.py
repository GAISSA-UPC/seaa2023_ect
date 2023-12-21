import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import mlflow
from src.environment import METRICS_DIR

POWER_MEASUREMENT_ERROR = 5

WATTS_TO_KWATTS = 1e-3

JOULES_TO_MJOULES = 1e-6
JOULES_TO_GJOULES = 1e-9
JOULES_TO_KWH = 1 / 3.6e6

MJOULES_TO_JOULES = 1e6
MJOULES_TO_KJOULES = 1e3

GJOULES_TO_JOULES = 1e9
GJOULES_TO_KJOULES = 1e6
GJOULES_TO_MJOULES = 1e3

KWH_CO2e_SPA = 232
gCO2e_TO_TCO2e = 1e-6
FLOPS_TO_GFLOPS = 1e-9

HOURS_TO_SECONDS = 3.6e3
HOURS_TO_MILISECONDS = 3.6e6
SECONDS_TO_HOURS = 1 / 3.6e3


def metric2float(value: str):
    if value == "None" or not value:
        return None
    return float(re.findall(r"\d+", value)[0])


def metric2int(value: str):
    if value == "None" or not value:
        return None
    return np.int32(re.findall(r"\d+", value)[0])


def build_metrics_dataset(input_folder=None, output_folder=None):
    """
    Build the metrics dataset from the data in `input_folder`. If `output_folder` is specified it will save the dataset
    as a ``gzip`` file.

    Parameters
    ----------
    input_folder : `os.PathLike`, default None
        The folder with the collected data. If `None`, the default saving location will be used.
    output_folder : `os.PathLike`, default None
        The folder where the dataset will be saved. If `None` the data will not be saved.

    Returns
    -------
    dataframe : `pd.DataFrame`
        A `pd.DataFrame` with the collected data properly formatted.
    """
    dataframes = []
    if input_folder is None:
        input_dir = Path(METRICS_DIR) / "raw"
    elif isinstance(input_folder, str):
        input_dir = Path(input_folder)
    else:
        input_dir = input_folder

    for train_environment in os.listdir(input_dir):
        folder = input_dir / train_environment
        for architecture in os.listdir(folder):
            cpu_files = (folder / architecture).glob(r"cpu*.csv")
            cpu_files = sorted(cpu_files, key=lambda x: os.path.basename(x).split("-")[-1])
            gpu_files = (folder / architecture).glob(r"gpu*.csv")
            gpu_files = sorted(gpu_files, key=lambda x: os.path.basename(x).split("-")[-1])
            for i in range(0, len(cpu_files)):
                cpu_data = pd.read_csv(
                    cpu_files[i],
                    header=0,
                    names=["timestamp", "cpu_usage", "memory_usage", "cpu_temperature"],
                    dtype={
                        "timestamp": str,
                        "cpu_usage": np.float32,
                        "memory_usage": np.float32,
                        "cpu_temperature": np.float32,
                    },
                    parse_dates=["timestamp"],
                    na_values="None",
                )
                cpu_data.drop(columns=["cpu_temperature"], inplace=True)
                cpu_data = cpu_data.sort_values(by="timestamp")

                gpu_data = pd.read_csv(
                    gpu_files[i],
                    header=0,
                    sep=",",
                    engine="python",
                    names=[
                        "timestamp",
                        "gpu_name",
                        "gpu_usage",
                        "gpu_memory_usage",
                        "gpu_total_memory",
                        "gpu_memory_used",
                        "gpu_power_draw",
                        "gpu_max_power",
                        "gpu_temperature",
                    ],
                    dtype={"timestamp": str, "gpu_temperature": int},
                    parse_dates=["timestamp"],
                    converters={
                        "gpu_usage": lambda x: metric2float(x) / 100,
                        "gpu_memory_usage": lambda x: metric2float(x) / 100,
                        "gpu_total_memory": metric2int,
                        "gpu_memory_used": metric2int,
                        "gpu_power_draw": metric2float,
                        "gpu_max_power": metric2float,
                    },
                    na_values="None",
                    skipinitialspace=True,
                    skiprows=3,  # remove the initial 3 seconds of measurements.
                    skipfooter=3,  # remove last 3 seconds of measurements.
                )
                gpu_data = gpu_data.sort_values(by="timestamp")
                tol = pd.Timedelta("1 second")
                df = pd.merge_asof(gpu_data, cpu_data, on="timestamp", direction="nearest", tolerance=tol)

                df["train_environment"] = train_environment
                df["architecture"] = architecture
                df["run"] = i
                cols = df.columns.tolist()
                df = df[cols[-3:] + cols[:-3]]
                dataframes.append(df)
    df = pd.concat(dataframes, axis="index", ignore_index=True)
    df = df.sort_values(by=["train_environment", "architecture", "run", "timestamp"])

    if output_folder is not None:
        out_file = os.path.join(output_folder, "dl-training-profiling-dataset.gzip")
        df.to_parquet(out_file, index=False, compression="gzip")

    return df


def get_duration(series):
    return (series.iloc[-1] - series.iloc[0]) / np.timedelta64(1, "h")


def build_analysis_dataset(metrics_file=None, output_folder=None):
    """
    Build the analysis dataset from the data in `input_folder`. If `output_folder` is specified it will save the dataset
    as a ``gzip`` file.

    Parameters
    ----------
    metrics_file : `os.PathLike`, default None
        The file with the collected metrics. If `None`, the default saving location will be used.
    output_folder : `os.PathLike`, default None
        The folder where the dataset will be saved. If `None` the data will not be saved.

    Returns
    -------
    dataframe : `pd.DataFrame`
        A `pd.DataFrame` with the experiment metrics processed and ready to be analyzed.
    """
    if metrics_file is None:
        df = pd.read_parquet(Path(METRICS_DIR) / "interim" / "dl-training-profiling-dataset.gzip")
    else:
        df = pd.read_parquet(metrics_file)
    grouping_features = ["train_environment", "architecture", "run"]
    df_grouped = _aggregate_metrics(df, grouping_features)
    mlflow_df = _get_mlflow_metrics(df, grouping_features, output_folder)
    analysis_df = df_grouped.join(mlflow_df, how="inner")
    analysis_df = analysis_df.reset_index()
    analysis_df.rename(columns={"train_environment": "training environment"}, inplace=True)
    analysis_df.replace(
        {
            "local": "Local Normal User",
            "local-v2": "Local ML Engineer/Gamer",
            "cloud": "Cloud",
        },
        inplace=True,
    )

    if output_folder is not None:
        out_file = os.path.join(output_folder, "dl-training-energy-consumption-dataset.gzip")
        analysis_df.to_parquet(out_file, index=False, compression="gzip")

    return analysis_df


def _get_mlflow_metrics(df, grouping_features, output_folder):
    mlflow_experiments = []
    for train_environment, architecture in df[["train_environment", "architecture"]].drop_duplicates().values:
        mlflow_runs = mlflow.search_runs(
            experiment_names=[f"{train_environment}-chesslive-occupancy-{architecture}"], order_by=["start_time ASC"]
        )

        # Select relevant metrics
        relevant_metrics = mlflow_runs[
            [
                "params.split_number",
                "params.train_size",
                "params.validation_size",
                "params.batch_size",
                "metrics.MACCS",
                "metrics.val_binary_accuracy",
                "metrics.val_precision",
                "metrics.val_recall",
                "metrics.val_auc",
            ]
        ]

        # Turn count of multiply accumulate operations into count of FLOPs
        relevant_metrics.loc[:, "metrics.MACCS"] = relevant_metrics["metrics.MACCS"].apply(
            lambda x: x * 2 * FLOPS_TO_GFLOPS
        )

        relevant_metrics.rename(
            columns={
                "params.split_number": "split number",
                "params.train_size": "training size",
                "params.validation_size": "validation size",
                "params.batch_size": "batch size",
                "metrics.MACCS": "GFLOPs",
                "metrics.val_binary_accuracy": "accuracy",
                "metrics.val_precision": "precision",
                "metrics.val_recall": "recall",
                "metrics.val_auc": "AUC",
            },
            inplace=True,
        )

        # Change training parameters from string to their numberic type
        relevant_metrics["training size"] = pd.to_numeric(relevant_metrics["training size"], errors="raise").astype(int)
        relevant_metrics["validation size"] = pd.to_numeric(relevant_metrics["validation size"], errors="raise").astype(
            int
        )
        relevant_metrics["batch size"] = relevant_metrics["batch size"].astype(int)
        relevant_metrics["split number"] = relevant_metrics["split number"].astype(int)

        # Compute new metrics
        relevant_metrics["trained epochs"] = _get_number_of_trained_epochs(mlflow_runs)
        training_size = relevant_metrics["training size"]
        validation_size = relevant_metrics["validation size"]
        batch_size = relevant_metrics["batch size"]
        relevant_metrics["total seen images"] = (
            training_size - (training_size % batch_size) + validation_size - (validation_size % batch_size)
        ) * relevant_metrics["trained epochs"]

        relevant_metrics["f1-score"] = (
            (2 * relevant_metrics.precision * relevant_metrics.recall)
            / (relevant_metrics.precision + relevant_metrics.recall)
        ).fillna(0)

        # Add experiment run identifier
        relevant_metrics.index.rename("run", inplace=True)
        relevant_metrics = relevant_metrics.reset_index()
        relevant_metrics["train_environment"] = train_environment
        relevant_metrics["architecture"] = architecture
        mlflow_experiments.append(relevant_metrics)

    mlflow_df = pd.concat(mlflow_experiments, axis=0)
    mlflow_df = mlflow_df.set_index(grouping_features)
    if output_folder is not None:
        out_folder = os.path.join(METRICS_DIR, "raw", "interim")
        os.makedirs(out_folder, exist_ok=True)
        out_file = os.path.join(out_folder, "model_metrics.gzip")
        mlflow_df.to_parquet(out_file, compression="gzip")
    return mlflow_df


def _get_number_of_trained_epochs(mlflow_runs):
    epochs_cl = mlflow_runs.get("params.epochs_cl", pd.Series(np.zeros(mlflow_runs.shape[0])))
    epochs_cl = epochs_cl.fillna(0).astype(int)

    if "metrics.restored_epoch" in mlflow_runs.columns:
        restored_epochs = mlflow_runs["metrics.restored_epoch"]
        restored_epochs = restored_epochs.fillna(0).astype(int)
        early_stopping_patience = mlflow_runs.get("params.earlystopping_patience", pd.Series(np.zeros(mlflow_runs.shape[0])))
        early_stopping_patience = early_stopping_patience.fillna(0).astype(int)

        trained_epochs_cl = (restored_epochs + early_stopping_patience).clip(upper=epochs_cl)
    else:
        trained_epochs_cl = epochs_cl

            
    epochs_ft = mlflow_runs.get("params.epochs_ft", pd.Series(np.zeros(mlflow_runs.shape[0])))
    epochs_ft = epochs_ft.fillna(0).astype(int)
    if "metrics.restored_epoch_ft" in mlflow_runs.columns:
        restored_epochs_ft = mlflow_runs["metrics.restored_epoch_ft"]
        restored_epochs_ft = restored_epochs_ft.fillna(0).astype(int)
        early_stopping_patience_ft = mlflow_runs.get("params.earlystopping_patience_ft", pd.Series(np.zeros(mlflow_runs.shape[0])))
        early_stopping_patience_ft = early_stopping_patience_ft.fillna(0).astype(int)

        trained_epochs_ft = (restored_epochs_ft + early_stopping_patience_ft).clip(upper=epochs_ft)
    else:
        trained_epochs_ft = epochs_ft
            
    trained_epochs = trained_epochs_cl + trained_epochs_ft
    return trained_epochs


def _aggregate_metrics(df, grouping_features):
    df_grouped = df.groupby(grouping_features).agg(
        hours_training=("timestamp", get_duration),
        gpu_name=("gpu_name", "first"),
        gpu_working_time=("gpu_usage", lambda x: np.sum(x) * SECONDS_TO_HOURS),
        gpu_usage=("gpu_usage", lambda x: np.mean(x) * 100),
        gpu_memory_working_time=("gpu_memory_usage", lambda x: np.sum(x) * SECONDS_TO_HOURS),
        gpu_memory_usage=("gpu_memory_usage", lambda x: np.mean(x)),
        memory_used_avg=("gpu_memory_used", np.mean),
        memory_used_std=("gpu_memory_used", np.std),
        W=("gpu_power_draw", np.sum),
        W_avg=("gpu_power_draw", np.mean),
        W_std=("gpu_power_draw", np.std),
        gpu_max_power=("gpu_max_power", "first"),
        temperature_avg=("gpu_temperature", np.mean),
        temperature_std=("gpu_temperature", np.std),
    )
    df_grouped["energy (MJ)"] = df.groupby(grouping_features).apply(
        lambda d: np.trapz(y=d.gpu_power_draw, dx=np.diff(d.timestamp) / np.timedelta64(1, "s")) * JOULES_TO_MJOULES
    )
    # df_grouped["energy (GJ)"] = df_grouped.W * (df_grouped.hours_training * HOURS_TO_SECONDS) * JOULES_TO_GJOULES
    df_grouped["emissions (gCO2e)"] = df_grouped["energy (MJ)"] * MJOULES_TO_JOULES * JOULES_TO_KWH * KWH_CO2e_SPA

    df_grouped.rename(
        columns={
            "hours_training": "training duration (h)",
            "gpu_name": "gpu model",
            "gpu_working_time": "gpu working time (h)",
            "gpu_usage": "gpu usage (%)",
            "gpu_memory_working_time": "gpu memory working time (h)",
            "gpu_memory_usage": "gpu memory usage",
            "memory_used_avg": "average memory used (MB)",
            "memory_used_std": "memory used std (MB)",
            "W": "total power (W)",
            "W_avg": "average power draw (W)",
            "W_std": "power draw std (W)",
            "gpu_max_power": "max power limit (W)",
            "temperature_avg": "average temperature (Celsius)",
            "temperature_std": "temperature std (Celsius)",
        },
        inplace=True,
    )

    return df_grouped
