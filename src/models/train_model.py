import os.path
import random as python_random
import warnings
from datetime import datetime
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import mlflow
from src import MLFLOW_ENABLED
from src.environment import FIGURES_DIR, METRICS_DIR, MODELS_DIR, PRETRAINED_MODELS_DIR
from src.models import (
    BATCH_SIZE,
    FINE_TUNE_AT,
    FT_EPOCHS,
    FT_LEARNING_RATE,
    FT_MOMENTUM,
    FT_OPTIMIZER,
    IMAGE_SIZE,
    INITIAL_EPOCHS,
    LEARNING_RATE,
    METRICS,
    MOMENTUM,
    OPTIMIZER,
)
from src.profiling.metrics import compute_maccs
from src.utils import plot_hist


def configure_optimizer(
    optimizer: str = "adam", learning_rate: float = 1e-3, momentum: float = None
) -> tf.keras.optimizers.Optimizer:
    """
    Configure the optimizer for a TensorFlow model.

    Parameters
    ----------
    optimizer : str
        The name of the optimizer. Available options are: "adam", "sgd".
    learning_rate : float
        The learning rate to use for the optimizer.
    momentum : float
        The momentum to use for the optimizer if used at all.

    Returns
    -------
    optimizer
        The configured optimizer.
    """
    if optimizer == "adam":
        return Adam(learning_rate)
    elif optimizer == "sgd":
        if momentum is None:
            raise ValueError("Momentum can not be None for SGD optimizer.")
        return SGD(learning_rate=learning_rate, momentum=momentum)
    else:
        raise NotImplementedError(f"Configuration for {optimizer} optimizer is not implemented.")


def create_model(architecture: str, weights: str = "imagenet") -> Tuple[tf.keras.models.Model, tf.keras.models.Model]:
    """
    Creates a ``tensorflow.keras.models.Model`` with the given architecture as the base model.

    Parameters
    ----------
    architecture : str
        The base architecture for the model.
    weights : str, default "imagenet"
        The weights to use for the pretrained model.

    Returns
    -------
    model : ``tensorflow.keras.models.Model``
        The created model.
    base_model : ``tensorflow.keras.models.Model``
        The base model.
    """
    optimizer = configure_optimizer(OPTIMIZER, LEARNING_RATE, MOMENTUM)
    if architecture == "vgg16":
        return create_VGG16(optimizer, weights)
    elif architecture == "resnet50":
        return create_ResNet50(optimizer, weights)
    elif architecture == "xception":
        return create_Xception(optimizer, weights)
    elif architecture == "mobilenet_v2":
        return create_MobileNetV2(optimizer, weights)
    elif architecture == "nasnet_mobile":
        return create_NASNetMobile(optimizer, weights)
    else:
        raise NotImplementedError("Unknown architecture type.")


def _log_basic_params():
    params = {
        "batch_size": BATCH_SIZE,
        "epochs_cl": INITIAL_EPOCHS,
        "epochs_ft": FT_EPOCHS,
        "epochs": INITIAL_EPOCHS + FT_EPOCHS,
        "optimizer": OPTIMIZER,
        "opt_lr": LEARNING_RATE,
        "opt_momentum": MOMENTUM,
        "optimizer_ft": FT_OPTIMIZER,
        "opt_lr_ft": FT_LEARNING_RATE,
        "opt_momentum_ft": FT_MOMENTUM,
        "image_size": IMAGE_SIZE,
    }
    mlflow.log_params(params)


def _log_dataset_info(train_size, val_size):
    params = {
        "train_size": train_size,
        "validation_size": val_size,
    }
    mlflow.log_params(params)


def _log_callbacks_params(callbacks, fine_tunning=False):
    suffix = "_ft" if fine_tunning else ""
    for callback in callbacks:
        if isinstance(callback, tf.keras.callbacks.ReduceLROnPlateau):
            params = {
                f"reducelr_monitor{suffix}": callback.monitor,
                f"reducelr_min_delta{suffix}": callback.min_delta,
                f"reducelr_patience{suffix}": callback.patience,
                f"reducelr_factor{suffix}": callback.factor,
            }
            mlflow.log_params(params)
        if isinstance(callback, tf.keras.callbacks.EarlyStopping):
            params = {
                f"earlystopping_monitor{suffix}": callback.monitor,
                f"earlystopping_min_delta{suffix}": callback.min_delta,
                f"earlystopping_patience{suffix}": callback.patience,
                f"earlystopping_restore_best_weights{suffix}": callback.restore_best_weights,
            }
            mlflow.log_params(params)


def _log_early_stop_callback_metrics(callback, history, metrics_logger, fine_tunning=False):
    suffix = "_ft" if fine_tunning else ""
    stopped_epoch = callback.stopped_epoch
    restore_best_weights = callback.restore_best_weights

    if not restore_best_weights or callback.best_weights is None:
        return

    monitored_metric = history.history.get(callback.monitor)
    if not monitored_metric:
        return

    restored_epoch = callback.best_epoch
    metrics_logger.record_metrics({f"restored_epoch{suffix}": restored_epoch, f"epochs{suffix}": stopped_epoch})
    restored_index = history.epoch.index(restored_epoch)
    restored_metrics = {key: metrics[restored_index] for key, metrics in history.history.items()}
    # Checking that a metric history exists
    metric_key = next(iter(history.history), None)
    if metric_key is not None:
        metrics_logger.record_metrics(restored_metrics, stopped_epoch + 1)


class MetricsMonitor(tf.keras.callbacks.Callback):
    def __init__(self, metrics_logger, log_every_n_steps=1):
        super().__init__()
        self.metrics_logger = metrics_logger
        self.log_every_n_steps = log_every_n_steps

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def on_train_begin(self, logs=None):
        sum_list = []
        try:
            self.model.summary(print_fn=sum_list.append)
            summary = "\n".join(sum_list)
            mlflow.log_text(summary, artifact_file="model_summary.txt")
        except ValueError as ex:
            if "This model has not yet been built" in str(ex):
                warnings.warn(str(ex))
            else:
                raise ex

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_every_n_steps == 0:
            self.metrics_logger.record_metrics(logs, epoch)


def train(
    model: tf.keras.models.Model,
    base_model: tf.keras.models.Model,
    architecture: str,
    train_ds: tf.data.Dataset,
    train_size: int,
    val_ds: tf.data.Dataset,
    val_size: int,
):
    metrics_dir = os.path.join(METRICS_DIR, "raw", architecture)
    os.makedirs(metrics_dir, exist_ok=True)
    outfile = os.path.join(metrics_dir, "raw", "model_flops.txt")
    maccs = compute_maccs(model, outfile)

    tf.keras.backend.clear_session()

    accuracy_metric = METRICS["acc"]
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor=f"val_{accuracy_metric}",
        patience=20,
        mode="auto",
        restore_best_weights=False,
        verbose=1,
        min_delta=1e-3,
    )
    callbacks = [
        early_stopping_callback,
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=f"val_{accuracy_metric}", mode="auto", factor=0.1, patience=10, min_delta=1e-3
        ),
    ]
    if MLFLOW_ENABLED:
        with mlflow.start_run(tags={"model": architecture}) as run, mlflow.tensorflow.batch_metrics_logger(
            run.info.run_id
        ) as metrics_logger:
            _log_basic_params()

            _log_dataset_info(train_size, val_size)

            callbacks.append(MetricsMonitor(metrics_logger))

            _log_callbacks_params(callbacks)
            history = train_classifier(model, train_ds, train_size, val_ds, val_size, callbacks)

            _log_early_stop_callback_metrics(early_stopping_callback, history, metrics_logger)

            early_stopping_callback.restore_best_weights = True
            _log_callbacks_params(callbacks, fine_tunning=True)
            history_fine = fine_tune_classifier(
                base_model, history, model, train_ds, train_size, val_ds, val_size, callbacks
            )
            _log_early_stop_callback_metrics(early_stopping_callback, history_fine, metrics_logger, fine_tunning=True)

            mlflow.log_metric("MACCS", maccs)
            # model.save(os.path.join(MODELS_DIR, f"{architecture}-{task}"))

        run_id = run.info.run_id
    else:
        run_id = None
        csv_logger = tf.keras.callbacks.CSVLogger(
            os.path.join(metrics_dir, f"performance-{datetime.now().strftime('%Y%m%dT%H%M%S')}.csv"),
            append=True,
        )
        callbacks.append(csv_logger)
        history = train_classifier(model, train_ds, train_size, val_ds, val_size, callbacks)

        early_stopping_callback.restore_best_weights = True
        history_fine = fine_tune_classifier(
            base_model, history, model, train_ds, train_size, val_ds, val_size, callbacks
        )
        model.save(os.path.join(MODELS_DIR, architecture))

    hist = join_histories(history, history_fine)
    return model, hist, run_id


def join_histories(history, history_fine):
    hist = {
        "loss": history.history["loss"] + history_fine.history["loss"],
        "val_loss": history.history["val_loss"] + history_fine.history["val_loss"],
    }
    for key in METRICS.keys():
        if key == "acc":
            metric = METRICS[key]
            hist[key] = history.history[metric] + history_fine.history[metric]
            hist[f"val_{key}"] = history.history[f"val_{metric}"] + history_fine.history[f"val_{metric}"]
        else:
            hist[key] = history.history[key] = history_fine.history[key]
            hist[f"val_{key}"] = history.history[f"val_{key}"] + history_fine.history[f"val_{key}"]
    return hist


def train_classifier(
    model: tf.keras.models.Model,
    train_ds: tf.data.Dataset,
    train_size: int,
    val_ds: tf.data.Dataset,
    val_size: int,
    callbacks: List[tf.keras.callbacks.Callback],
):
    history = model.fit(
        train_ds,
        steps_per_epoch=train_size // BATCH_SIZE,
        validation_data=val_ds,
        validation_steps=val_size // BATCH_SIZE,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def fine_tune_classifier(base_model, history, model, train_ds, train_size, val_ds, val_size, callbacks):
    base_model.trainable = True
    start_layer = int(len(base_model.layers) * FINE_TUNE_AT)
    for layer in base_model.layers[:start_layer]:
        layer.trainable = False

    optimizer = configure_optimizer(FT_OPTIMIZER, FT_LEARNING_RATE, FT_MOMENTUM)
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=list(METRICS.values()),
    )

    total_epochs = INITIAL_EPOCHS + FT_EPOCHS
    print("Starting fine-tunning...")
    history_fine = model.fit(
        train_ds,
        steps_per_epoch=train_size // BATCH_SIZE,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1] + 1,
        validation_data=val_ds,
        validation_steps=val_size // BATCH_SIZE,
        callbacks=callbacks,
    )
    return history_fine


def create_VGG16(optimizer: tf.keras.optimizers.Optimizer, weights: str = "imagenet"):
    shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    base_model = VGG16(weights=weights, include_top=False, input_shape=shape)
    # Freeze layers
    base_model.trainable = False

    # Establish new fully connected block
    inputs = tf.keras.Input(shape=shape)
    x = tf.cast(inputs, tf.float32)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    model = build_model(inputs, x, optimizer)
    return model, base_model


def create_ResNet50(optimizer: tf.keras.optimizers.Optimizer, weights: str = "imagenet"):
    shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    base_model = ResNet50(weights=weights, include_top=False, input_shape=shape)
    # Freeze layers
    base_model.trainable = False

    # Establish new fully connected block
    inputs = tf.keras.Input(shape=shape)
    x = tf.cast(inputs, tf.float32)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    model = build_model(inputs, x, optimizer)
    return model, base_model


def create_Xception(optimizer: tf.keras.optimizers.Optimizer, weights: str = "imagenet"):
    shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    base_model = Xception(weights=weights, include_top=False, input_shape=shape)
    # Freeze layers
    base_model.trainable = False

    # Establish new fully connected block
    inputs = tf.keras.Input(shape=shape)
    x = tf.cast(inputs, tf.float32)
    x = tf.keras.applications.xception.preprocess_input(x)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    model = build_model(inputs, x, optimizer)
    return model, base_model


def create_MobileNetV2(optimizer: tf.keras.optimizers.Optimizer, weights: str = "imagenet"):
    shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    base_model = MobileNetV2(weights=weights, include_top=False, input_shape=shape, alpha=0.5)
    # Freeze layers
    base_model.trainable = False

    # Establish new fully connected block
    inputs = tf.keras.Input(shape=shape)
    x = tf.cast(inputs, tf.float32)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    model = build_model(inputs, x, optimizer)
    return model, base_model


def create_NASNetMobile(optimizer: tf.keras.optimizers.Optimizer, weights: str = "imagenet"):
    shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    base_model = NASNetMobile(weights=weights, include_top=False, input_shape=shape)
    # Freeze layers
    base_model.trainable = False

    inputs = tf.keras.Input(shape=shape)
    x = tf.cast(inputs, tf.float32)
    x = tf.keras.applications.nasnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    model = build_model(inputs, x, optimizer)
    return model, base_model


def build_model(inputs, x, optimizer):
    # This is the model we will train
    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=list(METRICS.values()),
    )
    return model


def fix_seeds():
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(2022)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    python_random.seed(2022)

    # The below set_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/random/set_seed
    tf.random.set_seed(2022)

    os.environ["PYTHONHASHSEED"] = "0"


def run_main(architecture, train_ds, train_size, val_ds, val_size, reproducible=True):
    tf.keras.backend.clear_session()

    if reproducible:
        fix_seeds()

    model, base_model = create_model(architecture)
    model, history, run_id = train(model, base_model, architecture, train_ds, train_size, val_ds, val_size)

    figure = plot_hist(history, "acc", architecture, FIGURES_DIR)
    return run_id, figure, model
