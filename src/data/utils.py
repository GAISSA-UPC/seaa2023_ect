import logging

import tensorflow as tf

from src import SEED
from src.environment import USE_CACHE
from src.models import BATCH_SIZE, IMAGE_SIZE, TEST_SPLIT

_logger = logging.getLogger(__name__)


def load_data(data_folder: str):
    train_ds, val_ds = _load_data(data_folder)

    train_size = train_ds.cardinality().numpy()
    val_size = val_ds.cardinality().numpy()
    train_ds = prepare_data(train_ds, BATCH_SIZE, augment=True)
    val_ds = prepare_data(val_ds, BATCH_SIZE)
    return train_ds, train_size, val_ds, val_size


def _load_data(data_folder):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_folder,
        validation_split=TEST_SPLIT,
        subset="training",
        shuffle=True,
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=None,
        label_mode="binary",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_folder,
        validation_split=TEST_SPLIT,
        subset="validation",
        shuffle=True,
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=None,
        label_mode="binary",
    )
    return train_ds, val_ds


def prepare_data(ds, batch_size, augment=False):
    # Resize and rescale all datasets.
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Batch all datasets.
    ds = ds.batch(batch_size)

    if USE_CACHE:
        ds = ds.cache()

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.repeat().map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)


def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    return img


def resize_and_rescale(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return tf.image.resize(image, IMAGE_SIZE)


def data_augmentation(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    image = tf.image.random_hue(image, 0.1)
    return tf.clip_by_value(image, 0.0, 1.0)  # Clip to avoid values out of range after random_brightness
