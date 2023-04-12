"""
Module based on the code from `Keras <https://keras.io/examples/vision/grad_cam/#the-gradcam-algorithm>`_
by `fchollet <https://twitter.com/fchollet>`_.
"""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import cm
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenet_v2_preprocess_input,
)
from tensorflow.keras.applications.nasnet import (
    preprocess_input as nasnet_preprocess_input,
)
from tensorflow.keras.applications.resnet import (
    preprocess_input as resnet_preprocess_input,
)
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from tensorflow.keras.applications.xception import (
    preprocess_input as xception_preprocess_input,
)

from src.dataset.utils import resize_and_rescale
from src.environment import MODELS_DIR


def unwrap_model(model, arch):
    if arch == "resnet50":
        base_model = model.get_layer(arch)
    elif arch == "xception":
        base_model = model.get_layer(arch)
    elif arch == "mobilenet_v2":
        base_model = model.get_layer("mobilenetv2_0.50_128")
    elif arch == "nasnet_mobile":
        base_model = model.get_layer("NasNet")
    elif arch == "vgg16":
        base_model = model.get_layer(arch)
    else:
        raise ValueError(f"Architecture {arch} is not supported.")
    inp = base_model.input
    average_pooling_layer = get_classifier_avg_pooling_index(model)

    out = base_model.output
    for layer in model.layers[average_pooling_layer:]:
        out = layer(out)
    return tf.keras.models.Model(inp, out)


def get_classifier_avg_pooling_index(model):
    average_pooling_layer = -3
    for i, layer in enumerate(reversed(model.layers)):
        if layer.name == "global_average_pooling2d":
            average_pooling_layer = -i - 1
            break
    return average_pooling_layer


def make_gradcam_heatmap(arch, img, model, pred_index=None):
    img = tf.cast(img, tf.float32)
    unwrapped_model = unwrap_model(model, arch)
    if arch == "resnet50":
        last_conv_layer = unwrapped_model.get_layer("conv5_block3_3_conv")
        img = resnet_preprocess_input(img)
    elif arch == "xception":
        last_conv_layer = unwrapped_model.get_layer("block14_sepconv2")
        img = xception_preprocess_input(img)
    elif arch == "mobilenet_v2":
        last_conv_layer = unwrapped_model.get_layer("Conv_1")
        img = mobilenet_v2_preprocess_input(img)
    elif arch == "nasnet_mobile":
        last_conv_layer = unwrapped_model.get_layer("separable_conv_2_normal_left5")
        img = nasnet_preprocess_input(img)
    elif arch == "vgg16":
        last_conv_layer = unwrapped_model.get_layer("block5_conv3")
        img = vgg_preprocess_input(img)
    else:
        raise ValueError(f"Architecture {arch} is not supported.")

    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model([unwrapped_model.inputs], [last_conv_layer.output, unwrapped_model.output])

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also scale and normalize the heatmap between 0 & 1
    current_min = tf.reduce_min(heatmap)
    current_max = tf.reduce_max(heatmap)
    heatmap = (heatmap - current_min) / (current_max - current_min)
    return heatmap.numpy()


def plot_gradcam(img, heatmap, alpha=0.6):
    # Rescale to a range 0-255
    img = np.uint8(255 * img)
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    plt.imshow(superimposed_img)
    plt.axis("off")
    plt.show()


def visualize_activations(square_img, arch, task, orientation_a8):
    image = cv2.imread(square_img)
    img = resize_and_rescale(image)
    img = np.expand_dims(img, axis=0)

    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, f"{arch}-{task}"))

    preds = model.predict(img)
    print("Empty" if preds < 0.5 else "Occupied", preds)  # prints the class of image

    # Remove the last layer's activation
    model.layers[-1].activation = None

    heatmap = make_gradcam_heatmap(arch, img, model)
    plt.matshow(heatmap)
    plt.show()

    plot_gradcam(square_img, heatmap)
