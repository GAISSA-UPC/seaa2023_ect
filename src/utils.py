import datetime
import os
from datetime import datetime

from matplotlib import pyplot as plt


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def plot_hist(history, acc_metric, architecture, out_dir):
    acc = history[acc_metric]
    val_acc = history[f"val_{acc_metric}"]
    loss = history["loss"]
    val_loss = history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label="training")
    plt.plot(val_acc, label="validation")
    # plt.ylim([0.8, 1])
    plt.legend()
    plt.title(f"Accuracy for {architecture} model")

    plt.subplot(2, 1, 2)
    plt.plot(loss, label="training")
    plt.plot(val_loss, label="validation")
    # plt.ylim([0, 1.0])
    plt.legend()
    plt.title(f"Loss for {architecture} model")
    plt.xlabel("epoch")
    name = f"{architecture}_{datetime.now().strftime('%Y-%m-%dT%H_%M_%S')}.png"
    picture_path = os.path.join(out_dir, f"{name}")
    plt.savefig(picture_path)
    plt.close()
    return picture_path
