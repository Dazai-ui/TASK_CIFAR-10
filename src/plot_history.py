"""Plot training, validation history curves"""
import argparse
import json
import os

import matplotlib.pyplot as plt


def plot(history_path: str, out_dir: str):
    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    out_path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved training curves to: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--history", default="./models/history.json")
    p.add_argument("--out-dir", default="./reports")
    args = p.parse_args()
    plot(args.history, args.out_dir)
