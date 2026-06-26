"""evaluation script for the CIFAR-10 CNN model."""
import argparse
import json
import os

import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import get_dataloaders, CLASSES
from src.model import CIFAR10CNN


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size, augment=False)

    model = CIFAR10CNN().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()

    report = classification_report(all_labels, all_preds, target_names=CLASSES, digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Overall test accuracy: {accuracy:.4f}\n")
    print(report)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "classification_report.txt"), "w") as f:
        f.write(f"Overall test accuracy: {accuracy:.4f}\n\n{report}")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (acc={accuracy:.4f})")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=150)
    print(f"\nSaved report and confusion matrix to: {args.out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained CIFAR-10 model")
    p.add_argument("--checkpoint", default="./models/best_model.pt")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="./reports")
    p.add_argument("--batch-size", type=int, default=128)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
