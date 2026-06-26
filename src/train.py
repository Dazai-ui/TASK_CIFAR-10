"""training and  checkpointing."""
import argparse
import json
import os
import time

import torch
import torch.nn as nn

from src.dataset import get_dataloaders
from src.model import CIFAR10CNN


def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return loss_sum / total, correct / total


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        augment=not args.no_augment,
    )

    model = CIFAR10CNN(dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=5e-4, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
        epochs=args.epochs,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    patience, bad_epochs = args.patience, 0
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        loss_sum, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_sum += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

        train_loss, train_acc = loss_sum / total, correct / total
        val_loss, val_acc = evaluate(model, test_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} | {elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            bad_epochs = 0
            torch.save({"model_state": model.state_dict(), "val_acc": val_acc}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Best checkpoint saved to: {ckpt_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train a CNN on CIFAR-10")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="./models")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--no-augment", action="store_true", help="Disable training-time augmentation")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
