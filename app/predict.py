"""CLI inference tool: classify an arbitrary image using the trained CIFAR-10 model."""
import argparse
import sys

import torch
from PIL import Image
from torchvision import transforms

sys.path.append(".")
from src.dataset import CIFAR10_MEAN, CIFAR10_STD, CLASSES
from src.model import CIFAR10CNN


def load_model(checkpoint_path: str, device: torch.device) -> CIFAR10CNN:
    model = CIFAR10CNN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def predict_image(model, image_path: str, device: torch.device, top_k: int = 3):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    top_probs, top_idx = probs.topk(top_k)
    return [(CLASSES[i], p.item()) for i, p in zip(top_idx, top_probs)]


def main():
    p = argparse.ArgumentParser(description="Classify an image as a CIFAR-10 class")
    p.add_argument("image", help="Path to an image file")
    p.add_argument("--checkpoint", default="./models/best_model.pt")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    results = predict_image(model, args.image, device, args.top_k)

    print(f"\nPredictions for {args.image}:")
    for label, prob in results:
        print(f"  {label:12s} {prob * 100:5.2f}%")


if __name__ == "__main__":
    main()
