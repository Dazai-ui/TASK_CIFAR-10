# CIFAR-10 Image Classifier

A deep learning image classifier trained on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
dataset (60,000 32x32 color images across 10 classes), with data augmentation,
optimization, detailed evaluation, and a simple Streamlit app for live inference.

## Project structure

```
.
├── src/
│   ├── dataset.py        # Data loading + augmentation pipeline
│   ├── model.py           # CNN architecture (CIFAR10CNN)
│   ├── train.py           # Training loop (OneCycleLR, AMP, early stopping)
│   ├── evaluate.py         # Test-set evaluation, classification report, confusion matrix
│   └── plot_history.py    # Training/validation curve plots
├── app/
│   ├── predict.py          # CLI inference on a single image
│   └── streamlit_app.py    # Web app for interactive image classification
├── models/                # Saved checkpoints + training history
├── reports/                # Evaluation outputs: metrics, plots
├── data/                   # CIFAR-10 dataset (auto-downloaded)
└── requirements.txt
```

## Model

`CIFAR10CNN` is a compact VGG-style convolutional network:
- 6 convolutional layers (3 blocks of 2) with batch normalization and ReLU
- Max-pooling after each block (32→16→8→4 spatial resolution)
- Global average pooling + a 2-layer fully-connected classifier with dropout

## Data preprocessing & augmentation

- Normalization using CIFAR-10 per-channel mean/std
- Training-time augmentation: random crop (with padding), random horizontal
  flip, random rotation, color jitter, and random erasing — to reduce
  overfitting and improve generalization
- Test-time: normalization only (no augmentation), for a fair evaluation

## Optimization techniques used

- SGD with Nesterov momentum and weight decay
- `OneCycleLR` learning-rate scheduling
- Mixed-precision training (`torch.cuda.amp`) when a GPU is available
- Label smoothing in the loss function
- Early stopping on validation accuracy with checkpointing of the best model

## Setup instructions

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The CIFAR-10 dataset is downloaded automatically into `./data` on first run.

## Training instructions

```bash
python -m src.train --epochs 40 --batch-size 128 --lr 0.05
```

Key options (see `python -m src.train --help`):
- `--epochs`, `--batch-size`, `--lr`, `--dropout`, `--patience`
- `--no-augment` to disable training-time augmentation (for comparison)

Training saves the best checkpoint to `models/best_model.pt` and a
`models/history.json` log of per-epoch loss/accuracy.

To visualize training curves:

```bash
python -m src.plot_history --history ./models/history.json --out-dir ./reports
```

## Evaluation

```bash
python -m src.evaluate --checkpoint ./models/best_model.pt --out-dir ./reports
```

This prints overall accuracy and a per-class precision/recall/F1 report, and
saves `reports/classification_report.txt` and `reports/confusion_matrix.png`.

## Performance results

**Overall test accuracy: 92.28%**

| Class  | Precision | Recall | F1-score |
|--------|-----------|--------|----------|
| plane  | 0.9355    | 0.9290 | 0.9323   |
| car    | 0.9603    | 0.9670 | 0.9636   |
| bird   | 0.8939    | 0.8930 | 0.8934   |
| cat    | 0.8327    | 0.8460 | 0.8393   |
| deer   | 0.9115    | 0.9370 | 0.9241   |
| dog    | 0.8896    | 0.8700 | 0.8797   |
| frog   | 0.9463    | 0.9520 | 0.9492   |
| horse  | 0.9616    | 0.9260 | 0.9435   |
| ship   | 0.9430    | 0.9600 | 0.9514   |
| truck  | 0.9556    | 0.9480 | 0.9518   |

See [`reports/RESULTS.md`](reports/RESULTS.md) for full details, training
curves, and the confusion matrix.

## Real-world application: inference

**CLI:**

```bash
python -m app.predict path/to/image.jpg --checkpoint ./models/best_model.pt
```

Example prediction (top-5) for a plane image:

```
Predictions for plane.jpg:
  plane         96.14%
  bird           0.75%
  ship           0.55%
  cat            0.53%
  deer           0.42%
```

**Web app (Streamlit):**

```bash
streamlit run app/streamlit_app.py
```

Upload any image and the app classifies it into one of the 10 CIFAR-10
categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship,
truck) with confidence scores.

## Reproducibility

- All dependencies pinned in `requirements.txt`
- Dataset is fetched directly from the official CIFAR-10 source via
  `torchvision.datasets.CIFAR10`
- Training hyperparameters are exposed as CLI flags; checkpoints and history
  are versioned artifacts independent of code changes
