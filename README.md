# MLEProject
# 🔍 SceneSense

**Multi-label object recognition with transfer learning (PyTorch + ResNet50)**

SceneSense looks at a photo and finds **every** object in it from 12 everyday classes. Unlike standard image classification (one image → one label), each image here can contain several objects at once, such as a pen, a book *and* a laptop on the same desk.

```
pen · paper · book · clock · phone · laptop · chair · desk · bottle · keychain · backpack · calculator
```

---

## ✨ Highlights

- **Multi-label classification:** sigmoid outputs with `BCEWithLogitsLoss`, so each class is an independent yes/no decision.
- **Transfer learning:** an ImageNet-pretrained ResNet50 with the early layers frozen. Only `layer4` and a custom head are fine-tuned, which speeds up training and reduces overfitting.
- **Custom classification head:** BatchNorm → Dropout(0.4) → Linear(2048→512) → ReLU → Dropout(0.3) → Linear(512→12).
- **Data augmentation:** random resized crops, flips, rotation, affine shifts, color jitter and grayscale.
- **Handling label sparsity:** `pos_weight=2.0` upweights positive labels, since most images contain only a few of the 12 classes.
- **Training controls:** Adam with weight decay, `ReduceLROnPlateau` scheduling, and early stopping on validation loss.
- **Threshold tuning:** after training, the decision threshold is swept on the validation set to maximize micro-F1.
- **Rich evaluation:** exact match, Hamming accuracy, mean IoU (Jaccard), and micro precision/recall/F1.

---

## 📁 Project Structure

```
.
├── data/                 # Training images: one folder per label combination
├── train.py              # Training pipeline (fine-tuning + threshold sweep)
├── eval.py               # Dataset loader, model definition, metrics, evaluation CLI
├── project_model.pth     # Trained model weights (state_dict)
└── LICENSE
```

## 🗂️ Dataset Format

Folder names encode the labels, joined by `_`. Every image inside a folder gets all of those labels:

```
data/
├── pen/                  → [pen]
├── pen_book/             → [pen, book]
├── phone_laptop_desk/    → [phone, laptop, desk]
└── ...
```

- **~4,500 images** across **313 label combinations**
- Image files must be named `img<id>.png`
- Each image becomes a 12-dimensional multi-hot target vector

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install torch torchvision pillow
```

### 2. Train

```bash
python train.py
```

This will:
1. Split `data/` 80/20 into train/validation (fixed seed `42`)
2. Fine-tune the model for up to 30 epochs, with early stopping
3. Sweep thresholds (0.25 – 0.50) on validation and print the best one
4. Save the weights to `project_model.pth`

### 3. Evaluate

```bash
python eval.py \
  --model_path project_model.pth \
  --test_data path/to/test_data \
  --threshold 0.35 \
  --group_id 0 \
  --project_title "SceneSense"
```

Set `--threshold` to the best value printed by `train.py`.

| Argument | Default | Description |
|---|---|---|
| `--model_path` | *required* | Path to the trained weights |
| `--test_data` | `project_test_data` | Test folder (same layout as `data/`) |
| `--threshold` | `0.5` | Probability cutoff for predicting a label |
| `--image_size` | `128` | Input resolution |
| `--batch_size` | `32` | Batch size |
| `--group_id` | *required* | Project group ID |
| `--project_title` | *required* | Project title |

---

## 🛠️ Tech Stack

Python · PyTorch · torchvision · ResNet50 · PIL

