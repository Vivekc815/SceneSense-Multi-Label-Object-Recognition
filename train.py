"""
Training script for the CAI6108 multilabel image project.

Loads images from `data/` using the same folder layout and `LABEL_ORDER` as
`eval.py` (subdir names are underscore-separated object labels; each image is
a 12-dimensional multi-hot target). Trains a ResNet50-based model with
BCE-with-logits, saves `project_model.pth` (state_dict only), then sweeps a
decision threshold on the validation split so you can pass `--threshold` to
`eval.py` consistently with how the model was tuned.
"""
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import models, transforms

from eval import CustomDirectoryLayoutDataset, LABEL_ORDER


class DatasetWrapper(Dataset):
    """Applies train/val transforms after `random_split` (base dataset returns PIL)."""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def CREATE_YOUR_MODEL_HERE(num_labels=12):
    """
    ResNet50 with a deeper classification head.
    BatchNorm + two Dropout layers + hidden linear layer for better generalization.
    Must stay in sync with the same function in `eval.py` for checkpoint loading.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(num_ftrs),
        nn.Dropout(p=0.4),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_labels)
    )
    return model


def freeze_backbone(model):
    """
    Freeze all layers except layer4 and the classification head (fc).
    This speeds up training and prevents overfitting on early layers.
    """
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False


def find_best_threshold(model, val_loader, device):
    """
    Sweeps thresholds on the validation set and returns the one
    with the best F1 micro score.
    """
    from eval import evaluate_model
    best_thresh = 0.5
    best_f1 = 0.0

    print("\nSweeping thresholds on validation set...")
    for thresh in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        metrics = evaluate_model(model, val_loader, device, threshold=thresh)
        print(f"  Threshold {thresh:.2f} -> F1: {metrics['f1_micro']:.4f} | "
              f"Exact: {metrics['exact_match']:.4f} | "
              f"Recall: {metrics['recall_micro']:.4f} | "
              f"Precision: {metrics['precision_micro']:.4f}")
        if metrics['f1_micro'] > best_f1:
            best_f1 = metrics['f1_micro']
            best_thresh = thresh

    print(f"Best threshold: {best_thresh:.2f} with F1: {best_f1:.4f}\n")
    return best_thresh


def train_model():
    """80/20 train/val split, fine-tune last ResNet block + head, early-stop on val loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # --- Hyperparameters ---
    data_dir = "data"
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    image_size = 128

    # --- Transforms --- (ImageNet normalization matches pretrained ResNet weights)
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # --- Dataset ---
    print("Loading dataset...")
    base_dataset = CustomDirectoryLayoutDataset(root=data_dir, transform=None)
    print(f"Total samples: {len(base_dataset)}")

    train_size = int(0.8 * len(base_dataset))
    val_size = len(base_dataset) - train_size

    # Fixed seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(base_dataset, [train_size, val_size],
                                            generator=generator)

    # Same underlying samples as `base_dataset`; different torchvision pipelines per split
    train_dataset = DatasetWrapper(train_subset, transform=train_transforms)
    val_dataset = DatasetWrapper(val_subset, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)  # pin_memory helps host→GPU copies on CUDA
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # --- Model ---
    model = CREATE_YOUR_MODEL_HERE(num_labels=len(LABEL_ORDER)).to(device)
    freeze_backbone(model)
    print("Backbone frozen — training layer4 + fc only")

    # --- Loss: BCE on logits; pos_weight > 1 upweights positives (typical for sparse multilabel) ---
    pos_weight = torch.ones(len(LABEL_ORDER)).to(device) * 2.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Optimizer: only pass trainable params ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=1e-4)

    # --- Scheduler --- (reduce LR when validation loss stops decreasing)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=4, factor=0.5, verbose=True
    )

    # --- Training Loop ---
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 10

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            correct_labels = 0
            total_labels = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    logits = model(inputs)
                    loss = criterion(logits, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                # Per-bit accuracy at 0.5 (monitoring only; best threshold is chosen after training)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct_labels += (preds == labels).sum().item()
                total_labels += labels.numel()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_hamming = correct_labels / total_labels

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} | '
                  f'Hamming Acc: {epoch_hamming:.4f}')

            if phase == 'val':
                scheduler.step(epoch_loss)  # stepped once per epoch on full-val loss
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Current LR: {current_lr:.6f}')

                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    print("  >> New best model saved!")
                else:
                    epochs_no_improve += 1
                    print(f'  >> No improvement for {epochs_no_improve} epoch(s)')

        # Early stopping on validation loss (counter resets when val improves)
        if epochs_no_improve >= early_stop_patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs.')
            break

    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Loss: {best_val_loss:.4f}')

    # --- Load best val-loss checkpoint; tune probability threshold on val for eval.py ---
    model.load_state_dict(best_model_wts)
    best_thresh = find_best_threshold(model, val_loader, device)

    # --- Save weights only (eval.py rebuilds architecture then load_state_dict) ---
    torch.save(model.state_dict(), 'project_model.pth')
    print(f"Saved best model to 'project_model.pth'")
    print(f"Use threshold={best_thresh:.2f} when running eval.py with --threshold {best_thresh:.2f}")


if __name__ == '__main__':
    train_model()