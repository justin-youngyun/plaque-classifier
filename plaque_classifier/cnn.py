#!/usr/bin/env python3
"""
Binary CNN: Plaque vs NotPlaque.

The morphology classifier only makes sense on real plaques, so before subtyping I
run a ResNet50 that decides plaque vs not-plaque and drops everything the
segmenter picked up that is not actually a plaque (vessels, debris, folds). It is
a standard ImageNet-pretrained ResNet50 with a 2-way head, trained with a
weighted sampler for the class imbalance, heavy augmentation, cosine LR, and
early stopping on validation F1.

This is the one part of the pipeline that needs PyTorch. Everything else runs on
numpy / scipy / scikit-image / scikit-learn. The import is guarded so the rest of
the package works even when torch is not installed; calling any function here
without torch raises a clear error.
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    Dataset = object  # so the class body below still parses without torch


def require_torch():
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch is required for the CNN stage. Install it with "
            "`pip install torch torchvision`, then rerun. The feature + LDA "
            "pipeline does not need it."
        )


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Labels that mean "not a plaque"; anything else is treated as a plaque.
NOTPLAQUE_ALIASES = {'0', 'notplaque', 'notaplaque', 'nonplaque', 'np', 'background', 'bg'}


class TileDataset(Dataset):
    """Loads tile image files and applies a torchvision transform."""

    def __init__(self, files, labels, transform=None):
        self.files, self.labels, self.transform = files, labels, transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


def build_resnet50(n_classes=2, device='cpu'):
    """ImageNet-pretrained ResNet50 with a dropout + linear head of n_classes."""
    require_torch()
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    except Exception:
        model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, n_classes))
    return model.to(device)


def _binary_label(raw):
    s = str(raw).strip().lower()
    s = ''.join(ch for ch in s if ch not in ' _-')
    return 0 if s in NOTPLAQUE_ALIASES else 1


def train_binary_cnn(tiles_dir, labels_csv, epochs=50, batch=32, lr=1e-4, seed=42,
                     out_path=None, logger=print):
    """Train the binary plaque / not-plaque ResNet50.

    labels_csv needs a column 'tile' (a filename inside tiles_dir) and a column
    'label'. Any label in NOTPLAQUE_ALIASES becomes class 0, everything else
    class 1. Splits are made at the source-image level when an 'image' column is
    present so tiles from one image never straddle train and validation.
    """
    require_torch()
    t0 = time.time()
    from sklearn.metrics import classification_report, accuracy_score, f1_score

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df = pd.read_csv(labels_csv)
    df['Binary'] = df['label'].apply(_binary_label)
    df['TileLocal'] = df['tile'].apply(lambda p: os.path.join(tiles_dir, os.path.basename(str(p))))
    df = df[df['TileLocal'].apply(os.path.isfile)].copy()
    if 'image' not in df.columns:
        df['image'] = df['tile']

    n_np = int((df['Binary'] == 0).sum())
    n_pl = int((df['Binary'] == 1).sum())
    logger(f"  NotPlaque: {n_np}   Plaque: {n_pl}   Total: {len(df)}")

    # Image-level 80/20 split.
    images = df['image'].unique()
    np.random.shuffle(images)
    target_train = int(0.80 * len(df))
    train_imgs, val_imgs, cum = [], [], 0
    for img in images:
        nt = int((df['image'] == img).sum())
        if cum < target_train:
            train_imgs.append(img)
            cum += nt
        else:
            val_imgs.append(img)
    if not val_imgs and len(train_imgs) > 1:
        val_imgs.append(train_imgs.pop())
    dtr = df[df['image'].isin(train_imgs)]
    dva = df[df['image'].isin(val_imgs)]
    logger(f"  Train: {len(dtr)}   Val: {len(dva)}")

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.RandomAffine(0, translate=(0.08, 0.08), scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = TileDataset(dtr['TileLocal'].tolist(), dtr['Binary'].tolist(), train_tf)
    val_ds = TileDataset(dva['TileLocal'].tolist(), dva['Binary'].tolist(), val_tf)
    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)

    model = build_resnet50(2, device)

    # Upweight the smaller class in the loss.
    cw = torch.tensor([n_pl / max(n_np, 1), 1.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1, best_state, patience, no_improve = 0.0, None, 15, 0
    for epoch in range(epochs):
        model.train()
        rloss, correct, total = 0.0, 0, 0
        for imgs, labs in train_ld:
            imgs = imgs.to(device)
            labs = torch.tensor(labs, dtype=torch.long).to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            rloss += loss.item() * imgs.size(0)
            correct += out.argmax(1).eq(labs).sum().item()
            total += labs.size(0)
        scheduler.step()

        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for imgs, labs in val_ld:
                out = model(imgs.to(device))
                vp.extend(out.argmax(1).cpu().numpy())
                vl.extend(labs)
        vacc = accuracy_score(vl, vp)
        vf1 = f1_score(vl, vp, average='binary', zero_division=0)
        if (epoch + 1) % 5 == 0 or vf1 > best_f1:
            logger(f"  E{epoch + 1:3d}/{epochs} loss={rloss / max(total,1):.4f} "
                   f"acc={correct / max(total,1):.3f} | val_acc={vacc:.4f} val_F1={vf1:.4f}")
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger(f"  Early stop at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    vp, vl = [], []
    with torch.no_grad():
        for imgs, labs in val_ld:
            out = model(imgs.to(device))
            vp.extend(out.argmax(1).cpu().numpy())
            vl.extend(labs)
    logger("\n=== FINAL VALIDATION ===")
    logger(f"Accuracy: {accuracy_score(vl, vp):.4f}")
    logger(classification_report(vl, vp, target_names=['NotPlaque', 'Plaque'], digits=4, zero_division=0))

    if out_path is None:
        out_path = os.path.join(tiles_dir, 'binary_cnn_resnet50.pth')
    torch.save(model.state_dict(), out_path)
    logger(f"\nSaved: {out_path}   ({(time.time() - t0) / 60:.1f} min)")
    return out_path


def load_cnn(cnn_path, device='cpu'):
    """Load a trained head-2 (or multi-class) ResNet50 checkpoint for inference."""
    require_torch()
    state = torch.load(cnn_path, map_location=device)
    n_classes = state['fc.1.weight'].shape[0]
    model = build_resnet50(n_classes, device)
    model.load_state_dict(state)
    model.eval()
    return model, n_classes


def cnn_is_plaque(model, tile_256, n_classes, device='cpu', plaque_thresh=0.6):
    """True if the CNN calls this tile a plaque.

    Binary head: class 1 is Plaque, gated at plaque_thresh. Multi-class head:
    class 0 is NotPlaque, anything else is a plaque.
    """
    require_torch()
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    t_u8 = (np.clip(tile_256, 0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(np.stack([t_u8] * 3, axis=-1))
    inp = tf(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)
        probs = torch.softmax(logits, dim=1)
        pred = int(logits.argmax(1).item())
    if n_classes == 2:
        return probs[0, 1].item() > plaque_thresh
    return pred != 0
