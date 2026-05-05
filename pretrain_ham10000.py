"""HAM10000 domain-pretraining for EVA-02-L @ 448 — ROCm / AMD MI300X.

Run BEFORE train_amd.py. Produces `ham10000_pretrain.pt` (EMA backbone weights),
which train_amd.py loads via INIT_FROM_PRETRAIN as a warm start. The classifier
head (7 classes here) is dropped — only the backbone transfers to Skin40 (40).

HAM10000 layout (from Kaggle `kmader/skin-cancer-mnist-ham10000`):
  HAM10000_metadata.csv     # image_id, lesion_id, dx, ...
  HAM10000_images_part_1/   # 5000 .jpg
  HAM10000_images_part_2/   # 5015 .jpg
Same lesion_id can appear in multiple rows — split by lesion_id (not random)
to avoid train/val leakage of the same physical lesion.
"""
from __future__ import annotations

import copy
import math
import os
import random
import subprocess
import time
import zipfile
from glob import glob

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from timm.data import Mixup, create_transform
from timm.loss import SoftTargetCrossEntropy
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ==============================================================================
# 0. Dataset Download (Kaggle)
# ==============================================================================
KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"
EXTRACT_DIR = "./ham10000-data"

if not os.path.exists(EXTRACT_DIR):
    print(f"Downloading {KAGGLE_DATASET} from Kaggle...")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET], check=True)
        zip_filename = f"{KAGGLE_DATASET.split('/')[-1]}.zip"
        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        os.remove(zip_filename)
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        exit(1)

# ==============================================================================
# 1. Build image_id → path map and load metadata
# ==============================================================================
all_jpgs = glob(os.path.join(EXTRACT_DIR, "**", "*.jpg"), recursive=True)
print(f"Found {len(all_jpgs)} .jpg files in {EXTRACT_DIR}")
imageid_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in all_jpgs}

meta_candidates = glob(os.path.join(EXTRACT_DIR, "**", "HAM10000_metadata*"), recursive=True)
if not meta_candidates:
    print("HAM10000_metadata.csv not found.")
    exit(1)
df = pd.read_csv(meta_candidates[0])
df["path"] = df["image_id"].map(imageid_to_path)
df = df.dropna(subset=["path"]).reset_index(drop=True)
print(f"Metadata rows with valid image paths: {len(df)}")

CLASS_NAMES = sorted(df["dx"].unique().tolist())  # ['akiec','bcc','bkl','df','mel','nv','vasc']
class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
df["label"] = df["dx"].map(class_to_idx)
NUM_CLASSES = len(CLASS_NAMES)
print(f"Classes: {CLASS_NAMES}")

# ==============================================================================
# 2. Lesion-grouped train/val split (no leakage)
# ==============================================================================
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=123)
train_idx, val_idx = next(gss.split(df, df["label"], groups=df["lesion_id"]))
df_train = df.iloc[train_idx].reset_index(drop=True)
df_val = df.iloc[val_idx].reset_index(drop=True)
print(f"Train: {len(df_train)}  |  Val: {len(df_val)}")
print("Train class counts:")
print(df_train["dx"].value_counts())

# ==============================================================================
# 3. Config — must match train_amd.py for backbone transfer
# ==============================================================================
MODEL_NAME = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
IMG_SIZE = 448
BATCH_SIZE = 32
SEED = 123
EPOCHS = 8                      # HAM10000 is small; long pretrain risks overfit
LR = 5.0e-5                     # gentler than Skin40's 1e-4 — we don't want to overfit features
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.1
WARMUP_FRAC = 0.1
EMA_DECAY = 0.9999
NUM_WORKERS = 4
OUTPUT_PATH = "ham10000_pretrain.pt"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)
os.environ["HIP_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# 4. Dataset / Dataloaders
# ==============================================================================
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_transform = create_transform(
    input_size=IMG_SIZE, is_training=True,
    auto_augment="rand-m9-mstd0.5-inc1", interpolation="bicubic",
    re_prob=0.25, re_mode="pixel", re_count=1,
    hflip=0.5, vflip=0.5,           # dermoscopy has rotational symmetry
    mean=MEAN, std=STD,
)
val_transform = create_transform(
    input_size=IMG_SIZE, is_training=False, crop_pct=0.95,
    interpolation="bicubic", mean=MEAN, std=STD,
)


class HAM10000Dataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        return self.transform(img), int(row["label"])


train_ds = HAM10000Dataset(df_train, train_transform)
val_ds = HAM10000Dataset(df_val, val_transform)

# Effective-number class weights for sampler
labels_arr = df_train["label"].to_numpy()
class_counts = np.bincount(labels_arr, minlength=NUM_CLASSES)
beta = 0.9999
effective_num = 1.0 - np.power(beta, class_counts)
class_w = (1.0 - beta) / np.maximum(effective_num, 1e-9)
class_w = class_w / class_w.sum() * NUM_CLASSES
sample_weights = class_w[labels_arr]

sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    persistent_workers=NUM_WORKERS > 0,
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=NUM_WORKERS > 0,
)

mixup_fn = Mixup(
    mixup_alpha=0.1, cutmix_alpha=0.5,
    prob=0.5, switch_prob=0.5, mode="batch",
    label_smoothing=LABEL_SMOOTHING, num_classes=NUM_CLASSES,
)

# ==============================================================================
# 5. Model + EMA
# ==============================================================================
print(f"Initializing {MODEL_NAME} (img_size={IMG_SIZE}, num_classes={NUM_CLASSES})...")
model = timm.create_model(
    MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES,
    img_size=IMG_SIZE, drop_path_rate=0.1,
).to(device, memory_format=torch.channels_last)


class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach().to(v.dtype), alpha=1 - d)
            else:
                v.copy_(msd[k])


ema = ModelEMA(model, decay=EMA_DECAY)
train_criterion = SoftTargetCrossEntropy()
val_criterion = nn.CrossEntropyLoss()

# ==============================================================================
# 6. Optimizer + LR
# ==============================================================================
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
total_steps = EPOCHS * len(train_loader)
warmup_steps = int(WARMUP_FRAC * total_steps)


def lr_scale(step):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


# ==============================================================================
# 7. Train / Validate
# ==============================================================================
def train_epoch(epoch):
    model.train()
    correct, total = 0, 0
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        x_mix, targets = mixup_fn(x, y)

        cur_lr = LR * lr_scale(epoch * len(train_loader) + i)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x_mix)
            loss = train_criterion(out, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        ema.update(model)

        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()

        if i % 20 == 0:
            print(f"E{epoch+1}/{EPOCHS} step {i:>3}/{len(train_loader)}  "
                  f"loss={loss.item():.4f}  lr={cur_lr:.2e}")

    print(f"--- E{epoch+1} train acc≈{100.*correct/total:.2f}% ---")


@torch.no_grad()
def evaluate(eval_model, name=""):
    eval_model.eval()
    all_p, all_t, vl = [], [], 0.0
    for x, y in val_loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = eval_model(x)
            vl += val_criterion(out, y).item()
        all_p.append(out.argmax(1).cpu())
        all_t.append(y.cpu())
    p = torch.cat(all_p)
    t = torch.cat(all_t)
    acc = (p == t).float().mean().item()
    f1 = f1_score(t.numpy(), p.numpy(), average="macro")
    print(f"  [{name}] loss={vl/len(val_loader):.4f}  acc={acc*100:.2f}%  macroF1={f1:.4f}")
    return acc, f1


# ==============================================================================
# 8. Loop
# ==============================================================================
print(f"\nPretraining {EPOCHS} epochs  |  warmup={warmup_steps}/{total_steps}")
best_ema_f1 = 0.0
best_ema_acc = 0.0

for epoch in range(EPOCHS):
    t0 = time.time()
    train_epoch(epoch)

    print(f"Validation E{epoch+1}:")
    evaluate(model, "live")
    ema_acc, ema_f1 = evaluate(ema.module, "ema ")

    if ema_f1 > best_ema_f1:
        best_ema_f1 = ema_f1
        best_ema_acc = ema_acc
        torch.save({
            "model_state_dict": ema.module.state_dict(),
            "model_name": MODEL_NAME,
            "img_size": IMG_SIZE,
            "num_classes": NUM_CLASSES,
            "class_to_idx": class_to_idx,
            "val_acc": ema_acc,
            "val_macro_f1": ema_f1,
            "epoch": epoch + 1,
            "stage": "ham10000_pretrain",
        }, OUTPUT_PATH)
        print(f"  saved EMA → {OUTPUT_PATH}")

    print(f"  epoch_time={(time.time()-t0)/60:.1f} min\n")

print(f"\nDone. Best EMA  acc={best_ema_acc*100:.2f}%  macroF1={best_ema_f1:.4f}")
print(f"Backbone weights: {OUTPUT_PATH}")
print("Now run `python3 train_amd.py` — it will pick up these weights via INIT_FROM_PRETRAIN.")
