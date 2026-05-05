"""ConvNeXt V2-Huge @ 384 — second backbone for ensembling with EVA-02.

Different architecture (CNN, not ViT) → independent error patterns → meaningful
ensemble gain. Run after train_amd.py / continue_train.py to produce a 2nd model.

Output: convnextv2_best_ema.pt
"""
from __future__ import annotations

import copy
import math
import os
import random
import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from timm.data import Mixup, create_transform
from timm.loss import SoftTargetCrossEntropy
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets

# ==============================================================================
# Config
# ==============================================================================
BASE_EXTRACT_DIR = "./dermnet-skin40-cleaned-dataset"
DATA_DIR = os.path.join(BASE_EXTRACT_DIR, "kaggle/working/Merged_Dermnet_Skin40")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# ConvNeXt V1-XL — V2-Huge's GRN is bf16-unstable on long runs; V1 has no GRN.
# Same family of CNN inductive bias, similar accuracy, far more reliable.
MODEL_NAME = "convnext_xlarge.fb_in22k_ft_in1k_384"
IMG_SIZE = 384
BATCH_SIZE = 128                   # 200GB+ VRAM — plenty of room. Push to 192 only if epochs/loss look healthy
OUTPUT_PATH = "convnextv2_best_ema.pt"

SEED = 200                         # different seed = different sample order
EPOCHS = 25                        # CNN convergence is similar to ViT here
LR = 1.6e-4                        # sqrt-scaled for BS=128 (from 8e-5 @ BS=32); drop to 1.2e-4 if bouncy
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.1
WARMUP_FRAC = 0.1
MIXUP_OFF_FRAC = 0.2
EMA_DECAY = 0.9999
SWA_START_FRAC = 0.8
EARLY_STOPPING_PATIENCE = 7
NUM_WORKERS = 4

os.environ["HIP_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


set_seed(SEED)

# ==============================================================================
# Data
# ==============================================================================
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = create_transform(input_size=IMG_SIZE, is_training=True,
    auto_augment="rand-m9-mstd0.5-inc1", interpolation="bicubic",
    re_prob=0.25, re_mode="pixel", re_count=1, hflip=0.5, vflip=0.0,
    mean=MEAN, std=STD)
val_tf = create_transform(input_size=IMG_SIZE, is_training=False,
    crop_pct=0.95, interpolation="bicubic", mean=MEAN, std=STD)

train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
val_ds = datasets.ImageFolder(TEST_DIR, transform=val_tf)
NUM_CLASSES = len(train_ds.classes)
print(f"{len(train_ds)} train / {len(val_ds)} val  |  {NUM_CLASSES} classes")

labels_arr = np.array([y for _, y in train_ds.samples])
class_counts = np.bincount(labels_arr, minlength=NUM_CLASSES)
beta = 0.9999
eff = 1.0 - np.power(beta, class_counts)
cw = (1.0 - beta) / np.maximum(eff, 1e-9); cw = cw / cw.sum() * NUM_CLASSES
sample_weights = cw[labels_arr]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    persistent_workers=NUM_WORKERS > 0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0)

mixup_fn = Mixup(mixup_alpha=0.1, cutmix_alpha=0.5, prob=0.5, switch_prob=0.5,
    mode="batch", label_smoothing=LABEL_SMOOTHING, num_classes=NUM_CLASSES)

# ==============================================================================
# Model
# ==============================================================================
print(f"Building {MODEL_NAME} (img_size={IMG_SIZE}, num_classes={NUM_CLASSES})")
model = timm.create_model(MODEL_NAME, pretrained=True,
    num_classes=NUM_CLASSES, drop_path_rate=0.1)
model = model.to(device, memory_format=torch.channels_last)


class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters(): p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        d = self.decay; msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach().to(v.dtype), alpha=1 - d)
            else:
                v.copy_(msd[k])


ema = ModelEMA(model, decay=EMA_DECAY)
swa_model = AveragedModel(model)
train_criterion = SoftTargetCrossEntropy()
val_criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
total_steps = EPOCHS * len(train_loader)
warmup_steps = int(WARMUP_FRAC * total_steps)
mixup_off_epoch = int((1 - MIXUP_OFF_FRAC) * EPOCHS)
swa_start_epoch = int(SWA_START_FRAC * EPOCHS)


def lr_scale(step):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


# ==============================================================================
# Loop
# ==============================================================================
def train_epoch(epoch):
    model.train()
    use_mixup = epoch < mixup_off_epoch
    correct, total = 0, 0
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        if use_mixup:
            x_in, targets = mixup_fn(x, y)
        else:
            x_in = x
            targets = nn.functional.one_hot(y, NUM_CLASSES).float()
            targets = targets * (1 - LABEL_SMOOTHING) + LABEL_SMOOTHING / NUM_CLASSES

        for g in optimizer.param_groups:
            g["lr"] = LR * lr_scale(epoch * len(train_loader) + i)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x_in)
            loss = train_criterion(out, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if GRAD_CLIP: torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        ema.update(model)

        _, p = out.max(1); total += y.size(0); correct += p.eq(y).sum().item()
        if i % 30 == 0:
            tag = "MIX" if use_mixup else "CLN"
            print(f"E{epoch+1}/{EPOCHS} [{tag}] step {i:>4}/{len(train_loader)}  loss={loss.item():.4f}")
    print(f"--- E{epoch+1} train acc≈{100.*correct/total:.2f}% ---")


@torch.no_grad()
def evaluate(eval_model, name=""):
    eval_model.eval()
    p_all, t_all, vl = [], [], 0.0
    for x, y in val_loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = eval_model(x); vl += val_criterion(out, y).item()
        p_all.append(out.argmax(1).cpu()); t_all.append(y.cpu())
    p = torch.cat(p_all); t = torch.cat(t_all)
    acc = (p == t).float().mean().item()
    f1 = f1_score(t.numpy(), p.numpy(), average="macro")
    print(f"  [{name}] loss={vl/len(val_loader):.4f}  acc={acc*100:.2f}%  F1={f1:.4f}")
    return acc, f1


print(f"\nTraining {EPOCHS} epochs  |  warmup={warmup_steps}/{total_steps}  "
      f"|  mixup off after E{mixup_off_epoch}  |  SWA from E{swa_start_epoch}")
best_ema_f1, best_ema_acc, no_improve = 0.0, 0.0, 0
swa_started = False

for epoch in range(EPOCHS):
    t0 = time.time()
    train_epoch(epoch)
    print(f"Validation E{epoch+1}:")
    evaluate(model, "live")
    ema_acc, ema_f1 = evaluate(ema.module, "ema ")

    if ema_f1 > best_ema_f1:
        best_ema_f1 = ema_f1; best_ema_acc = ema_acc; no_improve = 0
        torch.save({
            "model_state_dict": ema.module.state_dict(),
            "class_to_idx": train_ds.class_to_idx,
            "val_acc": ema_acc, "val_macro_f1": ema_f1,
            "epoch": epoch + 1, "model_name": MODEL_NAME, "img_size": IMG_SIZE,
        }, OUTPUT_PATH)
        print(f"  saved → {OUTPUT_PATH}")
    else:
        no_improve += 1
        if no_improve >= EARLY_STOPPING_PATIENCE:
            print("Early stop."); break

    if epoch >= swa_start_epoch:
        swa_model.update_parameters(model); swa_started = True
    print(f"  time={(time.time()-t0)/60:.1f} min\n")

if swa_started:
    print("\nSWA finalize...")
    try:
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
    except Exception as e:
        print(f"  update_bn skipped: {e}")
    swa_acc, swa_f1 = evaluate(swa_model.module, "swa ")
    torch.save({
        "model_state_dict": swa_model.module.state_dict(),
        "class_to_idx": train_ds.class_to_idx,
        "val_acc": swa_acc, "val_macro_f1": swa_f1,
        "model_name": MODEL_NAME, "img_size": IMG_SIZE,
    }, "convnextv2_swa.pt")

print(f"\nDone. Best EMA F1={best_ema_f1:.4f} acc={best_ema_acc*100:.2f}%  →  {OUTPUT_PATH}")
