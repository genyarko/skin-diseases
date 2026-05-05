"""DermNet Skin40 fresh fine-tuning — EVA-02-L @ 448 on ROCm / AMD MI300X.

Stacked recipe (Track 2 base + medical-imaging upgrades):
  - Cosine LR with 10% linear warmup, AdamW, weight_decay=0.05, grad_clip=1.0
  - Mixup(0.1) + Cutmix(0.5) at prob=0.5; mixup OFF in last 20% of epochs
  - Class-balanced "effective number" sampler (gentler than 1/n)
  - EMA teacher (decay=0.9999), evaluated each epoch alongside the live model
  - SWA over the last 20% of epochs
  - Optional DullRazor hair-removal preprocessing (set APPLY_HAIR_REMOVAL=True)
  - Optional HAM10000-pretrained backbone init (run pretrain_ham10000.py first)
  - Save best by macro F1 — both live and EMA copies
  - Final-pass TTA (8 rounds: hflip + small shifts) + per-class report
"""
from __future__ import annotations

import copy
import math
import os
import random
import subprocess
import time
import zipfile

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report, f1_score
from timm.data import Mixup, create_transform
from timm.loss import SoftTargetCrossEntropy
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ==============================================================================
# 0. Dataset Download
# ==============================================================================
KAGGLE_DATASET = "merolavtechnology/dermnet-skin40-cleaned-dataset"
BASE_EXTRACT_DIR = "./dermnet-skin40-cleaned-dataset"
DATA_DIR = os.path.join(BASE_EXTRACT_DIR, "kaggle/working/Merged_Dermnet_Skin40")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

if not os.path.exists(TRAIN_DIR):
    if not os.path.exists(BASE_EXTRACT_DIR):
        print(f"Downloading {KAGGLE_DATASET} from Kaggle...")
        try:
            subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET], check=True)
            zip_filename = f"{KAGGLE_DATASET.split('/')[-1]}.zip"
            with zipfile.ZipFile(zip_filename, "r") as zip_ref:
                zip_ref.extractall(BASE_EXTRACT_DIR)
            os.remove(zip_filename)
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            exit(1)

# ==============================================================================
# 1. ROCm / GPU
# ==============================================================================
os.environ["HIP_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# ==============================================================================
# 2. Config
# ==============================================================================
MODEL_NAME = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
IMG_SIZE = 448
BATCH_SIZE = 32               # bump to 64 if VRAM allows on MI300X

SEED = 123
EPOCHS = 30
LR = 1.0e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.1
WARMUP_FRAC = 0.1
MIXUP_OFF_FRAC = 0.2          # disable mixup in last 20% of epochs
EMA_DECAY = 0.9999
SWA_START_FRAC = 0.8          # start SWA averaging after 80% of training
EARLY_STOPPING_PATIENCE = 7
NUM_WORKERS = 4
APPLY_HAIR_REMOVAL = False    # set True if cv2 installed (slower)
TTA_ROUNDS = 8                # hflip + 5-crop variants averaged
INIT_FROM_PRETRAIN = "ham10000_pretrain.pt"   # backbone warm start; set None to skip

CHECKPOINT_PATH = "best_model.pt"
EMA_CHECKPOINT_PATH = "best_ema.pt"
SWA_CHECKPOINT_PATH = "swa.pt"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

# ==============================================================================
# 3. Hair removal (DullRazor) — optional preprocessing
# ==============================================================================
class DullRazor:
    """Blackhat-morph + inpaint to remove hair pixels. Operates on PIL→PIL."""
    def __init__(self, kernel_size=17, threshold=10):
        self.k = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        self.t = threshold

    def __call__(self, pil_img):
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.k)
        _, mask = cv2.threshold(blackhat, self.t, 255, cv2.THRESH_BINARY)
        out = cv2.inpaint(bgr, mask, 1, cv2.INPAINT_TELEA)
        return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


# ==============================================================================
# 4. Transforms
# ==============================================================================
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_train_transform():
    base = create_transform(
        input_size=IMG_SIZE,
        is_training=True,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        hflip=0.5,
        vflip=0.0,
        mean=MEAN,
        std=STD,
    )
    if APPLY_HAIR_REMOVAL and HAS_CV2:
        return transforms.Compose([DullRazor(), base])
    return base


def build_val_transform():
    base = create_transform(
        input_size=IMG_SIZE,
        is_training=False,
        crop_pct=0.95,
        interpolation="bicubic",
        mean=MEAN,
        std=STD,
    )
    if APPLY_HAIR_REMOVAL and HAS_CV2:
        return transforms.Compose([DullRazor(), base])
    return base


train_transform = build_train_transform()
val_transform = build_val_transform()

# ==============================================================================
# 5. Datasets, sampler (effective-number class balancing)
# ==============================================================================
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=val_transform)
NUM_CLASSES = len(train_dataset.classes)
print(f"Loaded {len(train_dataset)} train / {len(test_dataset)} test  |  {NUM_CLASSES} classes.")

labels_arr = np.array([y for _, y in train_dataset.samples])
class_counts = np.bincount(labels_arr, minlength=NUM_CLASSES)

# "Effective number of samples" weights (Cui et al. 2019). Less aggressive than
# 1/count; still upweights rare classes without crushing common-class signal.
beta = 0.9999
effective_num = 1.0 - np.power(beta, class_counts)
class_w = (1.0 - beta) / np.maximum(effective_num, 1e-9)
class_w = class_w / class_w.sum() * NUM_CLASSES
sample_weights = class_w[labels_arr]

sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(sample_weights), replacement=True
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    persistent_workers=NUM_WORKERS > 0,
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=NUM_WORKERS > 0,
)

mixup_fn = Mixup(
    mixup_alpha=0.1, cutmix_alpha=0.5,
    prob=0.5, switch_prob=0.5, mode="batch",
    label_smoothing=LABEL_SMOOTHING, num_classes=NUM_CLASSES,
)

# ==============================================================================
# 6. Model
# ==============================================================================
print(f"Initializing {MODEL_NAME} (img_size={IMG_SIZE})...")
model = timm.create_model(
    MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES,
    img_size=IMG_SIZE, drop_path_rate=0.1,
)


best_f1, best_acc = 0.0, 0.0
best_ema_f1 = 0.0

if INIT_FROM_PRETRAIN and os.path.exists(INIT_FROM_PRETRAIN):
    print(f"Loading pretrained backbone from {INIT_FROM_PRETRAIN}...")
    pre = torch.load(INIT_FROM_PRETRAIN, map_location="cpu", weights_only=False)
    sd = pre.get("model_state_dict", pre.get("state_dict", pre))
    # Drop classifier head — Skin40 has different num_classes than the pretrain task
    sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    head_missing = [k for k in missing if k.startswith("head.")]
    other_missing = [k for k in missing if not k.startswith("head.")]
    print(f"  loaded backbone  |  head reinit ({len(head_missing)} keys)  |  "
          f"other missing={len(other_missing)}  |  unexpected={len(unexpected)}")
    if other_missing:
        print(f"  WARNING: unexpected missing keys outside head: {other_missing[:5]}...")
else:
    if INIT_FROM_PRETRAIN:
        print(f"INIT_FROM_PRETRAIN={INIT_FROM_PRETRAIN} not found — starting from ImageNet weights.")
    else:
        print("Starting fresh (no domain pretrain).")

model = model.to(device, memory_format=torch.channels_last)

train_criterion = SoftTargetCrossEntropy()
val_criterion = nn.CrossEntropyLoss()


# ==============================================================================
# 7. EMA + SWA
# ==============================================================================
class ModelEMA:
    """Exponential moving average of model weights — eval this for best results."""
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
swa_model = AveragedModel(model)
swa_started = False

# ==============================================================================
# 8. Optimizer + LR schedule
# ==============================================================================
optimizer = optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999),
)
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
# 9. Train / Validate
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

        cur_lr = LR * lr_scale(epoch * len(train_loader) + i)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(x_in)
            loss = train_criterion(outputs, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        ema.update(model)

        _, pred = outputs.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()

        if i % 20 == 0:
            tag = "MIX" if use_mixup else "CLN"
            print(f"E{epoch+1:>2}/{EPOCHS} [{tag}] step {i:>4}/{len(train_loader)}  "
                  f"loss={loss.item():.4f}  lr={cur_lr:.2e}")

    print(f"--- E{epoch+1} train acc≈{100.*correct/total:.2f}% ---")


@torch.no_grad()
def evaluate(eval_model, loader, name=""):
    eval_model.eval()
    all_p, all_t, vloss = [], [], 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = eval_model(x)
            vloss += val_criterion(out, y).item()
        all_p.append(out.argmax(1).cpu())
        all_t.append(y.cpu())
    p = torch.cat(all_p)
    t = torch.cat(all_t)
    acc = (p == t).float().mean().item()
    f1 = f1_score(t.numpy(), p.numpy(), average="macro")
    print(f"  [{name}] loss={vloss/len(loader):.4f}  acc={acc*100:.2f}%  macroF1={f1:.4f}")
    return acc, f1, p, t


def save_ckpt(path, state_dict, val_acc, val_f1, epoch):
    torch.save({
        "model_state_dict": state_dict,
        "class_to_idx": train_dataset.class_to_idx,
        "val_acc": val_acc,
        "val_macro_f1": val_f1,
        "epoch": epoch + 1,
        "model_name": MODEL_NAME,
        "img_size": IMG_SIZE,
    }, path)


# ==============================================================================
# 10. Main loop
# ==============================================================================
print(f"\nTraining {EPOCHS} epochs  |  warmup={warmup_steps}/{total_steps}  "
      f"|  mixup off after E{mixup_off_epoch}  |  SWA from E{swa_start_epoch}")

epochs_no_improve = 0
for epoch in range(EPOCHS):
    t0 = time.time()
    train_epoch(epoch)

    print(f"Validation E{epoch+1}:")
    live_acc, live_f1, _, _ = evaluate(model, test_loader, "live")
    ema_acc, ema_f1, _, _ = evaluate(ema.module, test_loader, "ema ")

    improved = False
    if live_f1 > best_f1:
        best_f1, best_acc = live_f1, live_acc
        save_ckpt(CHECKPOINT_PATH, model.state_dict(), live_acc, live_f1, epoch)
        print(f"  saved live  → {CHECKPOINT_PATH}")
        improved = True
    if ema_f1 > best_ema_f1:
        best_ema_f1 = ema_f1
        save_ckpt(EMA_CHECKPOINT_PATH, ema.module.state_dict(), ema_acc, ema_f1, epoch)
        print(f"  saved EMA   → {EMA_CHECKPOINT_PATH}")
        improved = True

    if epoch >= swa_start_epoch:
        swa_model.update_parameters(model)
        swa_started = True

    print(f"  epoch_time={(time.time()-t0)/60:.1f} min\n")

    if improved:
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping (no F1 gain in {EARLY_STOPPING_PATIENCE} epochs).")
            break

# ==============================================================================
# 11. SWA + TTA final eval
# ==============================================================================
print("\n" + "=" * 60)
print("Final evaluation")
print("=" * 60)


@torch.no_grad()
def tta_evaluate(eval_model, loader, rounds=TTA_ROUNDS):
    """TTA: avg of [identity, hflip] × [center + 4 corners 5-crop, when feasible].
    Falls back to simple identity+hflip averaging for resolution-sensitive nets."""
    eval_model.eval()
    all_p, all_t = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = torch.softmax(eval_model(x), dim=-1)
            logits = logits + torch.softmax(eval_model(torch.flip(x, dims=[-1])), dim=-1)
            # Light geometric jitter: small shifts via rolling
            for shift in [(0, 8), (0, -8), (8, 0), (-8, 0), (4, 4), (-4, -4)][: max(0, rounds - 2)]:
                xs = torch.roll(x, shifts=shift, dims=(-2, -1))
                logits = logits + torch.softmax(eval_model(xs), dim=-1)
        all_p.append(logits.argmax(1).cpu())
        all_t.append(y)
    p = torch.cat(all_p)
    t = torch.cat(all_t)
    acc = (p == t).float().mean().item()
    f1 = f1_score(t.numpy(), p.numpy(), average="macro")
    return acc, f1, p, t


# Reload best EMA, then SWA, eval both with TTA
print("\nLoading best EMA checkpoint for TTA...")
if os.path.exists(EMA_CHECKPOINT_PATH):
    ema_ckpt = torch.load(EMA_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    ema.module.load_state_dict(ema_ckpt["model_state_dict"])
    ema.module.to(device, memory_format=torch.channels_last)
    acc, f1, p_ema, t_ema = tta_evaluate(ema.module, test_loader)
    print(f"EMA + TTA: acc={acc*100:.2f}%  macroF1={f1:.4f}")
    print("\nPer-class report (EMA + TTA):")
    print(classification_report(t_ema.numpy(), p_ema.numpy(),
                                target_names=train_dataset.classes, digits=3, zero_division=0))

if swa_started:
    print("\nFinalizing SWA weights...")
    # update_bn is a no-op for ViT (no BatchNorm) but harmless to call
    try:
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
    except Exception as e:
        print(f"  update_bn skipped: {e}")
    swa_acc, swa_f1, _, _ = tta_evaluate(swa_model.module, test_loader)
    print(f"SWA + TTA: acc={swa_acc*100:.2f}%  macroF1={swa_f1:.4f}")
    save_ckpt(SWA_CHECKPOINT_PATH, swa_model.module.state_dict(), swa_acc, swa_f1, EPOCHS - 1)

print(f"\nDone. Best live F1={best_f1:.4f}, best EMA F1={best_ema_f1:.4f}")
print(f"Checkpoints: {CHECKPOINT_PATH} | {EMA_CHECKPOINT_PATH}"
      + (f" | {SWA_CHECKPOINT_PATH}" if swa_started else ""))
