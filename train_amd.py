import os
import subprocess
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import timm

from timm.data import create_transform, Mixup
from timm.loss import SoftTargetCrossEntropy

# ==============================================================================
# 0. Dataset Download
# ==============================================================================
KAGGLE_DATASET = "merolavtechnology/dermnet-skin40-cleaned-dataset"
BASE_EXTRACT_DIR = "./dermnet-skin40-cleaned-dataset"
DATA_DIR = os.path.join(BASE_EXTRACT_DIR, "kaggle/working/Merged_Dermnet_Skin40")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

if not os.path.exists(TRAIN_DIR):
    print(f"Downloading {KAGGLE_DATASET} from Kaggle...")
    subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET], check=True)
    zip_filename = f"{KAGGLE_DATASET.split('/')[-1]}.zip"
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(BASE_EXTRACT_DIR)
    os.remove(zip_filename)

# ==============================================================================
# 1. Device
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# ==============================================================================
# 2. Hyperparameters (FIXED)
# ==============================================================================
BATCH_SIZE = 64
IMG_SIZE = 224

EPOCHS = 40

BACKBONE_LR = 1e-4
HEAD_LR = 5e-4

EARLY_STOPPING_PATIENCE = 7

# ==============================================================================
# 3. Transforms (RESTORED STRONG AUGMENTATION)
# ==============================================================================
train_transform = create_transform(
    input_size=IMG_SIZE,
    is_training=True,
    auto_augment='rand-m9-mstd0.5-inc1',
    interpolation='bicubic',
    re_prob=0.25,
    re_mode='pixel',
    re_count=1,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==============================================================================
# 4. Dataset + Sampler (KEEP — helps imbalance)
# ==============================================================================
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

NUM_CLASSES = len(train_dataset.classes)
print(f"Loaded {len(train_dataset)} training images across {NUM_CLASSES} classes.")

# Class balancing
class_counts = [0] * NUM_CLASSES
for _, label in train_dataset.samples:
    class_counts[label] += 1

class_weights = 1.0 / np.array(class_counts)
sample_weights = [class_weights[label] for _, label in train_dataset.samples]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=8,
    pin_memory=True,
    drop_last=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)

# ==============================================================================
# 5. Mixup (RESTORED — CRITICAL FOR ViT)
# ==============================================================================
mixup_fn = Mixup(
    mixup_alpha=0.2,
    cutmix_alpha=0.0,
    prob=1.0,
    switch_prob=0.0,
    mode='batch',
    label_smoothing=0.1,
    num_classes=NUM_CLASSES
)

train_criterion = SoftTargetCrossEntropy()
val_criterion = nn.CrossEntropyLoss()

# ==============================================================================
# 6. Model
# ==============================================================================
print("Initializing DINOv2-Large model...")
model = timm.create_model(
    'vit_large_patch14_dinov2.lvd142m',
    pretrained=True,
    num_classes=NUM_CLASSES,
    img_size=IMG_SIZE
)

model = model.to(device)

# ==============================================================================
# 7. Optimizer (STRONGER LR)
# ==============================================================================
optimizer = optim.AdamW([
    {'params': model.head.parameters(), 'lr': HEAD_LR},
    {'params': model.blocks.parameters(), 'lr': BACKBONE_LR},
    {'params': model.patch_embed.parameters(), 'lr': BACKBONE_LR},
    {'params': model.norm.parameters(), 'lr': BACKBONE_LR},
])

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ==============================================================================
# 8. Training Functions
# ==============================================================================
def train_epoch(epoch):
    model.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        inputs, targets = mixup_fn(inputs, labels)

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda"):
            outputs = model(inputs)
            loss = train_criterion(outputs, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()

        if i % 20 == 0:
            print(f"Epoch {epoch+1}, Step {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

    scheduler.step()
    return running_loss / len(train_loader)


def validate():
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.autocast(device_type="cuda"):
                outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


# ==============================================================================
# 9. Training Loop
# ==============================================================================
best_acc = 0.0
patience = 0

print("\n🚀 Training started...")

for epoch in range(EPOCHS):
    train_loss = train_epoch(epoch)
    val_acc = validate()

    print(f"\nEpoch {epoch+1}: Loss={train_loss:.4f}, Val Acc={val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        patience = 0

        torch.save({
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'class_to_idx': train_dataset.class_to_idx
        }, "best_model.pt")

        print(f"✅ New best: {val_acc:.2f}% saved")

    else:
        patience += 1
        print(f"⚠️ No improvement ({patience}/{EARLY_STOPPING_PATIENCE})")

        if patience >= EARLY_STOPPING_PATIENCE:
            print("⛔ Early stopping triggered")
            break

print(f"\n🎉 Training complete! Best accuracy: {best_acc:.2f}%")