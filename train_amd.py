import os
import subprocess
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

from timm.utils import ModelEmaV2

# ==============================================================================
# 0. Dataset Download
# ==============================================================================
KAGGLE_DATASET = "merolavtechnology/dermnet-skin40-cleaned-dataset"
BASE_EXTRACT_DIR = "./dermnet-skin40-cleaned-dataset"
DATA_DIR = os.path.join(BASE_EXTRACT_DIR, "kaggle/working/Merged_Dermnet_Skin40")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

if not os.path.exists(TRAIN_DIR):
    print("Downloading dataset...")
    subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET], check=True)
    zip_filename = f"{KAGGLE_DATASET.split('/')[-1]}.zip"
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(BASE_EXTRACT_DIR)
    os.remove(zip_filename)

# ==============================================================================
# 1. Device
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# 2. Hyperparameters
# ==============================================================================
BATCH_SIZE = 32
IMG_SIZE = 384
EPOCHS_PHASE1 = 5
EPOCHS_PHASE2 = 40

HEAD_LR_PHASE1 = 1e-3
HEAD_LR = 3e-4
BACKBONE_LR = 3e-5

EARLY_STOPPING_PATIENCE = 7

# ==============================================================================
# 3. Transforms (derm-friendly)
# ==============================================================================
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(),
    normalize
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize
])

# ==============================================================================
# 4. Data
# ==============================================================================
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

NUM_CLASSES = len(train_dataset.classes)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
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
# 5. Model
# ==============================================================================
model = timm.create_model(
    'vit_large_patch14_dinov2.lvd142m',
    pretrained=True,
    num_classes=NUM_CLASSES,
    img_size=IMG_SIZE
)

model = model.to(device).to(memory_format=torch.channels_last)

ema = ModelEmaV2(model, decay=0.9999)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ==============================================================================
# 6. Training Functions
# ==============================================================================
def train_epoch(loader, optimizer):
    model.train()
    total_loss = 0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        ema.update(model)

        total_loss += loss.item()

    return total_loss / len(loader)


def tta_predict(model, inputs):
    # simple TTA: original + horizontal flip
    outputs = model(inputs)
    flipped = torch.flip(inputs, dims=[3])
    outputs_flip = model(flipped)
    return (outputs + outputs_flip) / 2


def validate():
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device).to(memory_format=torch.channels_last)
            labels = labels.to(device)

            with torch.autocast(device_type="cuda"):
                outputs = tta_predict(ema.module, inputs)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


# ==============================================================================
# 7. Phase 1 (Head Training)
# ==============================================================================
print("\n🔹 Phase 1: Training Head")

for param in model.parameters():
    param.requires_grad = False

for param in model.head.parameters():
    param.requires_grad = True

optimizer = optim.AdamW(model.head.parameters(), lr=HEAD_LR_PHASE1)

best_acc = 0
patience = 0

for epoch in range(EPOCHS_PHASE1):
    loss = train_epoch(train_loader, optimizer)
    acc = validate()

    print(f"[Phase1] Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.2f}%")

# ==============================================================================
# 8. Phase 2 (Full Fine-Tuning)
# ==============================================================================
print("\n🔹 Phase 2: Full Fine-Tuning")

for param in model.parameters():
    param.requires_grad = True

optimizer = optim.AdamW([
    {'params': model.head.parameters(), 'lr': HEAD_LR},
    {'params': model.blocks.parameters(), 'lr': BACKBONE_LR},
    {'params': model.patch_embed.parameters(), 'lr': BACKBONE_LR},
    {'params': model.norm.parameters(), 'lr': BACKBONE_LR},
])

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE2)

for epoch in range(EPOCHS_PHASE2):
    loss = train_epoch(train_loader, optimizer)
    acc = validate()
    scheduler.step()

    print(f"[Phase2] Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        patience = 0

        torch.save({
            "model": ema.module.state_dict(),
            "acc": acc,
            "class_to_idx": train_dataset.class_to_idx
        }, "best_model.pt")

        print(f"✅ New best: {acc:.2f}% saved")
    else:
        patience += 1
        print(f"⚠️ No improvement ({patience}/{EARLY_STOPPING_PATIENCE})")

        if patience >= EARLY_STOPPING_PATIENCE:
            print("⛔ Early stopping triggered")
            break

# ==============================================================================
# Done
# ==============================================================================
print(f"\n🎉 Training complete. Best accuracy: {best_acc:.2f}%")