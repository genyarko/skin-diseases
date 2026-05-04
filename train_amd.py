import os
import subprocess
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

# ==============================================================================
# 0. Dataset
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

print("Train exists:", os.path.exists(TRAIN_DIR))
print("Test exists:", os.path.exists(TEST_DIR))

# ==============================================================================
# 1. Device
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# 2. Hyperparameters
# ==============================================================================
BATCH_SIZE = 64
IMG_SIZE = 224
EPOCHS = 30

BACKBONE_LR = 5e-5
HEAD_LR = 3e-4

EARLY_STOPPING_PATIENCE = 5

# ==============================================================================
# 3. Transforms (SAFE + EFFECTIVE)
# ==============================================================================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ==============================================================================
# 4. Data
# ==============================================================================
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

NUM_CLASSES = len(train_dataset.classes)
print(f"Classes: {NUM_CLASSES}")

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
print("Loading DINOv2...")
model = timm.create_model(
    'vit_large_patch14_dinov2.lvd142m',
    pretrained=True,
    num_classes=NUM_CLASSES,
    img_size=IMG_SIZE
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# ==============================================================================
# 6. Optimizer
# ==============================================================================
optimizer = optim.AdamW([
    {'params': model.head.parameters(), 'lr': HEAD_LR},
    {'params': model.blocks.parameters(), 'lr': BACKBONE_LR},
    {'params': model.patch_embed.parameters(), 'lr': BACKBONE_LR},
    {'params': model.norm.parameters(), 'lr': BACKBONE_LR},
])

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ==============================================================================
# 7. Training
# ==============================================================================
def train_epoch(epoch):
    model.train()
    total_loss = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if i % 20 == 0:
            print(f"Epoch {epoch+1}, Step {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

    scheduler.step()
    return total_loss / len(train_loader)


def validate():
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


# ==============================================================================
# 8. Loop
# ==============================================================================
best_acc = 0
patience = 0

print("\n🚀 Training...")

for epoch in range(EPOCHS):
    loss = train_epoch(epoch)
    acc = validate()

    print(f"\nEpoch {epoch+1}: Loss={loss:.4f}, Val Acc={acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        patience = 0

        torch.save({
            'model_state_dict': model.state_dict(),
            'val_acc': acc,
            'class_to_idx': train_dataset.class_to_idx
        }, "best_model.pt")

        print(f"✅ New best: {acc:.2f}% saved")
    else:
        patience += 1
        print(f"⚠️ No improvement ({patience}/{EARLY_STOPPING_PATIENCE})")

        if patience >= EARLY_STOPPING_PATIENCE:
            print("⛔ Early stopping")
            break

print(f"\n🎉 Done. Best Acc: {best_acc:.2f}%")