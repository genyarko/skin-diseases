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

# ==============================================================================
# 0. Automatic Dataset Download (from Kaggle)
# ==============================================================================
KAGGLE_DATASET = "merolavtechnology/dermnet-skin40-cleaned-dataset"
BASE_EXTRACT_DIR = "./dermnet-skin40-cleaned-dataset" 

DATA_DIR = os.path.join(BASE_EXTRACT_DIR, "kaggle/working/Merged_Dermnet_Skin40")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

print("Checking for dataset...")
if not os.path.exists(TRAIN_DIR):
    if not os.path.exists(BASE_EXTRACT_DIR):
        print(f"Downloading {KAGGLE_DATASET} from Kaggle...")
        try:
            subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET], check=True)
            zip_filename = f"{KAGGLE_DATASET.split('/')[-1]}.zip"
            print(f"Extracting {zip_filename}...")
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(BASE_EXTRACT_DIR)
            os.remove(zip_filename)
        except Exception as e:
            print(f"🚨 Failed to download dataset: {e}")
            exit(1)
else:
    print("✅ Dataset already exists locally.")

# ==============================================================================
# 1. ROCm / AMD GPU Setup
# ==============================================================================
os.environ["HIP_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# ==============================================================================
# 2. Hyperparameters (Aligned with Hackathon Project)
# ==============================================================================
BATCH_SIZE = 128  # Pushing higher since MI300X has 192GB VRAM
PHASE_1_EPOCHS = 3
PHASE_2_EPOCHS = 12
HEAD_LR = 1e-3
BACKBONE_LR = 5e-5

# ==============================================================================
# 3. Data Transformations (Light RandAugment + RandomErasing)
# ==============================================================================
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Using Timm's advanced augmentation strategy (RandAugment)
from timm.data import create_transform
train_transform = create_transform(
    input_size=224,
    is_training=True,
    auto_augment='rand-m9-mstd0.5-inc1', # Light RandAugment
    interpolation='bicubic',
    re_prob=0.25, # Random Erasing
    re_mode='pixel',
    re_count=1,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

# ==============================================================================
# 4. Dataset Loading & Class Imbalance Handling
# ==============================================================================
print("\nLoading datasets into PyTorch...")
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)

NUM_CLASSES = len(train_dataset.classes)
print(f"Loaded {len(train_dataset)} training images across {NUM_CLASSES} classes.")

class_counts = [0] * NUM_CLASSES
for _, label in train_dataset.samples:
    class_counts[label] += 1

class_weights = 1.0 / np.array(class_counts)
sample_weights = [class_weights[label] for _, label in train_dataset.samples]

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=8, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

# ==============================================================================
# 5. Model Initialization (DINOv2 via timm)
# ==============================================================================
print("Initializing DINOv2-Large model...")
model = timm.create_model(
    'vit_large_patch14_dinov2.lvd142m', 
    pretrained=True, 
    num_classes=NUM_CLASSES
)

model = model.to(device)

# Label-smoothed cross-entropy
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ==============================================================================
# 6. Two-Phase Training Loop
# ==============================================================================
print("\n🚀 Starting Two-Phase Training on AMD MI300X...")
best_val_acc = 0.0

def train_epoch(epoch, num_epochs, phase_name, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0); correct += predicted.eq(labels).sum().item()
        
        if i % 10 == 0:
            print(f"[{phase_name}] Epoch {epoch+1}/{num_epochs}, Step {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
    print(f"--- Epoch {epoch+1} Train Acc: {100.*correct/total:.2f}% ---")

def validate(epoch):
    global best_val_acc
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(inputs)
            _, predicted = outputs.max(1)
            val_total += labels.size(0); val_correct += predicted.eq(labels).sum().item()
            
    val_acc = 100. * val_correct / val_total
    print(f"Validation Acc: {val_acc:.2f}%\n")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print(f"🌟 New best! Saving weights to best_model.pt...")
        torch.save({'model_state_dict': model.state_dict(), 'class_to_idx': train_dataset.class_to_idx}, "best_model.pt")

# --- PHASE 1: Linear Probe (Train Head Only) ---
print("\n--- PHASE 1: Linear Probing (Head Only) ---")
for param in model.parameters(): param.requires_grad = False
for param in model.head.parameters(): param.requires_grad = True

optimizer_phase1 = optim.AdamW(model.head.parameters(), lr=HEAD_LR)

for epoch in range(PHASE_1_EPOCHS):
    train_epoch(epoch, PHASE_1_EPOCHS, "PHASE 1", optimizer_phase1)
    validate(epoch)

# --- PHASE 2: Full Fine-Tune (Discriminative LRs) ---
print("\n--- PHASE 2: Full Fine-Tuning (Discriminative LRs) ---")
for param in model.parameters(): param.requires_grad = True

# Backbone gets tiny LR, head gets larger LR
optimizer_phase2 = optim.AdamW([
    {'params': model.blocks.parameters(), 'lr': BACKBONE_LR},
    {'params': model.head.parameters(), 'lr': HEAD_LR}
])

for epoch in range(PHASE_2_EPOCHS):
    train_epoch(epoch, PHASE_2_EPOCHS, "PHASE 2", optimizer_phase2)
    validate(epoch)

print("🎉 Training Complete!")
