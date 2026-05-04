import os
import subprocess
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import timm

# ==============================================================================
# 0. Automatic Dataset Download (from Kaggle)
# ==============================================================================
KAGGLE_DATASET = "merolavtechnology/dermnet-skin40-cleaned-dataset"
BASE_EXTRACT_DIR = "./dermnet-skin40-cleaned-dataset" 

# Because Kaggle zips often preserve absolute paths
DATA_DIR = os.path.join(BASE_EXTRACT_DIR, "kaggle/working/Merged_Dermnet_Skin40")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

print("Checking for dataset...")
if not os.path.exists(TRAIN_DIR):
    print(f"Dataset not found at {TRAIN_DIR}.")
    if not os.path.exists(BASE_EXTRACT_DIR):
        print(f"Downloading {KAGGLE_DATASET} from Kaggle...")
        try:
            subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET], check=True)
            zip_filename = f"{KAGGLE_DATASET.split('/')[-1]}.zip"
            print(f"Extracting {zip_filename}...")
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(BASE_EXTRACT_DIR)
            os.remove(zip_filename)
            print("✅ Dataset successfully downloaded and extracted!")
        except Exception as e:
            print(f"🚨 Failed to download dataset. Ensure 'kaggle' is installed and ~/.kaggle/kaggle.json is configured.")
            print(f"Error: {e}")
            exit(1)
            
    if not os.path.exists(TRAIN_DIR):
        print(f"🚨 Error: Extracted the zip, but could not find the train folder at: {TRAIN_DIR}")
        exit(1)
else:
    print("✅ Dataset already exists locally.")

# ==============================================================================
# 1. ROCm / AMD GPU Setup
# ==============================================================================
os.environ["HIP_VISIBLE_DEVICES"] = "0"

# Note: We removed the torch.backends.miopen.enabled = True flag here.
# While it speeds up standard CNNs, it can cause AttributeError crashes on 
# certain pre-compiled PyTorch ROCm wheels when used with massive Transformers.
# ROCm will still utilize the GPU perfectly without it.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# ==============================================================================
# 2. Hyperparameters
# ==============================================================================
BATCH_SIZE = 32 
EPOCHS = 10
LEARNING_RATE = 1e-4

# ==============================================================================
# 3. Data Transformations
# ==============================================================================
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    normalize
])

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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ==============================================================================
# 5. Model Initialization (DINOv2 via timm)
# ==============================================================================
print("Initializing DINOv2-Large model...")
model = timm.create_model(
    'vit_large_patch14_dinov2.lvd142m', 
    pretrained=True, 
    num_classes=NUM_CLASSES
)

# CRITICAL FIX: Move the model to the AMD GPU immediately upon creation.
model = model.to(device)

for param in model.parameters():
    param.requires_grad = False
for param in model.head.parameters():
    param.requires_grad = True

# ==============================================================================
# 6. Loss & Optimizer
# ==============================================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.head.parameters(), lr=LEARNING_RATE)

# ==============================================================================
# 7. Training Loop
# ==============================================================================
print("\n🚀 Starting Training on AMD MI300X...")
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
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
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
    train_acc = 100. * correct / total
    print(f"--- Epoch {epoch+1} Summary ---")
    print(f"Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
    
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(inputs)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            
    val_acc = 100. * val_correct / val_total
    print(f"Validation Acc: {val_acc:.2f}%\n")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print(f"🌟 New best validation accuracy! Saving weights to best_model.pt...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'class_to_idx': train_dataset.class_to_idx 
        }, "best_model.pt")

print("🎉 Training Complete!")
