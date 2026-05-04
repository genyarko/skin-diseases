import os
import subprocess
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import timm
from timm.data import create_transform, Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2
from collections import OrderedDict

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
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(BASE_EXTRACT_DIR)
            os.remove(zip_filename)
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            exit(1)

# ==============================================================================
# 1. Hardware Setup
# ==============================================================================
os.environ["HIP_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
print(f"VRAM available: ~{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB")

# ==============================================================================
# 2. Hyperparameters (192GB VRAM Optimized)
# ==============================================================================
IMG_SIZE = 518
BATCH_SIZE = 128               # Crank it up — you have 192GB
NUM_WORKERS = 16               # More workers for large batch

PHASE_1_EPOCHS = 30            # Longer head warmup
PHASE_2_EPOCHS = 200           # Train until convergence
HEAD_LR = 1e-3
BACKBONE_LR = 5e-6
WEIGHT_DECAY = 0.05
DROP_PATH = 0.2                # Aggressive regularization for 40 classes
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0
EMA_DECAY = 0.9999             # Slower EMA for stability

EARLY_STOPPING_PATIENCE = 25   # Very patient — 192GB means train properly

# ==============================================================================
# 3. Transforms (Heavy Aug for Medical)
# ==============================================================================
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = create_transform(
    input_size=IMG_SIZE,
    is_training=True,
    auto_augment='rand-m15-mstd0.5-inc1',
    interpolation='bicubic',
    re_prob=0.5,                 # Heavy random erasing
    re_mode='pixel',
    re_count=1,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize
])

# ==============================================================================
# 4. Data Loading
# ==============================================================================
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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, 
                          num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                          num_workers=NUM_WORKERS, pin_memory=True)

mixup_fn = Mixup(
    mixup_alpha=0.2, cutmix_alpha=1.0, cutmix_minmax=None, prob=1.0, 
    switch_prob=0.5, mode='batch', label_smoothing=LABEL_SMOOTHING, num_classes=NUM_CLASSES
)

# ==============================================================================
# 5. Focal Loss for Hard Examples (adds ~1-2% on imbalanced medical data)
# ==============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        loss = self.alpha * focal_term * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        return loss

# ==============================================================================
# 6. Model: DINOv2-Giant (1.1B params) — you have the VRAM
# ==============================================================================
print("Initializing DINOv2-GIANT (1.1B params) at 518x518...")

model = timm.create_model(
    'vit_giant_patch14_dinov2.lvd142m',   # 1.1B parameter model
    pretrained=True, 
    num_classes=NUM_CLASSES,
    img_size=IMG_SIZE,
    drop_path_rate=DROP_PATH,
)

# --- RESUME LOGIC ---
best_val_acc = 0.0
best_ema_acc = 0.0
start_epoch = 0
checkpoint_path = "best_model.pt"
ema_checkpoint_path = "best_ema_model.pt"

if os.path.exists(checkpoint_path):
    print(f"Found existing checkpoint: {checkpoint_path}. Resuming!")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        best_val_acc = checkpoint.get('val_acc', 0.0)
    else:
        model.load_state_dict(checkpoint)

model = model.to(device)

# EMA
model_ema = ModelEmaV2(model, decay=EMA_DECAY, device=device)
if os.path.exists(ema_checkpoint_path):
    ema_ckpt = torch.load(ema_checkpoint_path, map_location='cpu')
    if 'state_dict' in ema_ckpt:
        model_ema.module.load_state_dict(ema_ckpt['state_dict'])

# Losses
train_criterion = SoftTargetCrossEntropy()  # For mixup/cutmix
val_criterion = nn.CrossEntropyLoss()
focal_criterion = FocalLoss(gamma=2.0)

# ==============================================================================
# 7. Layer-wise LR Decay (24 blocks for Giant? Actually Giant has 40 blocks)
# ==============================================================================
def get_layer_wise_lr_params(model, base_lr, num_layers=40, decay=0.8):
    parameter_groups = OrderedDict()
    
    # Head
    parameter_groups['head'] = {
        'params': list(model.head.parameters()),
        'lr': base_lr * 20  # 20x for fresh head
    }
    
    # Embedding
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    parameter_groups['embed'] = {
        'params': [p for n, p in model.named_parameters() 
                   if ('patch_embed' in n or 'pos_embed' in n or 'cls_token' in n) 
                   and not any(nd in n for nd in no_decay)],
        'lr': base_lr * (decay ** (num_layers + 1))
    }
    
    # Transformer blocks
    for i in range(num_layers):
        block_name = f'blocks.{i}'
        block_params = [p for n, p in model.named_parameters() 
                       if block_name in n and not any(nd in n for nd in no_decay)]
        if block_params:
            parameter_groups[f'block_{i}'] = {
                'params': block_params,
                'lr': base_lr * (decay ** (num_layers - i))
            }
    
    return list(parameter_groups.values())

# ==============================================================================
# 8. Training Functions
# ==============================================================================
def train_epoch(epoch, num_epochs, phase_name, optimizer, scheduler=None, use_focal=False):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, targets = mixup_fn(inputs, labels)
        
        optimizer.zero_grad()
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(inputs)
            
            # Primary loss
            loss = train_criterion(outputs, targets)
            
            # Optional: add focal loss on clean labels for hard examples
            if use_focal and torch.rand(1).item() > 0.5:
                loss = 0.7 * loss + 0.3 * focal_criterion(outputs, labels)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        model_ema.update(model)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if i % 10 == 0:
            print(f"[{phase_name}] Epoch {epoch+1}/{num_epochs}, Step {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
    if scheduler:
        scheduler.step()
        
    print(f"--- Epoch {epoch+1} Train Acc: {100.*correct/total:.2f}% ---")

@torch.no_grad()
def validate(epoch, use_ema=False):
    global best_val_acc, best_ema_acc, epochs_without_improvement, stop_training
    
    eval_model = model_ema.module if use_ema else model
    eval_model.eval()
    
    val_correct, val_total, val_loss = 0, 0, 0.0
    
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = eval_model(inputs)
            loss = val_criterion(outputs, labels)
            
        val_loss += loss.item()
        _, predicted = outputs.max(1)
        val_total += labels.size(0)
        val_correct += predicted.eq(labels).sum().item()
            
    val_acc = 100. * val_correct / val_total
    model_type = "EMA" if use_ema else "Standard"
    print(f"[{model_type}] Val Loss: {val_loss/len(test_loader):.4f} | Val Acc: {val_acc:.2f}%")
    
    if not use_ema and val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'model_state_dict': model.state_dict(), 
            'class_to_idx': train_dataset.class_to_idx,
            'val_acc': val_acc,
            'epoch': epoch
        }, checkpoint_path)
    
    if use_ema and val_acc > best_ema_acc:
        best_ema_acc = val_acc
        epochs_without_improvement = 0
        print(f"🌟 New best EMA! Saving...")
        torch.save({
            'state_dict': model_ema.module.state_dict(),
            'class_to_idx': train_dataset.class_to_idx,
            'val_acc': val_acc,
            'epoch': epoch
        }, ema_checkpoint_path)
    elif use_ema:
        epochs_without_improvement += 1
        print(f"⚠️ No improvement. Counter: {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}")
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\n⛔ Early stopping!")
            stop_training = True

# ==============================================================================
# 9. PHASE 1: Head-Only Warmup
# ==============================================================================
print(f"\n{'='*60}")
print(f"PHASE 1: Head-Only Warmup ({PHASE_1_EPOCHS} epochs)")
print(f"{'='*60}")

for param in model.parameters():
    param.requires_grad = False
for param in model.head.parameters():
    param.requires_grad = True

optimizer_p1 = optim.AdamW(model.head.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
scheduler_p1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p1, T_max=PHASE_1_EPOCHS)

for epoch in range(PHASE_1_EPOCHS):
    train_epoch(epoch, PHASE_1_EPOCHS, "HEAD", optimizer_p1, scheduler_p1)
    validate(epoch, use_ema=True)

print(f"Phase 1 done. Best EMA: {best_ema_acc:.2f}%")

# ==============================================================================
# 10. PHASE 2: Full Fine-Tune with Layer-wise LR
# ==============================================================================
print(f"\n{'='*60}")
print(f"PHASE 2: Full Fine-Tune (up to {PHASE_2_EPOCHS} epochs)")
print(f"{'='*60}")

for param in model.parameters():
    param.requires_grad = True

param_groups = get_layer_wise_lr_params(model, base_lr=BACKBONE_LR, num_layers=40, decay=0.8)
optimizer_p2 = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=PHASE_2_EPOCHS)

epochs_without_improvement = 0
stop_training = False

for epoch in range(PHASE_2_EPOCHS):
    train_epoch(epoch, PHASE_2_EPOCHS, "FULL", optimizer_p2, scheduler_p2, use_focal=True)
    validate(epoch, use_ema=True)
    
    if stop_training:
        break

# ==============================================================================
# 11. Final Results
# ==============================================================================
print(f"\n{'='*60}")
print(f"TRAINING COMPLETE")
print(f"{'='*60}")
print(f"Best Standard: {best_val_acc:.2f}%")
print(f"Best EMA: {best_ema_acc:.2f}%")
print(f"Saved to: {ema_checkpoint_path}")