import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

# ROCm / AMD GPU Setup
os.environ["HIP_VISIBLE_DEVICES"] = "0"
torch.backends.miopen.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4

DATA_DIR = "./Skin_Disease_Dataset_Clean" # CHANGE THIS ON YOUR SERVER
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

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

print("Loading datasets...")
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)
NUM_CLASSES = len(train_dataset.classes)

print("Calculating class weights for WeightedRandomSampler...")
class_counts = [0] * NUM_CLASSES
for _, label in train_dataset.samples:
    class_counts[label] += 1

class_weights = 1.0 / np.array(class_counts)
sample_weights = [class_weights[label] for _, label in train_dataset.samples]

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print("Initializing model...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

print("\n🚀 Starting Training on AMD MI300X...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0; correct = 0; total = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0); correct += predicted.eq(labels).sum().item()
        if i % 50 == 0: print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i}], Loss: {loss.item():.4f}")
            
    print(f"--- Epoch {epoch+1} Summary ---\nTrain Loss: {running_loss/len(train_loader):.4f} | Train Acc: {100.*correct/total:.2f}%")
    
    model.eval()
    val_correct = 0; val_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            val_total += labels.size(0); val_correct += predicted.eq(labels).sum().item()
    print(f"Validation Acc: {100.*val_correct/val_total:.2f}%\n")

print("🎉 Training Complete!")
