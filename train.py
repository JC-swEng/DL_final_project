"""
Include some GPT Vibe coding to modify
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score
from utils.custom_dataset import CustomImageDataset  # import the custom dataset class

# === add for tensorboard ====
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/viz_emo_experiment")


# ==== Config ==== This should be in our config file
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_CLASSES = 8  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ==== Transforms ==== Need to be modular with all possible transformation and augementation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==== Dataset & DataLoader ====
train_dataset = CustomImageDataset("data/AffectnetYolo/train", transform=train_transform)
val_dataset   = CustomImageDataset("data/AffectnetYolo/valid", transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==== Model ==== This is the part we need make magic happen
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ==== Optimizer & Loss ==== Maybe having different optimizer and loss to test?
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==== Training Loop ====
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"[Epoch {epoch+1}] Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {acc:.4f}")

    # ==== Validation ====
    model.eval()
    val_loss = 0.0
    val_preds, val_labels = [], []

    

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    print(f"[Epoch {epoch+1}] Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.4f}")

    # === add tensorboard info after each epoch
    writer.add_scalar("Loss/train", train_loss / len(train_loader), epoch)
    writer.add_scalar("Accuracy/train", acc, epoch)
    writer.add_scalar("Loss/val", val_loss / len(val_loader), epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)

    # === our own custom visuals to include below? maybe!

# ==== Save Model ====
torch.save(model.state_dict(), "vizemo.pth")
print("Model saved")
