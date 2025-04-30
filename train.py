"""
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score
from utils.custom_dataset import AffectnetYoloDataset  # import the custom dataset class
from utils.visualize import plot_training_curves, show_bad_predictions
from models.viz_emo import get_model
import yaml

# === add for tensorboard ====
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/viz_emo_experiment")


# ==== Config ==== This should be in our config file

# Load config.yaml

with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# config value
BATCH_SIZE = config["BATCH_SIZE"]
NUM_EPOCHS = config["NUM_EPOCHS"]
LEARNING_RATE = config["LEARNING_RATE"]
NUM_CLASSES = config["NUM_CLASSES"]
MODEL_NAME = config["MODEL_NAME"]
DROPOUT_RATE = config["DROPOUT_RATE"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ==== Transforms ==== Need to be modular with all possible transformation and augementation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)), #TO have the image input size of EfficientNet
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# train_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.RandomCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2),
#     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
# Validation transforms
# valid_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])

# ==== Dataset & DataLoader ====
train_dataset = AffectnetYoloDataset("data/AffectnetYolo/train", transform=train_transform)
val_dataset   = AffectnetYoloDataset("data/AffectnetYolo/valid", transform=valid_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==== Model ==== we can modified models/viz_emo.py to customized specific model

model = get_model(MODEL_NAME, NUM_CLASSES, DEVICE, DROPOUT_RATE)
model = model.to(DEVICE)

# ==== Optimizer & Loss ==== Maybe having different optimizer and loss to test?
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==== Scheduler (optional strategies) ====
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
# OR for smooth cosine decay:
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# === for visuals

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

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
        #==== remove if no scheduler
        #scheduler.step()
        #====
        train_loss += loss.item()
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"[Epoch {epoch+1}] Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {acc:.4f}")

    # ==== Validation ====
    model.eval()
    val_loss = 0.0
    val_preds, val_labels = [], []
    bad_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

            # Collect wrong predictions
            wrong = preds != labels
            if wrong.any():
                bad_images = images[wrong]
                bad_labels = labels[wrong]
                bad_preds_batch = preds[wrong]

                for img, true_label, pred_label in zip(bad_images, bad_labels, bad_preds_batch):
                    bad_preds.append((img.cpu(), true_label.cpu(), pred_label.cpu()))

    val_acc = accuracy_score(val_labels, val_preds)
    print(f"[Epoch {epoch+1}] Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.4f}")


    # === add tensorboard info after each epoch
    writer.add_scalar("Loss/train", train_loss / len(train_loader), epoch)
    writer.add_scalar("Accuracy/train", acc, epoch)
    writer.add_scalar("Loss/val", val_loss / len(val_loader), epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)

    # === update for visuals
    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    train_accuracies.append(acc)
    val_accuracies.append(val_acc)

# === our own custom visuals to include below? maybe!
plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, "Model b0_4, BS 32, LR 0.000148, DR 0.691")

# ==== Visualize bad predictions ====
if len(bad_preds) > 0:
    show_bad_predictions(bad_preds, class_names=["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Suprise"])

# ==== Save Model ====
torch.save(model.state_dict(), "vizemo.pth")
print("Model saved")
