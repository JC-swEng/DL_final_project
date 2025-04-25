"""

"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils.custom_dataset import AffectnetYoloDataset
from torchvision import models
from models.viz_emo import get_model
import torch.nn as nn
import yaml

# ==== Config ==== This should be in our config file
# Load config.yaml
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

TEST_DIR = "data/AffectnetYolo/test"
MODEL_PATH = "vizemo.pth"
NUM_CLASSES = config["NUM_CLASSES"]
BATCH_SIZE = config["BATCH_SIZE"]
MODEL_NAME = config["MODEL_NAME"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ==== Transforms ====
test_transform = transforms.Compose([
    transforms.Resize((224, 224)), # TO have the image input size of EfficientNet
    transforms.ToTensor(),
])

# ==== Dataset & Loader ====
test_dataset = AffectnetYoloDataset(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==== Load Model ==== 
model = get_model(MODEL_NAME, NUM_CLASSES, DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ==== Evaluate ====
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ==== Metrics ====
acc = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {acc:.4f}")

# Confusion matrix and classification report
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("\nClassification Report:")
print(classification_report(all_labels, all_preds))
