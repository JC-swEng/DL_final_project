"""

"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from utils.custom_dataset import AffectnetYoloDataset
from torchvision import models
from models.viz_emo import get_model
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from torchviz import make_dot
import numpy as np
import io 
import os
import contextlib

# ==== Config ==== 
# Load config.yaml
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

TEST_DIR = "data/AffectnetYolo/test"
MODEL_PATH = "vizemo.pth"
NUM_CLASSES = config["NUM_CLASSES"]
BATCH_SIZE = config["BATCH_SIZE"]
MODEL_NAME = config["MODEL_NAME"]
DROPOUT_RATE = config["DROPOUT_RATE"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ==== Transforms ====
test_transform = transforms.Compose([
    transforms.Resize((224, 224)), # TO have the image input size of EfficientNet
    transforms.ToTensor(),
])
# test_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])

# ==== Dataset & Loader ====
test_dataset = AffectnetYoloDataset(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==== Load Model ==== 
model = get_model(MODEL_NAME, NUM_CLASSES, DEVICE, DROPOUT_RATE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ==== Display Model Architecture with torchviz ====
# Create a dummy input tensor
dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)

# Generate the model graph
graph = make_dot(model(dummy_input), params=dict(model.named_parameters()))

# Save the graph to a file
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
graph.render(os.path.join(output_dir, "model_b0_4_architecture"), format="png")

# ==== Evaluate ====
all_preds, all_labels = [], []
# flag = True
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        # if flag: 
        #     print(f"Model device: {next(model.parameters()).device}")
        #     print(f"Input tensor device: {images.device}")
        #     flag = False

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# # ==== Display Model Architecture ====
# summary_str = io.StringIO()
# with contextlib.redirect_stdout(summary_str):
#     summary(model, (3, 224, 224)) 

# model_summary = summary_str.getvalue()
# print("\nModel Architecture:")
# print(model_summary)

# with open("outputs/model_test_results.txt", "a") as f:
#     f.write("\nModel b0_4: \n")
#     f.write("\nModel Architecture:")
#     f.write(model_summary)

# ==== Metrics ====
acc = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {acc:.4f}")
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"F1 Score: {f1:.4f}")

# Confusion matrix and classification report
print("\nConfusion Matrix:")
conf_mat = confusion_matrix(all_labels, all_preds)
print(conf_mat)

print("\nClassification Report:")
class_report = classification_report(all_labels, all_preds)
print(class_report)

with open("outputs/model_test_results.txt", "a") as f:
    f.write("\nModel b0_4: \n")
    f.write(f"Test Accuracy: {acc:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(conf_mat))
    f.write("\nClassification Report:\n")
    f.write(class_report)

EMOTION_CLASSES = [
    "Anger", 
    "Contempt", 
    "Disgust", 
    "Fear", 
    "Happy", 
    "Neutral", 
    "Sad", 
    "Surprise"
]

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES)
plt.title("Confusion Matrix - Emotion Classification")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
# plt.show()
plt.savefig("outputs/confusion_matrix_testdata_b0_4.png")