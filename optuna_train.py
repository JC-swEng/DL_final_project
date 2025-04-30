import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import transforms, models
import yaml
import os
from utils.custom_dataset import AffectnetYoloDataset  
from utils.visualize import plot_training_curves, show_bad_predictions
from models.viz_emo import get_model

with open("outputs/optuna_results.txt", "a") as f:
    f.write("\n\n\nBEGINNING OPTUNA TRIALS: \n\n\n")


def objective(trial):
    # Hyperparameter tuning space
    MODEL_NAME = trial.suggest_categorical("MODEL_NAME", ["efficientnet_b0_4", "efficientnet_b0_5"]) 
    if MODEL_NAME == "efficientnet_b0_4":
        BATCH_SIZE = 32
    elif MODEL_NAME == "efficientnet_b0_5":
        BATCH_SIZE = 64
    NUM_EPOCHS = 5
    NUM_CLASSES = 8
    LEARNING_RATE = trial.suggest_loguniform("LEARNING_RATE", 1e-4, 5e-3)
    DROPOUT_RATE = trial.suggest_uniform("DROPOUT_RATE", 0.0, 0.75)

    # DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # ==== Transforms ====
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # ==== Dataset & DataLoader ====
    train_dataset = AffectnetYoloDataset("data/AffectnetYolo/train", transform=train_transform)
    val_dataset = AffectnetYoloDataset("data/AffectnetYolo/valid", transform=valid_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ==== Model ====
    model = get_model(MODEL_NAME, NUM_CLASSES, DEVICE, DROPOUT_RATE)
    model.to(DEVICE)

    # ==== Optimizer & Loss ====
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ==== Training Loop ====
    best_val_acc = 0.0
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

        train_acc = accuracy_score(all_labels, all_preds)
        print(f"[Epoch {epoch + 1}] Train Loss: {train_loss / len(train_loader):.4f} | Train Acc: {train_acc:.4f}")

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
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f"[Epoch {epoch + 1}] Val Loss: {val_loss / len(val_loader):.4f} | Val Acc: {val_acc:.4f}")

        # Update best validation accuracy
        best_val_acc = max(best_val_acc, val_acc)

    
    # Log trial results into txt file
    trial_result = f"Trial {trial.number}:  MODEL_NAME={MODEL_NAME}, BATCH_SIZE={BATCH_SIZE}, LEARNING_RATE={LEARNING_RATE:.6f}, DROPOUT_RATE={DROPOUT_RATE:.3f}\n  Val Acc={best_val_acc:.4f}\n"
    os.makedirs("outputs", exist_ok=True)

    # Append results to a txt file
    with open("outputs/optuna_results.txt", "a") as f:
        f.write(trial_result)
    
    # Return the negative of the validation accuracy to minimize
    return 1 - best_val_acc

# start the optuna trial process
study = optuna.create_study(direction="minimize")  # minimize the negative validation accuracy
study.optimize(objective, n_trials=15)  # number of optuna trials

# Print best hyperparameters
print("Best hyperparameters:", study.best_params)



