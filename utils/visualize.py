import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import numpy as np

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, title_suffix=""):
    epochs = range(1, len(train_losses) + 1)

    # Plot Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title(f"Loss Curve {title_suffix}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Acc')
    plt.plot(epochs, val_accuracies, label='Validation Acc')
    plt.title(f"Accuracy Curve {title_suffix}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("outputs/plot"+"_"+title_suffix+".png")

def show_bad_predictions(bad_preds, class_names, num_images=12):
    plt.figure(figsize=(15, 10))
    for idx in range(min(num_images, len(bad_preds))):
        img, true_label, pred_label = bad_preds[idx]
        img = img.permute(1, 2, 0).numpy()  # Convert to HWC
        img = np.clip(img, 0, 1)

        plt.subplot(3, 4, idx+1)
        plt.imshow(img)
        plt.title(f"True: {class_names[true_label]} / Pred: {class_names[pred_label]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("outputs/bad_predictions.png")