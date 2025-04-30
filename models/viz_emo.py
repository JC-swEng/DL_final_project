"""
To try different model configuration
"""
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights #, EfficientNet_B7_Weights, ViT_B_16_Weights

def get_model(model_name: str, num_classes: int, device):
    
    weights = EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    if model_name == "efficientnet_b0_1":
        """
        Baseline Fine-Tuning: Only replace the final classification layers

        Use of the EfficientNet_b0, DEFAULT points to the most up-to-date pretrained version on ImageNet.
        Since the last layer has 1000 classes, we  replace the final layer with our number of classes: 8
        """
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        #preprocess = weights.transforms()

    elif model_name == "efficientnet_b0_2":
        """
        Feature Extractor (Freezing All But Final Layer)
        """
        for param in model.features.parameters():
            param.requires_grad = False  # Freeze backbone
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "efficientnet_b0_3":
        """
        Partial unfreezing (last two blocks)
        """
        for name, param in model.named_parameters():
                if "features.6" in name or "features.7" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes, dropout)

    elif model_name == "efficientnet_b0_4":
        """
        Add Dropout Before Final Layer
        """
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
    
    elif model_name == "efficientnet_b0_5":
        """
        Add BatchNorm + Dropout Before Final Layer
        """
        model.classifier = nn.Sequential(
            nn.BatchNorm1d(model.classifier[1].in_features),
            nn.Dropout(dropout), 
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
    else:
        raise ValueError(f"Model '{model_name}' not supported.")

    
    return model