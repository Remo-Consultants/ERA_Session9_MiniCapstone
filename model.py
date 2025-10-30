# model.py

import torch
import torch.nn as nn
import torchvision.models as models

def build_resnet50_standard(num_classes, device):
    """
    Builds a standard ResNet50 model (not pre-trained) with output layer for full ImageNet.
    Args:
        num_classes (int): number of output classes (1000 for ImageNet)
        device (str or torch.device): the device to move the model to

    Returns:
        nn.Module: ResNet50 model on specified device
    """
    # Create a standard (untrained) ResNet50
    model = models.resnet50(weights=None)
    # Replace the final fully connected layer to suit number of classes (ImageNet = 1000)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)
    return model

def count_parameters(model):
    """
    Counts the number of trainable parameters in the model.

    Args:
        model (nn.Module): The PyTorch model instance.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)