import torch.nn as nn
from torchvision import models

def get_resnet50(num_classes=5, in_channels=1):
    model = models.resnet50(weights="IMAGENET1K_V2")

    if in_channels == 1:
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def get_resnet34(num_classes=5, in_channels=1):
    model = models.resnet34(weights="IMAGENET1K_V2")

    if in_channels == 1:
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def get_resnet152(num_classes=5, in_channels=1):
    model = models.resnet152(weights="IMAGENET1K_V2")

    if in_channels == 1:
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
