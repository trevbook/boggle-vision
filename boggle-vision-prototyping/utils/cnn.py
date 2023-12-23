# Pytorch-related import statements
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
from utils.settings import allowed_boggle_tiles


class BoggleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # First Conv layer with ReLU and Max-Pooling
            nn.Conv2d(1, 8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),
            # Second Conv layer with ReLU and Max-Pooling
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),
            # Third Conv layer with ReLU and Max-Pooling
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            # Fully Connected Layer 1 with ReLU
            nn.Linear(32 * 100, 128),
            nn.ReLU(),
            # Fully Connected Layer 2 (Output layer)
            nn.Linear(128, len(allowed_boggle_tiles)),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x


class SaveFeatures(nn.Module):
    """
    This hook will extract the activqaations of a particular layer.
    """

    def __init__(self, module):
        """
        The constructor of the class.
        """
        self.features = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        """
        This function will be called when the hook is triggered.
        """
        self.features = output.cpu().detach().numpy()

    def close(self):
        """
        This function will close the hook.
        """
        self.hook.remove()
