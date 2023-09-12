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
        super(BoggleCNN, self).__init__()

        self.features = nn.Sequential(
            # First Conv layer with ReLU and Max-Pooling
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Second Conv layer with ReLU and Max-Pooling
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            # Fully Connected Layer 1 with ReLU
            nn.Linear(64 * 23 * 23, 128),
            nn.ReLU(),
            # Fully Connected Layer 2 (Output layer)
            nn.Linear(128, len(allowed_boggle_tiles)),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x


# class EnhancedBoggleCNN(nn.Module):
#     def __init__(self):
#         super(EnhancedBoggleCNN, self).__init__()
        
#         # More Convolutional layers
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
            
#             nn.Conv2d(32, 64, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
            
#             nn.Conv2d(64, 128, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
            
#             nn.Conv2d(128, 256, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2)
#         )
        
#         # Placeholder for the feature map size, to be filled in later
#         self.feature_map_size = None
        
#         # More Fully connected layers
#         self.classifier = nn.Sequential(
#             nn.Linear(1, 256),  # Placeholder size, will adjust dynamically
#             nn.ReLU(),
#             nn.Dropout(0.5),  # Added Dropout to prevent overfitting
            
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.5),  # Added Dropout to prevent overfitting
            
#             nn.Linear(128, len(allowed_boggle_tiles))
#         )
    
#     def forward(self, x):
#         x = self.features(x)
        
#         # Dynamically calculate the feature map size
#         if self.feature_map_size is None:
#             self.feature_map_size = x.size(1) * x.size(2) * x.size(3)
#             self.classifier[0] = nn.Linear(self.feature_map_size, 256)
            
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x


# Here's the complete class with the suggested fix
class EnhancedBoggleCNN(nn.Module):
    def __init__(self):
        super(EnhancedBoggleCNN, self).__init__()
        
        # More Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Placeholder for the feature map size, to be filled in later
        self.feature_map_size = None
        
        # More Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(4096, 256),  # Correct the size based on your saved model
            nn.ReLU(),
            nn.Dropout(0.5),  # Added Dropout to prevent overfitting
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # Added Dropout to prevent overfitting
            
            nn.Linear(128, len(allowed_boggle_tiles))
        )
    
    def forward(self, x):
        x = self.features(x)
        
        # Dynamically calculate the feature map size
        if self.feature_map_size is None:
            self.feature_map_size = x.size(1) * x.size(2) * x.size(3)
            
            # Dynamically adjust if needed
            if self.feature_map_size != 4096:
                self.classifier[0] = nn.Linear(self.feature_map_size, 256)
            
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x