"""BoggleCNN architecture and batch inference."""

import numpy as np
import torch
import torch.nn as nn

from .constants import CLASS_LABELS


class BoggleCNN(nn.Module):
    """V0 BoggleCNN: 3-layer conv feature extractor → 2-layer classifier.

    Input: (N, 1, 100, 100) float32 in [0, 255] range.
    Output: (N, 32) logits over CLASS_LABELS.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.15),
            nn.Conv2d(8, 16, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.15),
            nn.Conv2d(16, 32, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 100, 128), nn.ReLU(),
            nn.Linear(128, len(CLASS_LABELS)),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def predict_tiles_batch(model, preprocessed_tiles):
    """Run CNN inference on a list of preprocessed tile arrays.

    Args:
        model: BoggleCNN in eval mode.
        preprocessed_tiles: list of (H, W) uint8 numpy arrays (output of preprocess_tile_v0).

    Returns:
        (letters, confidences) — parallel lists of str and float.
    """
    batch = np.stack(preprocessed_tiles).astype(np.float32)
    tensor = torch.from_numpy(batch).unsqueeze(1)  # (N, 1, H, W)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confs, idxs = probs.max(dim=1)
    letters = [CLASS_LABELS[i.item()] for i in idxs]
    return letters, confs.tolist()
