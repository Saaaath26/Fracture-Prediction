"""Optional architectures. Importing this module does not instantiate or train models."""

import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, channels=1, width=32):
        super().__init__()

        def block(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(b, b, 3, padding=1),
                nn.ReLU(),
            )

        self.enc1 = block(channels, width)
        self.enc2 = block(width, width * 2)
        self.bridge = block(width * 2, width * 4)
        self.dec2 = block(width * 6, width * 2)
        self.dec1 = block(width * 3, width)
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.bridge(F.max_pool2d(b, 2))
        d = self.dec2(
            torch.cat(
                [
                    F.interpolate(
                        c, size=b.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    b,
                ],
                1,
            )
        )
        e = self.dec1(
            torch.cat(
                [
                    F.interpolate(
                        d, size=a.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    a,
                ],
                1,
            )
        )
        return self.head(e)


class SiameseCNN(nn.Module):
    """Experimental pair classifier; pair labels need independently verified semantics."""

    def __init__(self, embedding=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, embedding),
        )
        self.head = nn.Linear(embedding, 1)

    def forward(self, a, b):
        return self.head(torch.abs(self.encoder(a) - self.encoder(b))).squeeze(1)
