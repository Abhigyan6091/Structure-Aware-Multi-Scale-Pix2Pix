import torch.nn as nn
from torch.nn.utils import spectral_norm

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(6,64,4,2,1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64,128,4,2,1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(128,1,4,1,1)),
        )

    def forward(self, x, y):
        return self.net(torch.cat([x,y], dim=1))
