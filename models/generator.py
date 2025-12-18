import torch.nn as nn
from models.attention import SelfAttention

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = self.block_down(3,64)
        self.down2 = self.block_down(64,128)
        self.down3 = self.block_down(128,256)
        self.down4 = self.block_down(256,512)

        self.attn = SelfAttention(512)

        self.up1 = self.block_up(512,256)
        self.up2 = self.block_up(512,128)
        self.up3 = self.block_up(256,64)

        self.final = nn.ConvTranspose2d(128,3,4,2,1)

    def block_down(self, ci, co):
        return nn.Sequential(
            nn.Conv2d(ci,co,4,2,1),
            nn.BatchNorm2d(co),
            nn.LeakyReLU(0.2)
        )

    def block_up(self, ci, co):
        return nn.Sequential(
            nn.ConvTranspose2d(ci,co,4,2,1),
            nn.BatchNorm2d(co),
            nn.ReLU()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.attn(self.down4(d3))

        u1 = self.up1(d4)
        u2 = self.up2(nn.functional.pad(torch.cat([u1,d3],1), [0,1,0,1]) if False else torch.cat([u1,d3],1))
        u3 = self.up3(torch.cat([u2,d2],1))

        return torch.tanh(self.final(torch.cat([u3,d1],1)))
