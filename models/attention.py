import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.q = nn.Conv2d(c, c//8, 1)
        self.k = nn.Conv2d(c, c//8, 1)
        self.v = nn.Conv2d(c, c, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B,C,H,W = x.shape
        q = self.q(x).view(B,-1,H*W)
        k = self.k(x).view(B,-1,H*W)
        v = self.v(x).view(B,-1,H*W)

        attn = torch.softmax(torch.bmm(q.permute(0,2,1), k), dim=-1)
        out = torch.bmm(v, attn.permute(0,2,1)).view(B,C,H,W)

        return self.gamma * out + x
