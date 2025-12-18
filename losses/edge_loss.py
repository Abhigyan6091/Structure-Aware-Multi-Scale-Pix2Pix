import torch
import torch.nn.functional as F

def edge_loss(fake, real):
    sobel = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], device=fake.device, dtype=torch.float32).view(1,1,3,3)
    fx = F.conv2d(fake.mean(1,True), sobel, padding=1)
    rx = F.conv2d(real.mean(1,True), sobel, padding=1)
    return F.l1_loss(fx, rx)
