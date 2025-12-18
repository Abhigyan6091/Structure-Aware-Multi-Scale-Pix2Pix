import torch
import torch.nn.functional as F

def gan_loss(pred, real=True):
    target = torch.ones_like(pred) if real else torch.zeros_like(pred)
    return F.binary_cross_entropy_with_logits(pred, target)
