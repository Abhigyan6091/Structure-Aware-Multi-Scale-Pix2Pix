import torch
from torch.utils.data import DataLoader
from models.generator import Generator
from models.discriminator_patch import PatchDiscriminator
from models.discriminator_global import GlobalDiscriminator
from losses.gan_loss import gan_loss
from losses.perceptual_loss import PerceptualLoss
from losses.edge_loss import edge_loss
from dataset import Pix2PixDataset
import torchvision.utils as vutils
import os
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader = DataLoader(Pix2PixDataset("maps/train"), batch_size=4, shuffle=True)
val_loader   = DataLoader(Pix2PixDataset("maps/val"), batch_size=4, shuffle=False)

G = Generator().to(device)
Dp = PatchDiscriminator().to(device)
Dg = GlobalDiscriminator().to(device)

optG = torch.optim.Adam(G.parameters(),2e-4,(0.5,0.999))
optD = torch.optim.Adam(list(Dp.parameters())+list(Dg.parameters()),2e-4,(0.5,0.999))

perc = PerceptualLoss().to(device)
os.makedirs("outputs", exist_ok=True)

for epoch in range(100):
    for x,y in tqdm(train_loader):
        x,y = x.to(device), y.to(device)
        fake = G(x)

        d_loss = (
            gan_loss(Dp(x,y), True) +
            gan_loss(Dp(x,fake.detach()), False) +
            gan_loss(Dg(x,y), True) +
            gan_loss(Dg(x,fake.detach()), False)
        )

        optD.zero_grad(); d_loss.backward(); optD.step()

        g_loss = (
            gan_loss(Dp(x,fake),True) +
            gan_loss(Dg(x,fake),True) +
            10 * torch.nn.functional.l1_loss(fake,y) +
            2 * perc(fake,y) +
            1 * edge_loss(fake,y)
        )

        optG.zero_grad(); g_loss.backward(); optG.step()

    print(f"Epoch {epoch+1}")

    if (epoch+1)%10 == 0:
        fake = G(x)
        vutils.save_image(fake*0.5+0.5, f"outputs/fake_{epoch+1}.png")
        vutils.save_image(y*0.5+0.5, f"outputs/real_{epoch+1}.png")
