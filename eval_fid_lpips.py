from pytorch_fid import fid_score
import lpips
from train import G, val_loader, device
import torchvision.utils as vutils
import shutil, os

os.makedirs("fid_real", exist_ok=True)
os.makedirs("fid_fake", exist_ok=True)

shutil.rmtree("fid_real"); shutil.rmtree("fid_fake")
os.makedirs("fid_real"); os.makedirs("fid_fake")

count=0
G.eval()

with torch.no_grad():
    for x,y in val_loader:
        fake = G(x.to(device))
        for i in range(fake.size(0)):
            vutils.save_image(y[i]*0.5+0.5, f"fid_real/{count}.png")
            vutils.save_image(fake[i]*0.5+0.5, f"fid_fake/{count}.png")
            count+=1

fid = fid_score.calculate_fid_given_paths(["fid_real","fid_fake"], batch_size=16, device=device)
print("FID:", fid)

lp = lpips.LPIPS(net='alex').to(device)
lp_val=0;count=0
with torch.no_grad():
    for x,y in val_loader:
        fake = G(x.to(device))
        lp_val += lp(fake, y.to(device)).mean().item()
        count+=1
print("LPIPS:", lp_val/count)
