import torchvision.utils as vutils

def save_sample(fake, real, path):
    vutils.save_image(fake, f"{path}/fake.png")
    vutils.save_image(real, f"{path}/real.png")
