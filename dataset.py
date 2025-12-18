from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import os

class Pix2PixDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(os.listdir(root))
        self.root = root
        self.tf = T.Compose([
            T.Resize((256,256)),
            T.ToTensor(),
            T.Normalize((0.5,)*3,(0.5,)*3)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self,i):
        img = Image.open(os.path.join(self.root,self.files[i])).convert("RGB")
        w,h = img.size
        x = img.crop((0,0,w//2,h))
        y = img.crop((w//2,0,w,h))
        return self.tf(x), self.tf(y)
