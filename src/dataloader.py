import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from medmnist import ChestMNIST

class ChestXray14(Dataset):
    def __init__(self, split="train", download=True, size=224, transform=None):
        self.dataset = ChestMNIST(split=split, download=download)
        self.transform = transform

    def __getitem__(self, idx):
        imageData, imageLabel = self.dataset[idx]
        # print(f"Type of imageData: {type(imageData)}")
        
        if isinstance(imageData, Image.Image):
            imageData = np.array(imageData)
        # print(f"Shape of imageData: {imageData.shape}")
        
        if imageData.dtype != np.uint8:
            imageData = imageData.astype(np.uint8)
        imageData = Image.fromarray(imageData).convert('RGB')
        imageLabel = torch.FloatTensor(imageLabel)

        if self.transform is not None:
            imageData = self.transform(imageData)

        return imageData, imageLabel

    def __len__(self):
        return len(self.dataset)

class JSRT(Dataset):
    def __init__(self,):
        pass
    def __getitem__(self, ):
        pass
    def __len__(self,):
        pass
