# -*- coding: utf-8 -*-
from PIL import Image
from torch.utils.data import Dataset
import os


class MNISTDataset(Dataset):
    def __init__(self, root='./', train=False, transform=None, img=None):
        self.root_dir = root
        self.image = img
        self.mnist = os.listdir(root)
        self.is_train = train
        self.transform = transform

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image = None
        label = 0
        if self.is_train:
            img_name = os.path.join(self.root_dir, self.mnist[idx])
            label = self.mnist[idx].split('.')[0][-1]
            image = Image.open(img_name)
        else:
            image = self.image
        if self.transform:
            image = self.transform(image)
        return image, int(label)
