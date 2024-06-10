import pandas as pd
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class OCTDataset(Dataset):
    def __init__(self, dataframe, img_dir, target_size=(256, 256)):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        try:
            img1_t_path = os.path.join(self.img_dir, row['image_at_ti'])
            img2_t_path = os.path.join(self.img_dir, row['LOCALIZER_at_ti'])
            img1_t1_path = os.path.join(self.img_dir, row['image_at_ti+1'])
            img2_t1_path = os.path.join(self.img_dir, row['LOCALIZER_at_ti+1'])

            img1_t = Image.open(img1_t_path).convert('L')
            img2_t = Image.open(img2_t_path).convert('L')
            img1_t1 = Image.open(img1_t1_path).convert('L')
            img2_t1 = Image.open(img2_t1_path).convert('L')

            img1_t = self.transform(img1_t)
            img2_t = self.transform(img2_t)
            img1_t1 = self.transform(img1_t1)
            img2_t1 = self.transform(img2_t1)

            label = row['delta_t']

            return img1_t, img2_t, img1_t1, img2_t1, label

        except Exception as e:
            print(f'Error loading image: {e}')
            return None

def get_dataloader(csv_path, img_dir, target_size=(256, 256), batch_size=32, shuffle=True, num_workers=4):
    dataframe = pd.read_csv(csv_path)
    dataset = OCTDataset(dataframe, img_dir, target_size=target_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader