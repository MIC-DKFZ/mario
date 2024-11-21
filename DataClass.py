import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MarioData(Dataset):
    def __init__(self, ID_list, data_frame, data_path, transform=None):
        self.new_Frame = data_frame[data_frame['id_patient'].isin(ID_list)]
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return self.new_Frame.shape[0]

    def __getitem__(self, item):
        row = self.new_Frame.iloc[item]
        im_ti = row['image_at_ti']
        im_ti_1 = row['image_at_ti+1']

        gt_label = row['label']
        im_ti = np.load(os.path.join(self.data_path, f"{os.path.splitext(im_ti)[0]}.npy"))
        im_ti_1 = np.load(os.path.join(self.data_path, f"{os.path.splitext(im_ti_1)[0]}.npy"))

        label = torch.zeros(4)
        if self.transform is not None:
            im_ti = self.transform(im_ti)
            im_ti_1 = self.transform(im_ti_1)

        label[int(gt_label)] = 1

        return im_ti, im_ti_1, label


class MarioDataTask2(Dataset):
    def __init__(self, ID_list, DataFrame, DataPath, transform=None):
        self.new_Frame = DataFrame[DataFrame['id_patient'].isin(ID_list)]
        self.DataPath = DataPath
        self.transform = transform

    def __len__(self):
        return self.new_Frame.shape[0]

    def __getitem__(self, item):
        row = self.new_Frame.iloc[item]
        img = row['image']

        gt_label = row['label']
        img = np.load(os.path.join(self.DataPath, f"{os.path.splitext(img)[0]}.npy"))

        label = torch.zeros(3)
        if self.transform is not None:
            img = self.transform(img)

        label[int(gt_label)] = 1

        return img, label


class MarioValData(Dataset):
    def __init__(self, DataFrame, DataPath, transform=None):
        self.new_Frame = DataFrame
        self.DataPath = DataPath
        self.transform = transform

    def __len__(self):
        return self.new_Frame.shape[0]

    def __getitem__(self, item):
        row = self.new_Frame.iloc[item]
        im_ti = row['image_at_ti']
        im_ti_1 = row['image_at_ti+1']
        case = row['case']
        im_ti = np.load(os.path.join(self.DataPath, f"{os.path.splitext(im_ti)[0]}.npy"))
        im_ti_1 = np.load(os.path.join(self.DataPath, f"{os.path.splitext(im_ti_1)[0]}.npy"))
        size1 = im_ti.size
        size2 = im_ti_1.size

        if self.transform is not None:
            im_ti = self.transform(im_ti)
            im_ti_1 = self.transform(im_ti_1)
        return im_ti, im_ti_1, case


class MarioValDataTask2(Dataset):
    def __init__(self, DataFrame, DataPath, transform=None):
        self.new_Frame = DataFrame
        self.DataPath = DataPath
        self.transform = transform

    def __len__(self):
        return self.new_Frame.shape[0]

    def __getitem__(self, item):
        row = self.new_Frame.iloc[item]
        image = row['image']
        case = row['case']
        image = np.load(os.path.join(self.DataPath, f"{os.path.splitext(image)[0]}.npy"))

        if self.transform is not None:
            image = self.transform(image)
        return image, case


class mAEData(torch.utils.data.Dataset):
    def __init__(self, cases, frame, data_path, transform=None):
        self.cases = cases
        self.frame = frame
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        img_id = self.frame['File Path'][idx]
        img_path = os.path.join(self.data_path, f"{img_id}")

        try:
            img = Image.open(img_path)
        except FileNotFoundError:
            print(f"File not found: {img_path}")
            raise

        # Ensure image is in RGB format
        if img.mode != 'L':
            img = img.convert('L')

        if self.transform:
            img = self.transform(img)

        return img
