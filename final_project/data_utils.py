import glob
import pandas as pd
import random
import torch
from os import listdir
from os.path import basename, dirname, exists, join

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, Grayscale, Resize, ToTensor


def create_img_pairs(data_path, csv_path, phases, LDCT_algo, SDCT_algo, subject=None):
    # read csv data
    csv_data = pd.read_csv(csv_path, dtype={'subject': str})

    # use filters to find LDCT images
    phase_filter = csv_data.phase.isin(phases)
    algo_filter = csv_data.algorithm == LDCT_algo
    dose_filter = csv_data.dose == 50
    # subject filter (optional)
    subject_filter = True if subject is None else csv_data.subject == subject

    # get list of LDCT images
    LDCT_data = csv_data[phase_filter & algo_filter & dose_filter & subject_filter].values
    # build a list of image pairs
    img_pairs = []
    
    for i, row in enumerate(LDCT_data):
        LDCT_path = row[-1]
        SDCT_path = join(data_path, row[0], f'{row[1]}_3mm_{SDCT_algo}_100', basename(LDCT_path))

        if exists(SDCT_path):
            img_pairs.append((LDCT_path, SDCT_path))
            
    return img_pairs


def prepare_data(csv_path, data_path, phases, ldct_algo, sdct_algo):
    # read csv data
    csv_data = pd.read_csv(csv_path, dtype={'subject': str})
    
    # select phases and reconstruction algorithms
    phases =  phases.split(',')
    
    # create image pairs
    img_pairs = create_img_pairs(data_path, csv_path, phases, ldct_algo, sdct_algo)
    # split image pairs into training, validation and testing
    train_img_pairs, val_img_pairs, test_img_pairs = split_img_pairs(img_pairs, 0.1, 0.02)
    
    print(f'Total image pairs: {len(img_pairs)}')
    print(f'Training image pairs: {len(train_img_pairs)}')
    print(f'Validation image pairs: {len(val_img_pairs)}')
    print(f'Testing image pairs: {len(test_img_pairs)}')
    
    return train_img_pairs, val_img_pairs, test_img_pairs
    
    
def split_img_pairs(img_pairs, test_size, val_size, random_state=42):
    # split data into train and test
    train_val_img_pairs, test_img_pairs = train_test_split(img_pairs, test_size=test_size, random_state=random_state)
    # split train into train and validation
    train_img_pairs, val_img_pairs = train_test_split(train_val_img_pairs, test_size=val_size, random_state=random_state)
    
    return train_img_pairs, val_img_pairs, test_img_pairs


class DoubleRandomPatch:
    def __init__(self, img_size, patch_size, mean_value=0.1):
        self.img_height, self.img_width = img_size
        self.patch_height, self.patch_width = patch_size
        self.mean_value = mean_value
        
        
    def __call__(self, LDCT_img, SDCT_img):
        while True:
            top = random.randint(0, self.img_height - self.patch_height)
            left = random.randint(0, self.img_width - self.patch_width)
            
            LDCT_patch = LDCT_img[:, top:top + self.patch_height, left:left + self.patch_width]
            SDCT_patch = SDCT_img[:, top:top + self.patch_height, left:left + self.patch_width]
            
            if torch.mean(LDCT_patch) > self.mean_value and torch.mean(SDCT_patch) > self.mean_value:
                break
            
        return LDCT_patch, SDCT_patch
        

class ImgPairsDataset(Dataset):
    def __init__(self, img_pairs, crop_size, patch_size):
        self.img_pairs = img_pairs
        self.crop_size = crop_size
        self.patch_size = patch_size
        
        self.transforms = Compose([
            Grayscale(),
            CenterCrop(self.crop_size),
            ToTensor()
        ])
        
        self.double_transform = DoubleRandomPatch(self.crop_size, self.patch_size)
            
            
    def __getitem__(self, index):
        LDCT_img = self.transforms(Image.open(self.img_pairs[index][0]))
        SDCT_img = self.transforms(Image.open(self.img_pairs[index][1]))
        descr = self.img_pairs[index][0].split('/')[-2]
        img_name = self.img_pairs[index][0].split('/')[-1]
        
        if self.crop_size != self.patch_size:
            LDCT_img, SDCT_img = self.double_transform(LDCT_img, SDCT_img)
        
        return LDCT_img, SDCT_img, descr, img_name

    
    def __len__(self):
        return len(self.img_pairs)