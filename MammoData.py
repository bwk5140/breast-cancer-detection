# -*- coding: utf-8 -*-
"""

Adapted from https://www.kaggle.com/code/quachnam/breast-cancer-detection-roi-crop (Nam, 2023)
with slight modifications to....
"""
import torchvision
from torchvision import datasets, transforms, tv_tensors
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import v2

from torch.utils.data import Dataset, DataLoader

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import plotly.express as px
from glob import glob
#import pydicom
#from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_windowing
from tqdm import tqdm
import time
from datetime import datetime
from IPython import display
import os
import torch
import multiprocessing as mp
import warnings
from pathlib import Path
#import PIL
#from PIL import Image
from torchvision.io import read_image
from multiprocessing import Process, freeze_support
#from torchvision.models import resnet50, ResNet50_Weights
#import helpers
#from helpers import plot
#import json
#import pycocotools

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cores = mp.cpu_count()
plt.rcParams.update({'font.size': 10})
plt.rcParams['figure.figsize'] = (8, 6)

print('Cores:', cores)
print('Device:', device)
print('Day: ', datetime.now())

root = Path(os.getcwd())
csv_path = root/'INbreast Release 1.0/INbreast Release 1.0/Labels.csv'
image_dir = root/"INbreast Release 1.0\INbreast Release 1.0\AllPNGs\*.png"

transformTrain = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=False),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transformTest = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=False),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class InBreast(Dataset):
    def __init__(self, csv_path, img_dir, split, transform_fn=None):
        self.img_paths = []
        self.labels = []
        self.laterality = []
        self.view = []
        self.transform_fn = transform_fn
        
        df = get_df("INBREAST")[['Path', 'Cancer', 'View', 'Laterality']]
        
        
        length = int(len(df) * 0.7)
        
        #Split the data
        if(split == "train"):   
            self.df = df[0: length]         
        else:
            self.df = df[length:]
            
        length = len(self.df)
        
        print('\nLoading InBreast ' f'{split} dataset')
        self.imgs = [None for i in range(length)]
        
        for i in tqdm(range(length)):
            #patient_id = df.at[i, 'patient_id']
            #image_id = df.at[i, 'File name']
            #img_name = f'{patient_id}@{image_id}.png'
            
            if(split == "test"):
                index = int(len(df) * 0.7) + i
                img_path = self.df.at[index, 'Path']
                label = self.df.at[index, 'Cancer']
                self.laterality.append(self.df.at[index, 'Laterality'])
                self.view.append(self.df.at[index, 'View'])
            else:
                img_path = self.df.at[i, 'Path']
                label = self.df.at[i, 'Cancer']
                self.laterality.append(self.df.at[i, 'Laterality'])
                self.view.append(self.df.at[i, 'View'])
            
            self.img_paths.append(img_path)
                     
            img = read_image(img_path)
            img = self.transform_fn(img)
            self.imgs[i] = img
                
            self.labels.append(label)

        print(f'Done loading ' f'{split} dataset with {len(self.labels)} samples.')
        self.df.len = len(self.labels)


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        
        if (label == 1):
            img = self.imgs[idx]
        else:
            img = read_image(img_path)
        
        #Apply transforms to the Image
        if self.transform_fn:
            img = self.transform_fn(img)
            
        return img, label
        
    def get_sampler_weights(self, pos_neg_ratio):
        assert pos_neg_ratio > 0
        labels = np.array(self.labels)
        num_pos = labels.sum()
        num_neg = len(labels) - num_pos
        ori_pos_neg_ratio = num_pos / num_neg
        pos_weight = pos_neg_ratio / ori_pos_neg_ratio
        print('\nOriginal pos/neg ratio:', ori_pos_neg_ratio)
        print('Expect pos/neg ratio:', pos_neg_ratio)
        print('Pos weight:', pos_weight, '\n')
        weights = np.ones_like(labels, dtype = np.float32)
        weights[labels==1] = pos_weight
        
        return weights

    def get_labels(self):
        return np.array(self.labels)
    
    def get_view(self, idx):
        view = self.view[idx]
        return view
    
    def get_laterality(self, idx):
        laterality = self.laterality[idx]
        return laterality

    def get_df(self):
        return self.df

def get_df(name_dataset):
    
    if name_dataset == "RSNA":
        df = pd.read_csv(f"input/rsna-breast-cancer-detection/train.csv")
        df.columns = df.columns.str.capitalize()
        df['Path'] = df[['Patient_id', 'Image_id']].apply(lambda x: '/kaggle/input/rsna-breast-cancer-detection/train_images/'+\
                                    str(x['Patient_id']) + "/" + str(x['Image_id']) + ".dcm", axis=1)
        for i in tqdm(range(len(df)), desc="Loading RSNA dataset"):
            i + 1

    elif name_dataset == "DDSM":
        is_cancer = {'Benign': 1, 'Cancer': 1, 'Normal': 0}
        data = {'Filename': [], 'Age':[], 'Density': [], 'Cancer':[], 'View': [], 'Laterality': [], 'Path': []}
        # extract the path and view
        paths = glob(f"input/miniddsm2/MINI-DDSM-Complete-PNG-16/*/*")
        for head in tqdm(paths, desc="Loading MINI-DDSM dataset"):
            path = glob(f"{head}/*")
            path_ics = [x for x in path if "ics" in x][0]
            path_img = [x for x in path if ("png" in x and 'Mask' not in x)]
            if len(path_img) >= 1:
                # get information from file *.png
                for txt in path_img:
                    view = txt.split('.')[-2].split('_')[1]
                    laterality = txt.split('.')[-2].split('_')[0]
                    data['View'].append(view)
                    data['Laterality'].append('L' if laterality=='LEFT' else 'R')
                    data['Path'].append(txt)
                    data['Cancer'].append(is_cancer[head.split('/')[-2]])

                    # get information from file *.ics
                    f = open(path_ics, "r")
                    ics_text = f.read().strip().split("\n")
                    for txt in ics_text:
                        if txt.split()[0].upper() == 'FILENAME':
                            data['Filename'].append(txt.split()[1] if len(txt.split()) > 1 else 'NaN')
                        if txt.split()[0].upper() == 'PATIENT_AGE':
                            data['Age'].append(txt.split()[1] if len(txt.split()) > 1 else 'NaN')
                        if txt.split()[0].upper() == 'DENSITY':
                            data['Density'].append(txt.split()[1] if len(txt.split()) > 1 else 'NaN')
        df = pd.DataFrame(data)

    elif name_dataset == "MIAS":
        df = pd.read_csv(f'input/mias-mammography/Info.txt', sep=" ").drop('Unnamed: 7',axis=1)
        df.columns = df.columns.str.capitalize()
        df['Path'] = df['Refnum'].apply(lambda x: '/kaggle/input/mias-mammography' + "/" + "all-mias" + "/" + x + ".pgm")
        df['Cancer'] = df['Class'].apply(lambda x: 0 if x.upper() == 'NORM' else 1)
        for i in tqdm(range(len(df)), desc="Loading MIAS dataset"):
            i+1
    elif name_dataset == "INBREAST":
        df = pd.read_csv(f'INbreast Release 1.0/INbreast Release 1.0/Labels.csv',engine = "python",
                         skipfooter=2)
        df.columns = df.columns.str.capitalize()
        paths = glob(f"INbreast Release 1.0\INbreast Release 1.0\AllPNGs\*.png")
        df['Path'] = df['File name'].apply(lambda x: find_matching_path(x, paths))
        df['Lesion annotation status'].fillna('cancer', inplace=True)
        df['Lesion annotation status'] = df['Lesion annotation status'].str.upper()
        df['Cancer'] = df['Lesion annotation status'].apply(lambda x: 0 if x == 'NO ANNOTATION (NORMAL)' else 1)
        
        #df['Key'] = "{}:{}".format(df['File name'], df['Cancer'])
        
        '''
        for i in tqdm(range(len(df)), desc="Loading INBREAST dataset"):
            i+1
            '''
    else:
        print("Dataset not found")
    return df

data_breast = ["RSNA", "DDSM", "MIAS", 'INBREAST']
#rsna = get_df(data_breast[0])[['Path', 'Cancer', 'View', 'Laterality']]
#ddsm = get_df(data_breast[1])[['Path', 'Cancer', 'View', 'Laterality']]
#mias = get_df(data_breast[2])[['Path', 'Cancer']]
#inbreast = get_df(data_breast[3])[['Path', 'Cancer', 'View', 'Laterality', 'Key']]

#print(inbreast)

def find_matching_path(x, paths):
    matching_paths = [path for path in paths if path.split('\\')[-1].split('_')[0] == str(x)]
    return matching_paths[0] if matching_paths else None


def get_data():
    
    train_data = InBreast(csv_path, image_dir, "train", transformTrain)
    #print(train_data)
    test_data = InBreast(csv_path, image_dir, "test", transformTest)
    #print(test_data)
    
    pos_neg_ratio = 1.0 / 8
    
    weights = train_data.get_sampler_weights(pos_neg_ratio)
    
    return train_data, test_data, weights

if __name__ == '__main__':
    freeze_support()
    #train, test, weights = get_data()
    
