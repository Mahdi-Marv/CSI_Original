import os

import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from glob import glob
from PIL import Image
from glob import glob
import pickle
import random
import rasterio
import re
from torchvision.datasets.folder import default_loader
import pydicom

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torchvision
import subprocess
from tqdm import tqdm
import requests
import shutil
from PIL import Image
import shutil
import random
import zipfile
import time
import gdown

class Camelyon17(Dataset):
    def __init__(self, transform, is_train=True, test_id=1):
        self.is_train = is_train
        self.transform = transform
        if is_train:
            node0_train = glob('/kaggle/input/camelyon17-clean/node0/train/normal/*')
            node1_train = glob('/kaggle/input/camelyon17-clean/node1/train/normal/*')
            node2_train = glob('/kaggle/input/camelyon17-clean/node2/train/normal/*')

            self.image_paths = node0_train + node1_train + node2_train
        else:
            if test_id == 1:
                node0_test_normal = glob('/kaggle/input/camelyon17-clean/node0/test/normal/*')
                node0_test_anomaly = glob('/kaggle/input/camelyon17-clean/node0/test/anomaly/*')

                node1_test_normal = glob('/kaggle/input/camelyon17-clean/node1/test/normal/*')
                node1_test_anomaly = glob('/kaggle/input/camelyon17-clean/node1/test/anomaly/*')

                node2_test_normal = glob('/kaggle/input/camelyon17-clean/node2/test/normal/*')
                node2_test_anomaly = glob('/kaggle/input/camelyon17-clean/node2/test/anomaly/*')

                test_path_normal = node0_test_normal + node1_test_normal + node2_test_normal
                test_path_anomaly = node0_test_anomaly + node1_test_anomaly + node2_test_anomaly

                self.image_paths = test_path_normal + test_path_anomaly
                self.test_label = [0] * len(test_path_normal) + [1] * len(test_path_anomaly)
            else:
                node3_test_normal = glob('/kaggle/input/camelyon17-clean/node3/test/normal/*')
                node3_test_anomaly = glob('/kaggle/input/camelyon17-clean/node3/test/anomaly/*')

                node4_test_normal = glob('/kaggle/input/camelyon17-clean/node4/test/normal/*')
                node4_test_anomaly = glob('/kaggle/input/camelyon17-clean/node4/test/anomaly/*')

                shifted_test_path_normal = node3_test_normal + node4_test_normal
                shifted_test_path_anomaly = node3_test_anomaly + node4_test_anomaly

                self.image_paths = shifted_test_path_normal + shifted_test_path_anomaly
                self.test_label = [0] * len(shifted_test_path_normal) + [1] * len(shifted_test_path_anomaly)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.is_train:
            return img, 0
        else:
            return img, self.test_label[idx]
    def __len__(self):
        return len(self.image_paths)
class Brain(Dataset):
    def __init__(self, transform, is_train=True, test_id=1):
        print('brain dataset')
        self.is_train = is_train
        self.transform = transform
        if is_train:
            self.image_paths = glob('./Br35H/dataset/train/normal/*')
            self.test_label = [0] * len(self.image_paths)
        else:
            if test_id==1:
                test_normal_path = glob('./Br35H/dataset/test/normal/*')
                test_anomaly_path = glob('./Br35H/dataset/test/anomaly/*')

                self.image_paths = test_normal_path + test_anomaly_path
                self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)
            else:
                test_normal_path = glob('./brats/dataset/test/normal/*')
                test_anomaly_path = glob('./brats/dataset/test/anomaly/*')

                self.image_paths = test_normal_path + test_anomaly_path
                self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.is_train:
            return img, 0
        else :
            return img, self.test_label[idx]
class Chest(Dataset):
    def __init__(self, transform, is_train=True, test_id=1):
        print('brain dataset')
        self.is_train = is_train
        self.transform = transform
        self.test_id = test_id
        if is_train:
            self.image_paths = glob('/kaggle/working/train/normal/*')
            self.test_label = [0] * len(self.image_paths)
        else:
            test_normal_path = glob('/kaggle/working/test/normal/*')
            test_anomaly_path = glob('/kaggle/working/test/anomaly/*')

            self.image_paths = test_normal_path + test_anomaly_path
            self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)

            if self.test_id == 2:
                shifted_test_normal_path = glob('/kaggle/working/4. Operations Department/Test/1/*')
                shifted_test_anomaly_path = (glob('/kaggle/working/4. Operations Department/Test/0/*') + glob(
                    '/kaggle/working/4. Operations Department/Test/2/*') +
                                             glob('/kaggle/working/4. Operations Department/Test/3/*'))

                self.image_paths = shifted_test_normal_path + shifted_test_anomaly_path
                self.test_label = [0] * len(shifted_test_normal_path) + [1] * len(shifted_test_anomaly_path)

            if self.test_id == 3:
                test_normal_path = glob('/kaggle/working/chest_xray/test/NORMAL/*')
                test_anomaly_path = glob('/kaggle/working/chest_xray/test/PNEUMONIA/*')

                self.image_paths = test_normal_path + test_anomaly_path
                self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.test_id==1 or self.is_train:
            dicom = pydicom.dcmread(self.image_paths[idx])
            image = dicom.pixel_array

            # Convert to a PIL Image
            image = Image.fromarray(image).convert('RGB')

            # Apply the transform if it's provided
            if self.transform is not None:
                image = self.transform(image)
            return image, self.test_label[idx]



        img = Image.open(self.image_paths[idx])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.is_train:
            return img, 0
        else :
            return img, self.test_label[idx]

class Aptos(Dataset):
    def __init__(self, transform, is_train=True, test_id=1):
        print('aptos dataset')
        self.is_train = is_train
        self.transform = transform
        if is_train:
            self.image_paths = glob('/kaggle/working/APTOS/train/NORMAL/*')
            self.test_label = [0] * len(self.image_paths)
        else:
            if test_id==1:
                test_normal_path = glob('/kaggle/working/APTOS/test/NORMAL/*')
                test_anomaly_path = glob('/kaggle/working/APTOS/test/ABNORMAL/*')

                self.image_paths = test_normal_path + test_anomaly_path
                self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)
            else:
                df = pd.read_csv('/kaggle/input/ddrdataset/DR_grading.csv')
                label = df["diagnosis"].to_numpy()
                path = df["id_code"].to_numpy()

                normal_path = path[label == 0]
                anomaly_path = path[label != 0]

                shifted_test_path = list(normal_path) + list(anomaly_path)
                shifted_test_label = [0] * len(normal_path) + [1] * len(anomaly_path)

                shifted_test_path = ["/kaggle/input/ddrdataset/DR_grading/DR_grading/" + s for s in shifted_test_path]

                self.image_paths = shifted_test_path
                self.test_label = shifted_test_label



    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.is_train:
            return img, 0
        else :
            return img, self.test_label[idx]

class Isic(Dataset):
    def __init__(self, transform, is_train=True, test_id=1):
        print('brain dataset')
        self.is_train = is_train
        self.transform = transform
        if is_train:
            self.image_paths = glob('/kaggle/input/isic-task3-dataset/dataset/train/NORMAL/*')
            self.test_label = [0] * len(self.image_paths)
        else:
            if test_id==1:
                test_normal_path = glob('/kaggle/input/isic-task3-dataset/dataset/test/NORMAL/*')
                test_anomaly_path = glob('/kaggle/input/isic-task3-dataset/dataset/test/ABNORMAL/*')

                self.image_paths= test_normal_path + test_anomaly_path
                self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)
            else:
                df = pd.read_csv('/kaggle/input/pad-ufes-20/PAD-UFES-20/metadata.csv')

                shifted_test_label = df["diagnostic"].to_numpy()
                shifted_test_label = (shifted_test_label != "NEV")

                shifted_test_path = df["img_id"].to_numpy()
                shifted_test_path = '/kaggle/input/pad-ufes-20/PAD-UFES-20/Dataset/' + shifted_test_path

                self.image_paths = shifted_test_path
                self.test_label = shifted_test_label


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.is_train:
            return img, 0
        else :
            return img, self.test_label[idx]
