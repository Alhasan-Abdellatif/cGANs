from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision
import xml.etree.ElementTree as ET
from tqdm import tqdm_notebook as tqdm
import os
import numpy as np
import glob
from PIL import Image,ImageOps,ImageEnhance
import matplotlib.pyplot as plt
import torch
import pandas as pd 
import ast 
from utils import *
from random import sample

class Channels(Dataset):
    def __init__(self, path = None,ext = 'txt',labels_path = None,sampling = None):       
        self.path = path
        self.ext =ext
        self.labels_path = labels_path
        
        if labels_path ==None:
            self.img_list = os.listdir(path)
        else:
            self.labels = pd.read_csv(labels_path)
            self.img_list = self.labels.name.to_list()
        if sampling:
            self.img_list = sample(self.img_list,sampling)
        if self.ext != 'txt' and self.ext != 'grdecl': # png,jpg
            self.transform1 = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
        else:
            self.transform1 = None
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        full_img_path = os.path.join(self.path,img_name)
        
        if self.ext == 'txt': 
            x = np.loadtxt(full_img_path) #normalized
            img = torch.from_numpy(x).unsqueeze(0).float()
        elif self.ext == 'png':
            img =Image.open(full_img_path)
        elif self.ext == 'grdecl':
            img =read_gcl(full_img_path) 
            img = normalize_by_replace(img)
            img = torch.from_numpy(img).unsqueeze(0).float()
            
        
        if self.labels_path == None:
            if self.transform1:
                img = self.transform1(img) 
            return {0: img}
        else:
            l = self.labels[self.labels['name']==img_name]['label'].values[0]
            if type(l) == str:
                l = ast.literal_eval(l)
                #l = from_np_array(l)                              
            if self.transform1:
                img = self.transform1(img) 
                
            l = torch.tensor(l).float()
            return {0: img, 1: l}

        
def from_np_array(array_string):
    if array_string[1] == '[':
        array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))

def read_gcl(filepath):
    file_pointer = open(f"{filepath}", "r")
    data_list = []
    for line in file_pointer:
        line = line.strip().split(' ')

        if line[0] == '--' or line[0] == '' or line[0].find('_') > 0:
            continue
        for data_str_ in line:
            if data_str_ == '':
                continue
            elif data_str_.find('*') == -1:
                try:
                    data_list.append(int(data_str_))
                except:
                    pass # automatically excludes '/'
            else:
                run = data_str_.split('*')
                inflation = [int(run[1])] * int(run[0])
                data_list.extend(inflation)

    file_pointer.close()
    data_np = np.array(data_list)
    # print(data_np.shape)
    data_np = data_np.reshape(100, 100)
    return data_np

d = {2:-1,3:0,5:1} # maps for non-stat


def normalize_by_replace(img):
    for key, value in d.items():
        img[img==np.array(key)] = value
    return img
