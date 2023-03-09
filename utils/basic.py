import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST

import numpy as np
import matplotlib.pyplot as plt

import os
import yaml


class Accumulator:
    '''used to accumulate related metrics'''
    def __init__(self, n: int):
        self.arr = [0] * n
        
    def add(self, *args):
        self.arr = [a + float(b) for a, b in zip(self.arr, args)]
    
    def __getitem__(self, idx: int):
        return self.arr[idx]


def set_random_seed(seed: int):    
    '''
    set random seed
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def load_yaml(dataset: str):
    with open('config.yaml', encoding = 'utf-8') as file:
        content = file.read()
    
    config = yaml.load(content, Loader = yaml.FullLoader)
    return  config[dataset]

def generate_data_iter(dataset: str, batch_size: int = 128, seed: int = 0):
    '''
    generate data iter. The dataset can be `CIFAR10` or `CIFAR100`
    '''
    set_random_seed(seed)
    data_pth = f'./data/{dataset}'
    if not os.path.exists(data_pth):
        os.makedirs(data_pth)
        
    transform = transforms.Compose([
        # transforms.AutoAugment(),
        transforms.ToTensor(), # numpy -> Tensor
        # transforms.Normalize(0.5, 0.5)
    ])
    
    train_set = eval(dataset)(
        root = data_pth,
        train = True,
        download = True,
        transform = transform
    )
    
    test_set = eval(dataset)(
        root = data_pth,
        train = False,
        download = True,
        transform = transform
    )
    
    train_iter = data.DataLoader(
        train_set, 
        batch_size = batch_size, 
        shuffle = True
    )
    test_iter = data.DataLoader(
        test_set, 
        batch_size = batch_size, 
        shuffle = False
    )
    return train_iter, test_iter

def get_correct_num(Y: torch.Tensor, Y_hat: torch.Tensor):
    with torch.no_grad():
        if Y_hat.dim() > 1 and Y_hat.shape[1] > 1:
            Y_hat = F.softmax(Y_hat, dim = 1)
            Y_hat = Y_hat.argmax(dim = 1)
        cmp = Y_hat.type(Y.dtype) == Y
        return float(cmp.type(Y.dtype).sum())

def imshow(img: torch.Tensor):
    '''
    show the pic
    '''
    img = img / 2 + 0.5 # reverse normalization
    np_img = img.numpy()  # tensor --> numpy
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()