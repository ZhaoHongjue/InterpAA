import torch
from torch import nn
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100

import numpy as np
import matplotlib.pyplot as plt

import os

def set_random_seed(seed):    
    '''
    set random seed
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def generate_data_iter(dataset: str, batch_size: int = 128):
    '''
    generate data iter. The dataset can be `CIFAR10` or `CIFAR100`
    '''
    data_pth = f'./data/{dataset}'
    if not os.path.exists(data_pth):
        os.makedirs(data_pth)
        
    transform = transforms.Compose([
        transforms.ToTensor(), # numpy -> Tensor
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
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

def imshow(img: torch.tensor):
    '''
    show the pic
    '''
    img = img / 2 + 0.5 # reverse normalization
    np_img = img.numpy()  # tensor --> numpy
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()