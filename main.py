import torch
from torch import nn
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100

import os
import argparse

from utils import *
from model import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Basic Settings')
    parser.add_argument('--model', default = 'VGG')
    parser.add_argument('--dataset', default = 'CIFAR10')
    parser.add_argument('--seed', default = 0, type = int)
    parser.add_argument('--use_wandb', action = 'store_true')
    args = parser.parse_args()
    
    set_random_seed(args.seed)
    train_iter, test_iter = generate_data_iter(args.dataset, batch_size = 128)
    X, Y = next(iter(train_iter))
    
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    model = eval(args.model)(3, 10, conv_arch)
    print(model(X).shape)
    