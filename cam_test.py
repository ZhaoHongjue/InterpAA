import torch
from torch import nn

import os
import argparse

import utils
from utils.cam import *

imagenette_dirs = (
    'n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079', 
    'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257'
)
imagenette_classes = (
    'tench', 'English springer', 'cassette player', 'chain saw', 'church', 
    'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute'
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Basic Settings')
    parser.add_argument('--dataset', default = 'imagenette')
    parser.add_argument('--cam', default = 'CAM')
    parser.add_argument('--seed', default = 0, type = int)
    args = parser.parse_args()
    
    trainer = utils.Trainer(
        resnet_mode = args.resnet_mode,
        dataset = args.dataset,
        batch_size = 128,
        lr = 0.0001,
        seed = args.seed,
        cuda = 0,
        use_wandb = False
    )
    trainer.load()
    
    cam: BaseCAM = eval(args.cam)(
        trainer.model,
        target_layer = 'layer4',
        fc_layer = 'fc',
    )