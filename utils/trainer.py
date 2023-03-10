import torch
from torch import nn
import timm

import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import logging
import time

from .basic import *

class Trainer:
    def __init__(
        self, 
        resnet_mode: str = 'resnet18',    # which model you would like to use
        dataset: str = 'CIFAR10',       # which dataset you would like to use
        batch_size: int = 128,          # batch size during training
        lr: float = 1e-4,               # learning rate for optimizer
        seed: int = 0,                  # the random seed
        cuda: int = 0,                  # cuda index
        use_lr_sche: bool = True,
        use_wandb: bool = False,        # whether to use weights & bias to monitor whole process
        **kwargs
    ) -> None:
        '''
        Args:
        - `model_name`: which model you would like to use. Default is AlexNet
        - `dataset`: which dataset you would like to use. Default is CIFAR10
        - `batch_size`: batch size during training
        - `lr`: learning rate for optimizer
        - `seed`: the random seed
        - `cuda`: cuda index
        - `use_wandb`: whether to use weights & bias to monitor whole process
        '''
        if resnet_mode not in timm.list_models('resnet*'):
            raise ValueError('This is not suitable model!')
        self.seed, self.resnet_mode, self.dataset = seed, resnet_mode, dataset
        set_random_seed(seed)
        self.model: nn.Module = timm.create_model(
            resnet_mode, num_classes = load_yaml(dataset), pretrained = True
        )
        self.model_name = f'{resnet_mode}-{dataset}-bs{batch_size}-lr{lr}-seed{seed}'
        self.makedirs()
        
        self.loss_fn = nn.CrossEntropyLoss(reduction = 'mean')
        self.opt = torch.optim.Adam(self.model.parameters(), lr = lr)
        if use_lr_sche:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt, 'max', factor = 0.5, patience = 2
            )
        
        self.train_iter, self.test_iter = generate_data_iter(dataset, batch_size, seed)
        self.device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')
        
        logging.basicConfig(
            filename = self.log_pth + self.model_name + '.log', 
            level = logging.INFO
        )
        if use_wandb:
            try:
                import wandb
                self.use_wandb = True
                wandb.init(
                    project = 'Interp-AA',
                    name = self.model_name,
                    config = {
                        'resnet_mode': resnet_mode,
                        'dataset': dataset,
                        'batch_size': batch_size,
                        'lr': lr,
                        'seed': seed,
                    }
                )
            except:
                print('import wandb fail!')
                self.use_wandb = False
        else:
            self.use_wandb = False
        
    def fit(self, epochs: int = 100):
        set_random_seed(self.seed)
        metrics = {
            'train_loss': [],
            'test_loss': [],
            'train_acc': [],
            'test_acc': []
        }
        max_acc, best_epoch = 0.0, 0
        
        logging.info(f'train on {self.device}')
        self.model.to(self.device)
        
        for epoch in range(epochs):
            start = time.time()
            train_loss, train_acc = self.train_epoch(epoch, epochs)
            test_loss, test_acc = self.test_epoch()
            if hasattr(self, 'scheduler'):
                self.scheduler.step(test_acc)
            final = time.time()
            
            if test_acc > max_acc:
                max_acc = test_acc
                best_epoch = epoch
                torch.save(
                    self.model.state_dict(), 
                    self.model_pth + self.model_name
                )
                
            for metric in list(metrics.keys()):
                metrics[metric].append(eval(metric))
            
            if self.use_wandb:
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'train_acc': train_acc,
                    'test_acc': test_acc
                })
            
            if epoch % 10 == 0:
                pd.DataFrame(metrics).to_csv(
                    self.metric_pth + self.model_name + '.csv'
                )
                
            train_info = f'train loss: {train_loss:.2e},  train acc: {train_acc * 100:4.2f}%'
            test_info = f'test loss: {test_loss:.2e},  test acc: {test_acc * 100:4.2f}%'
            other_info = f'time: {final-start:.2f},  best epoch: {best_epoch}'
            info = f'epoch: {epoch:3},  {train_info},  {test_info},  {other_info}'
            logging.info(info)
            
            print(info)
            
    def train_epoch(self, epoch, epochs):
        self.model.train()
        accu = Accumulator(3)
        with tqdm(enumerate(self.train_iter), total = len(self.train_iter), leave = True) as t:
            for idx, (X, Y) in t:
                self.opt.zero_grad()
                X, Y = X.to(self.device), Y.to(self.device)
                Y_hat: torch.Tensor = self.model(X)
                loss: torch.Tensor = self.loss_fn(Y_hat, Y)
                loss.backward()
                self.opt.step()
                with torch.no_grad():
                    correct_num = get_correct_num(Y, Y_hat)
                accu.add(loss.item() * len(Y), correct_num, len(Y))
                t.set_description(f'Epoch: [{epoch}/{epochs}]')
                t.set_postfix({
                    'batch': f'{idx} / {len(self.train_iter)}',
                    'training loss': f'{accu[0] / accu[-1]:.2e}',
                    'training acc': f'{(accu[1] / accu[-1]) * 100:4.2f}%'
                })
        return accu[0] / accu[-1], accu[1] / accu[-1]
        
        # for X, Y in self.train_iter:
        #     self.opt.zero_grad()
        #     X, Y = X.to(self.device), Y.to(self.device)
        #     Y_hat: torch.Tensor = self.model(X)
        #     loss: torch.Tensor = self.loss_fn(Y_hat, Y)
        #     loss.backward()
        #     self.opt.step()
        #     with torch.no_grad():
        #         correct_num = get_correct_num(Y, Y_hat)
        #     accu.add(loss.item() * len(Y), correct_num, len(Y))
        # return accu[0] / accu[-1], accu[1] / accu[-1]
            
    def test_epoch(self):
        self.model.eval()
        accu = Accumulator(3)
        with torch.no_grad():
            for X, Y in self.test_iter:
                X, Y = X.to(self.device), Y.to(self.device)
                Y_hat: torch.Tensor = self.model(X)
                loss: torch.Tensor = self.loss_fn(Y_hat, Y)
                correct_num = get_correct_num(Y, Y_hat)
                accu.add(loss.item() * len(Y), correct_num, len(Y))
        return accu[0] / accu[-1], accu[1] / accu[-1]
    
    def load(self):
        load_pth = self.model_pth + self.model_name
        self.model.load_state_dict(torch.load(load_pth, map_location = 'cpu'))
        self.model.eval()    
    
    def makedirs(self):
        self.results_pth = f'./results/{self.dataset}/{self.resnet_mode}'
        self.model_pth = f'{self.results_pth}/models/'
        self.metric_pth = f'{self.results_pth}/metrics/'
        self.log_pth = f'{self.results_pth}/logs/'
        
        pths = [self.model_pth, self.metric_pth, self.log_pth]
        for pth in pths:
            if not os.path.exists(pth):
                os.makedirs(pth)
    
    