import argparse
from tabulate import tabulate

import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Basic Settings')
    parser.add_argument('--model_name', default = 'AlexNet')
    parser.add_argument('--dataset', default = 'FashionMNIST')
    parser.add_argument('--bs', default = 128, type = int)
    parser.add_argument('--lr', default = 0.0001, type = float)
    parser.add_argument('--epochs', default = 100, type = int)
    parser.add_argument('--seed', default = 1, type = int)
    parser.add_argument('--use_gap', action = 'store_true')
    parser.add_argument('--use_wandb', action = 'store_true')
    args = parser.parse_args()
    print(tabulate(
        list(vars(args).items()), headers = ['attr', 'setting'], tablefmt ='orgtbl'
    ))
    
    trainer = utils.Trainer(**vars(args))
    trainer.fit(args.epochs)
    
    # data_iter, _ = utils.generate_data_iter(args.dataset)
    # X, Y = next(iter(data_iter))
    # print(X.shape, Y.shape)
    
    # import model
    # alex = model.AlexNet(1, 10)
    # verbose_alex = utils.VerboseExe(alex)
    # _ = verbose_alex(X)
    
    