import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .metrics import concordance_correlation_coefficient

def train_one_epoch(args, dataloader, model, optimizer, criterion):
    # set loss and acc lists
    total_loss,total_acc = [],[]
    # train mode
    model.train()
    # loop batch
    with tqdm(total=len(dataloader)) as pbar:
        for i, (inputs, targets) in enumerate(dataloader):
            # clear optimizer
            optimizer.zero_grad()
            # model computation
            if args.method == 'baseline': 
                outputs = model(inputs)
            elif args.method == 'TFN':
                outputs = model(inputs[0],inputs[1],inputs[2])
            else:
                NotImplementedError
            # loss and acc
            loss = criterion(outputs, targets)
            acc = concordance_correlation_coefficient(outputs.detach().numpy(), targets.detach().numpy())
            # backward and step over
            loss.backward()
            optimizer.step()
            # sum losses in an epoch
            total_loss.append(loss.detach().cpu())
            total_acc.append(acc)
            pbar.set_description('training')
            pbar.set_postfix({'loss(iter)':float(loss.detach().cpu()), 
                              'loss(mean)':np.mean(total_loss),
                              'acc(iter)':acc,
                              'acc(mean)':np.mean(total_acc)})
            pbar.update(1)

def evaluate_one_epoch(args, dataloader, model, criterion):
    # set loss and acc lists
    total_loss,total_acc = [],[]
    # train mode
    model.eval()
    # loop batch
    with tqdm(total=len(dataloader)) as pbar:
        for i, (inputs, targets) in enumerate(dataloader):
            # model computation
            with torch.no_grad():
                # model computation
                if args.method == 'baseline': 
                    outputs = model(inputs)
                elif args.method == 'TFN':
                    outputs = model(inputs[0],inputs[1],inputs[2])
                else:
                    NotImplementedError
            # loss and acc
            loss = criterion(outputs, targets)
            acc = concordance_correlation_coefficient(outputs.detach().numpy(), targets.detach().numpy())
            # sum losses in a epoch
            total_loss.append(loss.detach().cpu())
            total_acc.append(acc)
            pbar.set_description('evaluation')
            pbar.set_postfix({'loss(iter)':float(loss.detach().cpu()), 
                              'loss(mean)':np.mean(total_loss),
                              'acc(iter)':acc,
                              'acc(mean)':np.mean(total_acc)})
            pbar.update(1)

        
