import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def train_one_epoch(dataloader, model, optimizer, criterion):
    # set loss list
    total_loss = []
    # train mode
    model.train()
    # loop batch
    with tqdm(total=len(dataloader)) as pbar:
        for i, (inputs, targets) in enumerate(dataloader):
            # clear optimizer
            optimizer.zero_grad()
            # model computation
            outputs = model(inputs)
            print(outputs)
            print(targets)
            loss = criterion(outputs, targets)
            # backward and step over
            loss.backward()
            optimizer.step()
            # sum losses in an epoch
            total_loss.append(loss.detach().cpu())
            pbar.set_description('training')
            pbar.set_postfix({'loss(iter)':float(loss.detach().cpu()), 
                              'loss(mean)':np.mean(total_loss)})
            pbar.update(1)

def evaluate_one_epoch(dataloader, model, criterion):
    # set loss list
    total_loss = []
    # train mode
    model.eval()
    # loop batch
    with tqdm(total=len(dataloader)) as pbar:
        for i, (inputs, targets) in enumerate(dataloader):
            # model computation
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            # sum losses in a epoch
            total_loss.append(loss.detach().cpu())
            pbar.set_description('evaluation')
            pbar.set_postfix({'loss(iter)':float(loss.detach().cpu()), 'loss(mean)':np.mean(total_loss)})
            pbar.update(1)

        