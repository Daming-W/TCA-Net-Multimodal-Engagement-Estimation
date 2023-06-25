import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .metrics import concordance_correlation_coefficient

def train_one_epoch(args, dataloader, model, optimizer, criterion, lr_scheduler, logger, device):
    # set loss and acc lists
    total_loss,total_acc = [],[]
    # train mode
    model.train()
    # loop batch
    with tqdm(total=len(dataloader)) as pbar:
        for i, (inputs, targets) in enumerate(dataloader):

            inputs = [tensor.to(device) for tensor in inputs]
            targets = targets.to(device)
            
            # clear optimizer
            optimizer.zero_grad()
            # model computation
            if args.method == 'baseline': 
                outputs = model(inputs)
            elif 'TFN' in args.method:
                outputs = model(inputs[0],inputs[1],inputs[2])
            else:
                NotImplementedError
            # loss and acc
            loss = criterion(outputs, targets)
            acc = concordance_correlation_coefficient(outputs.detach().cpu().numpy(), targets.detach().cpu().numpy())
            # backward and step over
            loss.backward()
            optimizer.step()
            # do scheduler
            if args.do_scheduler==True:
                lr_scheduler.step()
            # sum losses in an epoch
            total_loss.append(loss.detach().cpu())
            total_acc.append(acc)
            pbar.set_description('training')
            pbar.set_postfix({'loss(iter)':float(loss.detach().cpu()), 
                              'loss(mean)':np.mean(total_loss),
                              'acc(iter)':acc,
                              'acc(mean)':np.mean(total_acc)})
            pbar.update(1)
    # update logger    
    epoch_loss = np.nanmean(total_loss)    
    epoch_acc = np.nanmean(total_acc)
    logger.append(f'train_epoch_loss: {epoch_loss}')
    logger.append(f'train_epoch_acc: {epoch_acc}')
    logger.append(f'train_lr : {lr_scheduler.get_last_lr()[0]}')

def evaluate_one_epoch(args, dataloader, model, criterion, logger, device):
    # set loss and acc lists
    total_loss,total_acc = [],[]
    # train mode
    model.eval()
    # loop batch
    with tqdm(total=len(dataloader)) as pbar:
        for i, (inputs, targets) in enumerate(dataloader):

            inputs = [tensor.to(device) for tensor in inputs]
            targets = targets.to(device)
            
            # model computation
            with torch.no_grad():
                # model computation
                if args.method == 'baseline': 
                    outputs = model(inputs)
                elif 'TFN' in args.method:
                    outputs = model(inputs[0],inputs[1],inputs[2])
                else:
                    NotImplementedError
            # loss and acc
            loss = criterion(outputs, targets)
            acc = concordance_correlation_coefficient(outputs.detach().cpu().numpy(), targets.detach().cpu().numpy())
            # sum losses in a epoch
            total_loss.append(loss.detach().cpu())
            total_acc.append(acc)
            pbar.set_description('evaluation')
            pbar.set_postfix({'loss(iter)':float(loss.detach().cpu()), 
                              'loss(mean)':np.mean(total_loss),
                              'acc(iter)':acc,
                              'acc(mean)':np.mean(total_acc)})
            pbar.update(1)
    # update logger  
    epoch_loss = np.nanmean(total_loss)
    epoch_acc = np.nanmean(total_acc)
    logger.append(f'evalutation_epoch_loss: {epoch_loss}')    
    logger.append(f'evalutation_epoch_acc: {epoch_acc}')   
