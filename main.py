import torch
import torchvision
import pickle
import random
import numpy as np
import os,argparse,glob,scipy
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from utils.dataset import get_dataset, process_dataset, noxi_dataset
from utils.model import ModalityFusionModel
from utils.engine import train_one_epoch, evaluate_one_epoch

######################
### ArgumentParser ###
######################
def parse_arguments():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--dataset_path", type=str, default="/Users/damingw/ACM_MM/dataset/", help="directory of dataset")
    parser.add_argument("--preprocess", type=str, default="dim_reduction", help="method of preprocess")
    # model
    parser.add_argument("--type", type=str, default="concat", help="main model selected")
    parser.add_argument("--input_shape", type=int, default=83, help="fusion module input shape")
    parser.add_argument("--save_path", type=str, default="checkpoints/", help="path to save model ckp")
    parser.add_argument("--save_freq", type=int, default=3, help="save model per number of epoch")
    parser.add_argument("--logger_name", type=str, default="test", help="logger name")
    # hyperparameters
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers")
    parser.add_argument("--batch_size", type=int, default="64", help="batch size")
    parser.add_argument("--num_epoch", type=int, default="30", help="number of epochs")  
    parser.add_argument("--lr", type=float, default=5e-7, help="learning rate")      
    parser.add_argument("--weight_decay", type=float, default=0.001, help="weight_decay")
    parser.add_argument("--eps", type=float, default=1e-8, help="eps")
    return parser.parse_args()

#####################
### Mainw Workder ###
#####################
if __name__ == '__main__':
    args = parse_arguments()

    # pre define modality and dim
    modalities = [
        ".audio.gemaps.stream~", 
        ".audio.soundnet.stream~", 
        ".video.openface2.stream~",
        ".kinect2.skeleton.stream~",
        ".video.openpose.stream~",
        ".kinect2.au.stream~"
        ]
    modalities_dim = [
        58, 
        256, 
        673,
        350, 
        350,
        17
        ]

    # setup subset path
    train_dir = os.path.join(args.dataset_path, "train")
    val_dir = os.path.join(args.dataset_path, "val")
    test_dir = os.path.join(args.dataset_path, "test")

    # get dataset
    train_data,train_labels = get_dataset(train_dir, modalities, modalities_dim)
    val_data,val_labels = get_dataset(val_dir, modalities, modalities_dim)
    # inspect
    print(np.asarray(train_data).shape)
    print(np.asarray(train_labels).shape)
    print(np.asarray(val_data).shape)
    print(np.asarray(val_labels).shape)

    # do precess: normalization, pca
    X_train, y_train, x_val_pca, val_labels =  process_dataset(train_data, train_labels, val_data, val_labels)
    # inspect
    print(X_train.shape)
    print(np.asarray(y_train).shape)
    print(x_val_pca.shape)
    print(np.asarray(val_labels).shape)
    
    # dataloader
    train_dataset = noxi_dataset(X_train, y_train)
    val_dataset = noxi_dataset(x_val_pca, val_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # Get the next batch of data and labels
    inputs, labels = next(iter(train_dataloader))
    print(labels)
    
    # model
    input_shapes = 83  
    hidden_units = 64
    output_units = 1
    model = ModalityFusionModel(input_shapes, hidden_units, output_units)

    # get optimizer
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)
    named_parameters = list(model.named_parameters())
    # filter out the frozen parameters
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": gain_or_bias_params, "weight_decay": args.weight_decay},
        {"params": rest_params, "weight_decay": args.weight_decay}],
        lr=args.lr, eps=args.eps)
    # criterion
    criterion=torch.nn.MSELoss()

    # training
    print('\n Start Training \n')
    for epoch in range(args.num_epoch):
        train_one_epoch(train_dataloader, model, optimizer, criterion)
        evaluate_one_epoch(val_dataloader, model, criterion)

        # save ckp
        if (epoch+1)%args.save_freq==0:
            torch.save(model.state_dict(),os.path.join(args.save_path, f'ckp_{args.logger_name}_epoch{epoch+1}.pt'))
            print(f"save dict for epoch : {epoch+1} ")