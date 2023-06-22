import torch
import torchvision
import random
import numpy as np
import os,argparse,glob,scipy
from tqdm import tqdm
from torch.utils.data import DataLoader,Subset

from utils.dataset import get_dataset, process_dataset, noxi_dataset
from utils.dataset import get_dataset_TFN, noxi_dataset_TFN
from utils.model import get_model
from utils.engine import train_one_epoch, evaluate_one_epoch
from utils.logger import Logger

######################
### ArgumentParser ###
######################
def parse_arguments():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--dataset_path", type=str, default="/Users/damingw/ACM_MM/dataset/", help="directory of dataset")
    parser.add_argument("--preprocess", type=str, default="dim_reduction", help="method of preprocess")
    # model
    parser.add_argument("--method", type=str, default="TFN", help="main model selected")
    parser.add_argument("--save_path", type=str, default="checkpoints/", help="path to save model ckp")
    parser.add_argument("--save_freq", type=int, default=5, help="save model per number of epoch")
    parser.add_argument("--logger_name", type=str, default="test", help="logger name")
    # baseline model parameters
    parser.add_argument("--baseline_dim_list", type=list, default=[83,64,1], 
                        help="[input_shapes,hidden_units,output_units]")
    # TFN model parameters
    parser.add_argument("--TFN_hidden_dims", type=list, default=[31,102,37], # 314,1023,367
                        help="[audio_dim,video_dim,kinect_dim]")
    parser.add_argument("--TFN_dropouts", type=list, default=[0.01,0.01,0.01,0.01], 
                        help="[audio_dropout,video_dropout,kinect_dropout,post_fusion_dropout]")  
    parser.add_argument("--TFN_post_fusion_dim", type=int, default=16, 
                        help="specifying the size of the sub-networks after tensorfusion")    
    # hyperparameters
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--num_epoch", type=int, default=30, help="number of epochs")  
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")      
    parser.add_argument("--min_lr", type=float, default=1e-6, help="min learning rate for scheduler") 
    parser.add_argument("--do_scheduler", type=bool, default=True, help="do scheduler or not") 
    parser.add_argument("--weight_decay", type=float, default=0.001, help="weight_decay")
    parser.add_argument("--eps", type=float, default=1e-8, help="eps")
    return parser.parse_args()

####################
### Main Workder ###
####################
if __name__ == '__main__':
    args = parse_arguments()
    print(args,'\n')

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

    if args.method == 'baseline':
        # get dataset
        train_data,train_labels = get_dataset(train_dir, modalities, modalities_dim)
        val_data,val_labels = get_dataset(val_dir, modalities, modalities_dim)

        # do precess: normalization, pca
        X_train, x_val_pca =  process_dataset(train_data,val_data,83)

        # dataloader
        train_dataset = noxi_dataset(X_train, train_labels)
        val_dataset = noxi_dataset(x_val_pca, val_labels)

    elif args.method == 'TFN':
        # get dataset
        t_data_audio,t_data_video,t_data_kinect, t_labels = get_dataset_TFN(train_dir, modalities, modalities_dim)
        v_data_audio,v_data_video,v_data_kinect, v_labels = get_dataset_TFN(val_dir, modalities, modalities_dim)

        # do precess: normalization, pca
        t_audio_pca, v_audio_pca =  process_dataset(t_data_audio, v_data_audio, dim=args.TFN_hidden_dims[0])
        t_video_pca, v_video_pca =  process_dataset(t_data_video, v_data_video, dim=args.TFN_hidden_dims[1])
        t_kinect_pca, v_kinect_pca =  process_dataset(t_data_kinect, v_data_kinect, dim=args.TFN_hidden_dims[2])
        print('modality shapes: ',np.asarray(t_audio_pca).shape,np.asarray(t_video_pca).shape,np.asarray(t_kinect_pca).shape,'\n')
        # dataloader
        train_dataset = noxi_dataset_TFN(t_audio_pca, t_video_pca, t_kinect_pca, t_labels)
        val_dataset = noxi_dataset_TFN(v_audio_pca, v_video_pca, v_kinect_pca, v_labels)
        # shuffle
        train_dataset = torch.utils.data.Subset(train_dataset, torch.randperm(len(train_dataset)))
        val_dataset = torch.utils.data.Subset(train_dataset, torch.randperm(len(val_dataset)))
    else:
        NotImplementedError

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # Get the next batch of data and labels
    inputs, labels = next(iter(train_dataloader))
    print(labels)
    
    # model 
    model = get_model(args)
    print(f'loaded model with the method->{args.method}')

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
    
    # set lr scheduler
    lr_scheduler_cosann = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(len(train_dataloader) * args.num_epoch), eta_min=args.min_lr)

    # criterion
    criterion=torch.nn.MSELoss()

    # setup logger
    logger = Logger('logs/'+args.logger_name+' '+args.method+'.log',True)
    logger.append(args)
    print('finish setting logger')

    # training
    print('\n Start Training \n')

    for epoch in range(args.num_epoch):
        
        print(f'Current EPOCH : {epoch+1}')
        logger.append(f'epoch : {epoch+1}')

        train_one_epoch(args, train_dataloader, model, optimizer, criterion, lr_scheduler_cosann, logger)
        evaluate_one_epoch(args, val_dataloader, model, criterion, logger)

        # save ckp
        if (epoch+1)%args.save_freq==0:
            torch.save(model.state_dict(),os.path.join(args.save_path, f'ckp_{args.method}_epoch{epoch+1}.pt'))
            print(f"save dict for epoch : {epoch+1} ")