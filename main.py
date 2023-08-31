import torch
import torchvision
import random
import numpy as np
import os,argparse,glob,scipy
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader,Subset, SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler

from utils.dataset import get_dataset, process_dataset, noxi_dataset
from utils.dataset import get_dataset_TCA_Net, noxi_dataset_TCA_Net
from utils.model import get_model
from utils.engine import train_one_epoch, evaluate_one_epoch
from utils.logger import Logger

######################
### ArgumentParser ###
######################
def parse_arguments():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--dataset_path", type=str, default="/scratch/jr19/hh4436/engagement/dataset/", help="directory of dataset")
    parser.add_argument("--preprocess", type=str, default="pca", help="method of preprocess")
    parser.add_argument("--dataset_ratio", type=float, default=1, help="the ratio of the dataset")
    
    # model
    parser.add_argument("--method", type=str, default="TCA_Net", help="main model selected")
    parser.add_argument("--save_path", type=str, default="checkpoints/", help="path to save model ckp")
    parser.add_argument("--save_freq", type=int, default=10, help="save model per number of epoch")
    parser.add_argument("--logger_name", type=str, default="TCA_Net", help="logger name")
    
    # triplet co-attention/cross-attention
    parser.add_argument("--data_dim", type=list, default=[105,341,122], #orig; 314,1023,367 #baseline pca:105,341,122
                        help="[audio_dim,video_dim,kinect_dim]")
    parser.add_argument("--proj_dim", type=int, default=128, 
                        help="three modality projection dim")  
    
    # hyperparameters
    parser.add_argument("--num_workers", type=int, default=12, help="number of workers")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--num_epoch", type=int, default=20, help="number of epochs")  
    parser.add_argument("--lr", type=float, default=3e-6, help="learning rate")      
    parser.add_argument("--min_lr", type=float, default=1e-7, help="min learning rate for scheduler") 
    parser.add_argument("--do_scheduler", type=bool, default=True, help="do scheduler or not") 
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight_decay")
    parser.add_argument("--eps", type=float, default=1e-8, help="eps")

    # test/train
    parser.add_argument("--test", type=bool, default=False, help="do test mode or train mode")   
    return parser.parse_args()


####################
### Main Workder ###
####################
if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore")
    
    args = parse_arguments()
    print(args,'\n')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    t_data_audio,t_data_video,t_data_kinect, t_labels = get_dataset_TCA_Net(train_dir, modalities, modalities_dim)
    v_data_audio,v_data_video,v_data_kinect, v_labels = get_dataset_TCA_Net(val_dir, modalities, modalities_dim)

    # do precess: normalization, pca
    if args.preprocess == 'pca':
        t_audio_pca, v_audio_pca =  process_dataset(t_data_audio, v_data_audio, dim=args.data_dim[0])
        t_video_pca, v_video_pca =  process_dataset(t_data_video, v_data_video, dim=args.data_dim[1])
        t_kinect_pca, v_kinect_pca =  process_dataset(t_data_kinect, v_data_kinect, dim=args.data_dim[2])

        print('modality shapes: ',np.asarray(t_audio_pca).shape,np.asarray(t_video_pca).shape,np.asarray(t_kinect_pca).shape,'\n')

        # dataloader
        train_dataset = noxi_dataset_TCA_Net(t_audio_pca, t_video_pca, t_kinect_pca, t_labels)
        val_dataset = noxi_dataset_TCA_Net(v_audio_pca, v_video_pca, v_kinect_pca, v_labels)
    
    # if use projectors to norm dims, only apply minmaxscaler for data preprocessing
    else:
        scaler = MinMaxScaler()

        scaler.fit(t_data_audio + v_data_audio)
        t_data_audio = np.nan_to_num(scaler.transform(t_data_audio))
        v_data_audio = np.nan_to_num(scaler.transform(v_data_audio))

        scaler.fit(t_data_video + v_data_video)
        t_data_video = np.nan_to_num(scaler.transform(t_data_video))
        v_data_video = np.nan_to_num(scaler.transform(v_data_video))

        scaler.fit(t_data_kinect + v_data_kinect)
        t_data_kinect = np.nan_to_num(scaler.transform(t_data_kinect))
        v_data_kinect = np.nan_to_num(scaler.transform(v_data_kinect))

        train_dataset = noxi_dataset_TCA_Net(t_data_audio, t_data_video, t_data_kinect, t_labels)
        val_dataset = noxi_dataset_TCA_Net(v_data_audio,v_data_video,v_data_kinect, v_labels)           
 

    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,  sampler=SubsetRandomSampler(list(range(len(train_dataset)))),num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(list(range(len(val_dataset)))), num_workers=args.num_workers)
    #test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # Get the next batch of data and labels
    #inputs, labels = next(iter(train_dataloader))
    #print(labels)
    
    # model 
    model = get_model(args)
    # model.load_state_dict(torch.load('checkpoints/ckp_test1.011_20_09_epoch5.pt'))
    model = model.to(device)
    print(f'loaded model with the method->{args.method}')

    # get optimizer
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)
    named_parameters = list(model.named_parameters())

    optimizer = torch.optim.AdamW(params = model.parameters(),
                                  lr=args.lr, 
                                  weight_decay=args.weight_decay,
                                  eps=args.eps)
    
    # set lr scheduler
    lr_scheduler_cosann = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(len(train_dataloader) * args.num_epoch), eta_min=args.min_lr)

    # criterion
    criterion=torch.nn.MSELoss()

    if args.test:
        print(f'Start testing')
        #evaluate_one_epoch(args, test_dataloader, model, criterion, None, device)

    # setup logger
    now = datetime.now()
    timestamp = now.strftime("%d_%H_%M")
    logger = Logger('logs_TCA_Net_25/'+args.method+'_'+str(args.preprocess)+'_'+timestamp+'.log',True)
    logger.append(args)
    print('finish setting logger')

    # training
    print('\n Start Training \n')

    for epoch in range(args.num_epoch):
        
        print(f'Current EPOCH : {epoch+1}')
        logger.append(f'epoch : {epoch+1}')

        # evaluate_one_epoch(args, val_dataloader, model, criterion, logger, device)
        train_one_epoch(args, train_dataloader, model, optimizer, criterion, lr_scheduler_cosann, logger, device)
        evaluate_one_epoch(args, val_dataloader, model, criterion, logger, device)

        # save ckp
        if (epoch+1)%args.save_freq==0:
            torch.save(model.state_dict(),os.path.join(args.save_path, f'ckp_{args.method+str(args.dataset_ratio)+timestamp}_epoch{epoch+1}.pt'))
            print(f"save dict for epoch : {epoch+1} ")
    
