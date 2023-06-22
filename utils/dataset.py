import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset, DataLoader

def get_dataset_TFN(subset_dir, modalities, modalities_dim):

    anno = {}
    data_audio,data_video,data_kinect, labels = [],[],[],[]
    for path, sub_dirs, files in os.walk(subset_dir):
        for f in files:
            session_id = path.split("\\")[-1]
            role = f.split(".")[0]
            key = role + ";" + session_id
            if ".engagement.annotation.csv" in f:
                anno[key] = os.path.join(os.path.join(path), f)

    for entry in anno:
        features = {}
        lengths = []
        mod_name = []

        for idx, modality in enumerate(modalities):
            print(f'read from {str("/".join(anno[entry].split("/")[0:-1:1]) + "/" + entry.split(";")[0] + modality)}')
            f = open("/".join(anno[entry].split("/")[0:-1:1]) + "/" + entry.split(";")[0] + modality)
            a = np.fromfile(f, dtype=np.float32)
            a = a.reshape((int(a.shape[0] / modalities_dim[idx]), modalities_dim[idx]))
            features[modality] = a
            lengths.append(a.shape[0])
            mod_name.append((modality, modalities_dim[idx]))

        anno_file = np.genfromtxt(anno[entry], delimiter="\n", dtype=str)
        lengths.append(len(anno_file))
        num_samples = min(lengths)

        for i in range(num_samples):

            sample_audio, sample_video,sample_kinect= None,None,None

            for idx, modality in enumerate(modalities):
                if 'audio' in modality:
                    if sample_audio is None: sample_audio = np.nan_to_num(features[modality][i])
                    else: sample_audio = np.concatenate([sample_audio, features[modality][i]])
 
                if 'video' in modality:
                    if sample_video is None: sample_video = np.nan_to_num(features[modality][i])
                    else: sample_video = np.concatenate([sample_video, features[modality][i]])
                    
                if 'kinect' in modality:
                    if sample_kinect is None: sample_kinect = np.nan_to_num(features[modality][i])
                    else: sample_kinect = np.concatenate([sample_kinect, features[modality][i]])
                    
            data_audio.append(sample_audio)
            data_video.append(sample_video)
            data_kinect.append(sample_kinect)
            labels.append(float(anno_file[i]))
    
    print('finish loading dataset \n')
    return data_audio,data_video,data_kinect, labels

def get_dataset(subset_dir, modalities, modalities_dim):

    anno = {}
    data, labels = [],[]
    for path, sub_dirs, files in os.walk(subset_dir):
        for f in files:
            session_id = path.split("\\")[-1]
            role = f.split(".")[0]
            key = role + ";" + session_id
            if ".engagement.annotation.csv" in f:
                anno[key] = os.path.join(os.path.join(path), f)

    for entry in anno:
        features = {}
        lengths = []
        mod_name = []

        for idx, modality in enumerate(modalities):
            print(f'read from {str("/".join(anno[entry].split("/")[0:-1:1]) + "/" + entry.split(";")[0] + modality)}')
            f = open("/".join(anno[entry].split("/")[0:-1:1]) + "/" + entry.split(";")[0] + modality)
            a = np.fromfile(f, dtype=np.float32)
            a = a.reshape((int(a.shape[0] / modalities_dim[idx]), modalities_dim[idx]))
            features[modality] = a
            lengths.append(a.shape[0])
            mod_name.append((modality, modalities_dim[idx]))

        anno_file = np.genfromtxt(anno[entry], delimiter="\n", dtype=str)
        lengths.append(len(anno_file))
        num_samples = min(lengths)

        for i in range(num_samples):
            sample = None
            for idx, modality in enumerate(modalities):
                if sample is None:
                    sample = np.nan_to_num(features[modality][i])
                else:
                    sample = np.concatenate([sample, features[modality][i]])

            labels.append(float(anno_file[i]))
            data.append(sample)
    
    print('finish loading dataset \n')
    return data, labels


def process_dataset(train_data, val_data, dim):
    scaler = MinMaxScaler()
    scaler.fit(train_data + val_data)
    train_data_normalized = scaler.transform(train_data)
    val_data_normalized = scaler.transform(val_data)

    train_data_normalized = np.nan_to_num(train_data_normalized)
    val_data_normalized = np.nan_to_num(val_data_normalized)

    train_data = None
    val_data = None
    pca = PCA(n_components=dim, random_state=123)
    pca.fit(np.concatenate([train_data_normalized, val_data_normalized]))

    x_train_pca = pca.transform(train_data_normalized)
    x_val_pca = pca.transform(val_data_normalized)

    train_data_normalized = None
    val_data_normalized = None

    # X_train, y_train = shuffle(np.concatenate([x_train_pca, x_val_pca]), train_labels+val_labels, random_state=123)
    x_train_pca = np.nan_to_num(x_train_pca)
    x_val_pca = np.nan_to_num(x_val_pca)

    print(f'finish processing dataset (target dim:{dim})')
    
    return x_train_pca, x_val_pca

class noxi_dataset(Dataset):

    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    

class noxi_dataset_TFN(Dataset):

    def __init__(self, data_audio, data_video, data_kinect, labels):
        self.data_audio = torch.tensor(data_audio, dtype=torch.float32)
        self.data_video = torch.tensor(data_video, dtype=torch.float32)
        self.data_kinect = torch.tensor(data_kinect, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data_audio)

    def __getitem__(self, index):
        return [self.data_audio[index], self.data_video[index], self.data_kinect[index]], self.labels[index]
    
if __name__ == '__main__':

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
    train_dir = r"/Users/damingw/ACM_MM/dataset/train"
    val_dir = r"/Users/damingw/ACM_MM/dataset/val"
    test_dir = r"/Users/damingw/ACM_MM/dataset/test"

    # get dataset
    #train_data, train_labels = get_dataset(train_dir, modalities, modalities_dim)
    #val_data,val_labels = get_dataset(val_dir, modalities, modalities_dim)
    t_data_audio,t_data_video,t_data_kinect,t_labels = get_dataset_TFN(train_dir, modalities, modalities_dim)
    v_data_audio,v_data_video,v_data_kinect,v_labels = get_dataset_TFN(val_dir, modalities, modalities_dim)

    t_audio_pca, v_audio_pca =  process_dataset(t_data_audio, v_data_audio, dim=int(314/8))
    t_video_pca, v_video_pca =  process_dataset(t_data_video, v_data_video, dim=int(1023/10))
    t_kinect_pca, v_kinect_pca =  process_dataset(t_data_kinect, v_data_kinect, dim=int(367/8))

'''
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
'''