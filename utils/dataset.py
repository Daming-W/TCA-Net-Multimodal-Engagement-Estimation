import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset, DataLoader

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

def process_dataset(train_data, train_labels, val_data, val_labels):
    scaler = MinMaxScaler()
    scaler.fit(train_data + val_data)
    train_data_normalized = scaler.transform(train_data)
    val_data_normalized = scaler.transform(val_data)

    train_data_normalized = np.nan_to_num(train_data_normalized)
    val_data_normalized = np.nan_to_num(val_data_normalized)

    train_data = None
    val_data = None
    pca = PCA(n_components=83, random_state=123)
    pca.fit(np.concatenate([train_data_normalized, val_data_normalized]))

    x_train_pca = pca.transform(train_data_normalized)
    x_val_pca = pca.transform(val_data_normalized)

    train_data_normalized = None
    val_data_normalized = None

    X_train, y_train = shuffle(np.concatenate([x_train_pca, x_val_pca]), train_labels+val_labels, random_state=123)
    X_train = np.nan_to_num(X_train)
    y_train = np.nan_to_num(y_train)

    print('finish loading dataset \n')
    
    return X_train, y_train, x_val_pca, val_labels

class noxi_dataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    

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
    train_data,train_labels = get_dataset(train_dir, modalities, modalities_dim)
    val_data,val_labels = get_dataset(val_dir, modalities, modalities_dim)

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