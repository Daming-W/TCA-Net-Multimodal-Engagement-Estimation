import numpy as np
import pandas as pd
import math
import random
import glob
import gc
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras import layers, models
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.utils import shuffle
from sklearn import neighbors
from sklearn.decomposition import PCA

import keras_tuner as kt
from keras_tuner import HyperModel


def concordance_correlation_coefficient(y_true, y_pred):
    """Concordance correlation coefficient."""
    # Remove NaNs
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })
    df = df.dropna()
    y_true = df['y_true']
    y_pred = df['y_pred']
    # Pearson product-moment correlation coefficients
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Mean
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Variance
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Standard deviation
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    return numerator / denominator


class MultimediateHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
            model = models.Sequential()
            hp_units = hp.Int('units', min_value=8, max_value=512, step=8)
            model.add(layers.Dense(units=hp_units, activation='relu', input_shape=(self.input_shape,)))
            hp_units_2 = hp.Int('units', min_value=8, max_value=512, step=8)
            model.add(layers.Dense(hp_units_2, activation='relu'))
            model.add(layers.Dropout(0.25))
            hp_units_3 = hp.Int('units', min_value=8, max_value=512, step=8)
            model.add(layers.Dense(hp_units_3, activation='relu'))
            model.add(layers.Dense(1, activation='relu'))

            # Tune the learning rate for the optimizer
            # Choose an optimal value from 0.01, 0.001, or 0.0001
            hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

            model.compile(optimizer='adam',
                        loss="mse",
                        metrics=[tf.keras.metrics.RootMeanSquaredError()])

            return model 

train_dir = r"..\MultiMediate\2023\public\train"

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

epochs = 10
batch_size = 64

train_anno = {}

for path, sub_dirs, files in os.walk(train_dir):
    for f in files:
        session_id = path.split("\\")[-1]
        role = f.split(".")[0]
        key = role + ";" + session_id
        if ".engagement.annotation.csv" in f:
            train_anno[key] = os.path.join(os.path.join(path), f)


train_data = []
train_labels = []

for entry in train_anno:
    features = {}
    lengths = []
    mod_name = []
    for idx, modality in enumerate(modalities):
        f = open("\\".join(train_anno[entry].split("\\")[0:-1:1]) + "\\" + entry.split(";")[0] + modality)
        a = np.fromfile(f, dtype=np.float32)
        a = a.reshape((int(a.shape[0] / modalities_dim[idx]), modalities_dim[idx]))
        features[modality] = a
        lengths.append(a.shape[0])
        mod_name.append((modality, modalities_dim[idx]))

    anno_file = np.genfromtxt(train_anno[entry], delimiter="\n", dtype=str)
    lengths.append(len(anno_file))
    num_samples = min(lengths)


    for i in range(num_samples):
        sample = None
        for idx, modality in enumerate(modalities):
            if sample is None:
                sample = np.nan_to_num(features[modality][i])
            else:
                sample = np.concatenate([sample, features[modality][i]])

        train_labels.append(float(anno_file[i]))
        train_data.append(sample)

print("train data loaded")

val_dir = r"..\MultiMediate\2023\public\val"

val_anno = {}

for path, sub_dirs, files in os.walk(val_dir):
    for f in files:
        session_id = path.split("\\")[-1]
        role = f.split(".")[0]
        key = role + ";" + session_id
        if ".engagement.annotation.csv" in f:
            val_anno[key] = os.path.join(os.path.join(path), f)

val_data = []
val_labels = []

for entry in val_anno:
    features = {}
    lengths = []
    for idx, modality in enumerate(modalities):
        f = open("\\".join(val_anno[entry].split("\\")[0:-1:1]) + "\\" + entry.split(";")[0] + modality)
        a = np.fromfile(f, dtype=np.float32)
        a = a.reshape((int(a.shape[0] / modalities_dim[idx]), modalities_dim[idx]))
        features[modality] = a
        lengths.append(a.shape[0])

    anno_file = np.genfromtxt(val_anno[entry], delimiter="\n", dtype=str)
    lengths.append(len(anno_file))
    num_samples = min(lengths)

    for i in range(num_samples):
        sample = None
        for idx, modality in enumerate(modalities):
            if sample is None:
                sample = np.nan_to_num(features[modality][i])
            else:
                sample = np.concatenate([sample, features[modality][i]])
        val_labels.append(float(anno_file[i]))
        val_data.append(sample)

print("val data loaded")

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

test_dir = r"..\MultiMediate\2023\private\test"
test_dir_data_path = r"..\MultiMediate\2023\public\test"

test_anno = {}

for path, sub_dirs, files in os.walk(test_dir):
    for f in files:
        session_id = path.split("\\")[-1]
        role = f.split(".")[0]
        key = role + ";" + session_id
        if ".engagement.annotation.csv" in f:
            test_anno[key] = os.path.join(os.path.join(path), f)

test_data = []
test_labels = []

for entry in test_anno:
    features = {}
    lengths = []
    for idx, modality in enumerate(modalities):
        f = open("\\".join(test_anno[entry].replace("private", "public").split("\\")[0:-1:1]) + "\\" + entry.split(";")[0] + modality)
        a = np.fromfile(f, dtype=np.float32)
        a = a.reshape((int(a.shape[0] / modalities_dim[idx]), modalities_dim[idx]))
        features[modality] = a
        lengths.append(a.shape[0])

    anno_file = np.genfromtxt(test_anno[entry], delimiter="\n", dtype=str)
    lengths.append(len(anno_file))
    num_samples = min(lengths)

    for i in range(num_samples):
        sample = None
        for idx, modality in enumerate(modalities):
            if sample is None:
                sample = np.nan_to_num(features[modality][i])
            else:
                sample = np.concatenate([sample, features[modality][i]])
        test_labels.append(float(anno_file[i]))
        test_data.append(sample)

print("test data loaded")
test_data_normalized = scaler.transform(test_data)

test_data_normalized = np.nan_to_num(test_data_normalized)

x_test_pca = pca.transform(test_data_normalized)

np.save("data/x_train_pca.npy", X_train)
np.save("data/train_labels.npy", np.asarray(y_train))

np.save("data/x_val_pca.npy", x_val_pca)
np.save("data/val_labels.npy", np.asarray(val_labels))

np.save("data/x_test_pca.npy", x_test_pca)
np.save("data/test_labels.npy", np.asarray(test_labels))

gc.collect()

