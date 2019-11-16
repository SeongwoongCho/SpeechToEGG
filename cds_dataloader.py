import os
import numpy as np
from cds_utils import *
import librosa ## audio preprocessing
import random
import multiprocessing as mp

import torch
import torch.utils.data
from sklearn.model_selection import train_test_split

seed_everything(42)

def process(t):
    drt, file, n_frame = t
    X,Y = [],[]
    [x,y],sr = librosa.load(drt+file, sr=16000, mono=False)
    audio_length = len(x)
    padding = n_frame - audio_length%n_frame
    x = np.pad(x,pad_width=(0,padding),mode='constant')
    y = np.pad(y,pad_width=(0,padding),mode='constant')
    for frame in range(audio_length//n_frame+1):
        X.append(x[frame*n_frame:(frame+1)*n_frame])
        Y.append(y[frame*n_frame:(frame+1)*n_frame])
    return X,Y

def load_datas(n_frame,is_test = False):
    X,y = [],[]
    pool = mp.Pool(mp.cpu_count())
    
    print("load Train Datas")
    args = [] ## [..(drt,file,n_frame)..]
    
    for drt in ['./datasets/TrainData/Alexis/','./datasets/TrainData/vietnam/','./datasets/TrainData/Childer/',
               './datasets/TrainData/CMU/','./datasets/TrainData/saarbrucken/']:
        for file in os.listdir(drt):
            if 'wav' in file:
                args.append((drt,file,n_frame))
    
    if is_test:
        args = args[:50]
    
    tmp = pool.map(process,args) ## [..[X,Y]..] X = [...[20000][20000][20000]...]
    for _X,_Y in tmp: ## _X : [...[20000]...]
        X += _X
        y += _Y
    pool.close()
    pool.join()
    X = np.array(X)
    y = np.array(y)
    
    train_X,val_X,train_y,val_y = train_test_split(X,y,test_size=0.3,random_state=42)
    return train_X,train_y,val_X,val_y

def load_datas_path(n_frame,is_test = False):
    TrainPath_x = './datasets/TrainData/trainX_%d/'%n_frame
    TrainPath_y = './datasets/TrainData/trainy_%d/'%n_frame
    
    ValidPath_x = './datasets/TrainData/valX_%d/'%n_frame
    ValidPath_y = './datasets/TrainData/valy_%d/'%n_frame

    Train_x = []
    Train_y = []
    Val_x = []
    Val_y = []
    
    for file in os.listdir(TrainPath_x):
        if 'npy' in file:
            Train_x.append(TrainPath_x + file)
            Train_y.append(TrainPath_y + file)
    for file in os.listdir(ValidPath_x):
        if 'npy' in file:
            Val_x.append(ValidPath_x + file)
            Val_y.append(ValidPath_y + file)
    
    if is_test:
        Train_x = Train_x[:300]
        Train_y = Train_y[:300]
        Val_x = Val_x[:300]
        Val_y = Val_y[:300]
    return Train_x,Train_y,Val_x,Val_y

def load_noise(n_frame,is_test=False):
    normal_dir = './datasets/TrainData/normal_noise/'
    musical_dir = './datasets/TrainData/musical_noise/'
    normal_noise_files = os.listdir(normal_dir)
    musical_noise_files = os.listdir(musical_dir)
    
    if is_test:
        normal_noise_files = normal_noise_files[:10]
        musical_noise_files = musical_noise_files[:10]
    
    normal_noise = [librosa.load(normal_dir+file,sr=16000)[0] for file in normal_noise_files]
    musical_noise = [librosa.load(musical_dir+file,sr=16000)[0] for file in musical_noise_files]
    
    return normal_noise, musical_noise

class Dataset(torch.utils.data.Dataset):
    def __init__(self,X,y,n_fft,hop_length,n_frame,is_train = True,normal_noise = None,musical_noise=None,aug=None):
        self.X = X
        self.y = y
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_frame = n_frame
        self.is_train = is_train
        self.aug= aug
        assert len(self.X) == len(self.y)
        if is_train:
            self.normal_noise,self.musical_noise = load_noise(self.n_frame)
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        if isinstance(self.X[idx],str):
            X = np.load(self.X[idx])
            Y = np.load(self.y[idx])
        else:
            X = self.X[idx]
            Y = self.y[idx]
        X = loudness_normalize(X).astype('float32')
        Y = loudness_normalize(Y).astype('float32')
        if self.aug:
            X = self.aug(X,self.normal_noise,self.musical_noise)
#         X = stft_with_phase(X,self.n_fft,self.hop_length,'channel')
#         Y = stft_with_phase(Y,self.n_fft,self.hop_length,'channel')
        return X,Y