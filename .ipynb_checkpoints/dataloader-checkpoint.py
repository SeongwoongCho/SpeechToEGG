import os
import numpy as np
from utils import *
import librosa ## audio preprocessing
import random
import multiprocessing as mp

import torch
import torch.utils.data
from sklearn.model_selection import train_test_split
from SpecAug.sparse_image_warp_pytorch import sparse_image_warp,dense_image_warp

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
    
    for drt in ['../eggdata/TrainData/Alexis/','../eggdata/TrainData/vietnam/','../eggdata/TrainData/Childer/',
               '../eggdata/TrainData/CMU/','../eggdata/TrainData/saarbrucken/']:
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
    TrainPath_x = '../eggdata/TrainData/trainX_%d/'%n_frame
    TrainPath_y = '../eggdata/TrainData/trainy_%d/'%n_frame
    
    ValidPath_x = '../eggdata/TrainData/valX_%d/'%n_frame
    ValidPath_y = '../eggdata/TrainData/valy_%d/'%n_frame

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
        Train_x = Train_x[:1000]
        Train_y = Train_y[:1000]
        Val_x = Val_x[:1000]
        Val_y = Val_y[:1000]
    return Train_x,Train_y,Val_x,Val_y

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

def load_stft_datas_path(is_test = False):
    TrainPath = '../eggdata/TrainSTFT/train_data/'
    ValidPath = '../eggdata/TrainSTFT/valid_data/'

    Train = []
    Val = []
    
    for file in os.listdir(TrainPath):
        if 'npy' in file:
            Train.append(TrainPath + file)
    for file in os.listdir(ValidPath):
        if 'npy' in file:
            Val.append(ValidPath + file)
            
    if is_test:
        Train = Train[:1000] ## N,(2,F,T)
        Val = Val[:1000] ## N,(2,F,T)
    return Train,Val

def load_stft_noise(is_test=False):
    normal_dir = '../eggdata/TrainSTFT/noise/normal/'
    musical_dir = '../eggdata/TrainSTFT/noise/musical/'
    normal_noise_files = os.listdir(normal_dir)
    musical_noise_files = os.listdir(musical_dir)
    
    if is_test:
        normal_noise_files = normal_noise_files[:10]
        musical_noise_files = musical_noise_files[:10]
    
    normal_noise = [np.load(normal_dir+file) for file in normal_noise_files]
    musical_noise = [np.load(musical_dir+file) for file in musical_noise_files]
    
    return normal_noise, musical_noise

class STFTDataset(torch.utils.data.Dataset):
    def __init__(self,X,n_frame,is_train = True,normal_noise = None,musical_noise=None,aug=None):
        self.data = X
        self.n_frame = n_frame
        self.is_train = is_train
        self.aug= aug
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        if isinstance(self.data[idx],str):
            [X,Y] = np.load(self.data[idx])
        else:
            [X,Y] = self.data[idx]
#         X = loudness_normalize(X).astype('float32')
#         Y = loudness_normalize(Y).astype('float32')
        
        ### random crop
        T = X.shape[1]
        if T<self.n_frame:
            left = np.random.randint(self.n_frame-T)
            right = self.n_frame-T-left
            X = np.pad(X,((0,0),(left,right)),mode = 'constant',constant_values=(-1))
            Y = np.pad(Y,((0,0),(left,right)),mode = 'constant',constant_values=(-1))
        elif T>self.n_frame:
            pi = np.random.randint(T-self.n_frame)
            X = X[:,pi:pi+self.n_frame]
            Y = Y[:,pi:pi+self.n_frame]
#         print(X.shape)
        
        # Mixing before mel
        if self.aug:
            X = self.aug(X,self.normal_noise,self.musical_noise)
#         print(X.shape)
        X = stft_to_mel(X)[np.newaxis,:]
        Y = stft_to_mel(Y)[np.newaxis,:]
        
        # SpecAug after mel
        if self.is_train:
#             X = torch.Tensor(X)
#             Y = torch.Tensor(Y)
            
#             X,dense_flows = time_warp(X, W = 10)
#             Y = dense_image_warp(Y, dense_flows)
#             X,Y = X.squeeze(3),Y.squeeze(3)
            
#             X = X.cpu().detach().numpy()
#             Y = Y.cpu().detach().numpy()
            
            X = spec_masking(X, F = 10, T = 5, num_masks = 2, prob = 1, replace_with_zero = True)
            
        return X,Y