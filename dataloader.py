import os
import numpy as np
import torch
import torch.utils.data
import multiprocessing

from utils.utils import seed_everything
from utils.prep_utils import stft_to_mel, stft_process
from utils.aug_utils import spec_masking, add_whitenoise
from sklearn.model_selection import train_test_split

import time

from utils.prep_utils import dynamic_range_decompression
from scipy.special import expit

seed_everything(42)

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
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self,X,n_frame,stride,is_train = True,normal_noise = None,musical_noise=None,aug=None,pseudo_mode=False):
        self.data = np.load(X,mmap_mode='r') if isinstance(X,str) else X
        self.n_frame = n_frame
        self.stride = stride
        self.is_train = is_train
        self.normal_noise = normal_noise
        self.musical_noise = musical_noise
        self.aug= aug
        self.pseudo_mode=pseudo_mode
    def __len__(self):
        return (self.data.shape[-1] - self.n_frame - self.stride) // self.stride
    def __getitem__(self,idx):
        if self.stride <=self.n_frame:
            offset = np.random.randint(low=0,high=self.stride)
            [X,Y] = self.data[:,:,offset + idx*self.stride:offset + idx*self.stride + self.n_frame] ## 2,F,T
        else:
            random_idx = np.random.randint(low = idx*self.stride, high=(idx+1)*self.stride - self.n_frame - 1)
            [X,Y] = self.data[:,:,random_idx:random_idx+self.n_frame]
        # Mixing before mel
        if self.aug:
            X = self.aug(X,self.normal_noise,self.musical_noise)
        X = stft_process(X).astype(np.float32) ## 2,F,T mag,phase
        Y = stft_process(Y,mask=True).astype(np.float32) ## 3,F,T mag,phase,mask
        
        # SpecAug after mel
        if self.is_train:
            X = add_whitenoise(X)
        
        if self.pseudo_mode:
            return X,Y,True
        return X,Y


## dataset cat 하기     
    
##### 
"""
def load_stft_unlabel_datas_path(is_test = False):
    TrainPath = '../eggdata/TrainSTFT/unlabeled/'
#     TrainPath = '../eggdata/TrainSTFT/unlabeled/split/'
    Train = []
    dsets = ['DSD100', 'KSS', 'zeroth_korean', 'speech_ko'] ## except librispeech 
    
    for dataset in os.listdir(TrainPath):
        if dataset in dsets:
            for file in os.listdir(TrainPath + dataset):
                if 'npy' in file:
                    Train.append(os.path.join(TrainPath + dataset,file))
    if is_test:
        Train = Train[:10000]
    return Train

def load_stft_datas_path(is_test = False,pseudo = None):
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
    
    realNum = len(Train)
        
    if pseudo is not None:
        pseudoTrainPath = '../eggdata/TrainSTFT/pseudo_label/%s/'%pseudo
        for dataset in os.listdir(pseudoTrainPath):
            for file in os.listdir(pseudoTrainPath + dataset):
                if 'npy' in file:
                    Train.append(os.path.join(pseudoTrainPath + dataset,file))
    if is_test:
        Train = Train[:1000] + Train[-1000:]
        Val = Val[:1000] + Val[-1000:]
        
    return Train,Val

def pseudoPath2Path(pseudoPath):
    path = '../eggdata/TrainSTFT/unlabeled/' + '/'.join(pseudoPath.split('/')[-2:])
    return path

class Dataset(torch.utils.data.Dataset):
    def __init__(self,X,n_frame,is_train = True,normal_noise = None,musical_noise=None,aug=None):
        self.data = X
        self.n_frame = n_frame
        self.is_train = is_train
        self.normal_noise = normal_noise
        self.musical_noise = musical_noise
        self.aug= aug
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        if isinstance(self.data[idx],str):
            [X,Y] = np.load(self.data[idx])
        else:
            [X,Y] = self.data[idx]
        
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
        
        # Mixing before mel
        if self.aug:
            X = self.aug(X,self.normal_noise,self.musical_noise)
        X = stft_process(X) ## 2,F,T mag,phase
        Y = stft_process(Y,mask=True) ## 3,F,T mag,phase,mask
        
        # SpecAug after mel
        if self.is_train:
            X = add_whitenoise(X)
            X = spec_masking(X, F = 5, T = 1, num_masks = 10, prob = 1, replace_with_zero = True)
            
        return X,Y


class STDataset(torch.utils.data.Dataset):
    def __init__(self,X,n_frame,is_train = True,normal_noise = None,musical_noise=None,aug=None):
        self.data = X
        self.n_frame = n_frame
        self.is_train = is_train
        self.normal_noise = normal_noise
        self.musical_noise = musical_noise
        self.aug= aug
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        pseudo = 1 if 'pseudo' in self.data[idx] else 0
        
#         a = time.time()
        if isinstance(self.data[idx],str):
            if pseudo == 0:
                [X,Y] = np.load(self.data[idx])
            else:
                X = np.load(pseudoPath2Path(self.data[idx]))
                Y = np.load(self.data[idx])
        else:
            [X,Y] = self.data[idx]
        
#         print('pseudo :',pseudo==1, time.time()-a)
        
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
            Y = Y[:,pi:pi+self.n_frame] if pseudo == 0 else Y[:,:,pi:pi+self.n_frame]
        
        # Mixing before mel
        if self.aug:
            X = self.aug(X,self.normal_noise,self.musical_noise)
        X = stft_process(X) ## 2,F,T mag,phase
        if pseudo == 0:
            Y = stft_process(Y,mask=True) ## 3,F,T mag,phase,mask
        else:
            Y[2,:,:] = np.round(expit(Y[2,:,:]))
            
        # SpecAug after mel
        if self.is_train:
            X = add_whitenoise(X)
            X = spec_masking(X, F = 5, T = 1, num_masks = 10, prob = 1, replace_with_zero = True)
            
        return X,Y,pseudo
    
class SSLDataset(torch.utils.data.Dataset):
    def __init__(self,X,unlabel,n_frame,is_train = True,normal_noise = None,musical_noise=None,aug=None):
        self.data = X
        self.unlabel = unlabel
        self.n_frame = n_frame
        self.is_train = is_train
        self.normal_noise = normal_noise
        self.musical_noise = musical_noise
        self.aug= aug
        
        self.data_len = len(self.data)
    def __len__(self):
        return len(self.data) + len(self.unlabel)
    
    def __getitem__(self,idx):
        labeled = 1
        if idx < self.data_len:
            if isinstance(self.data[idx],str):
                [X,Y] = np.load(self.data[idx],allow_pickle=True)
            else:
                [X,Y] = self.data[idx]
        else:
            if isinstance(self.unlabel[self.data_len - idx],str):
                X = np.load(self.unlabel[self.data_len - idx],allow_pickle=True)
            else:
                X = self.unlabel[self.data_len - idx]
            labeled = 0
        ### random crop
        T = X.shape[1]
        if T<self.n_frame:
            left = np.random.randint(self.n_frame-T)
            right = self.n_frame-T-left
            X = np.pad(X,((0,0),(left,right)),mode = 'constant',constant_values=(-1))
            if labeled == 1:
                Y = np.pad(Y,((0,0),(left,right)),mode = 'constant',constant_values=(-1))
        elif T>self.n_frame:
            pi = np.random.randint(T-self.n_frame)
            X = X[:,pi:pi+self.n_frame]
            if labeled == 1:
                Y = Y[:,pi:pi+self.n_frame]
        
        # Mixing before mel
        if self.aug:
            X1 = self.aug(X,self.normal_noise,self.musical_noise)
            X2 = self.aug(X,self.normal_noise,self.musical_noise)
        
        X1 = stft_process(X1) ## 2,F,T mag,phase
        X2 = stft_process(X2) ## 2,F,T mag,phase
        if labeled == 1:
            Y = stft_process(Y,mask=True) ## 3,F,T mag,phase,mask
        else:
            Y = np.zeros((3,257,self.n_frame)).astype(X1.dtype)
        
        # SpecAug after mel
        if self.is_train:
            X1 = add_whitenoise(X1)
            X1 = spec_masking(X1, F = 10, T = 2, num_masks = 3, prob = 1, replace_with_zero = True)
            X2 = add_whitenoise(X2)
            X2 = spec_masking(X2, F = 10, T = 2, num_masks = 3, prob = 1, replace_with_zero = True)

        return (X1,X2),Y, labeled
"""