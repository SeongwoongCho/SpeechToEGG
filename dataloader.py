import os
import numpy as np
import torch
import torch.utils.data

from utils.utils import seed_everything
from utils.prep_utils import stft_to_mel, stft_process
from utils.aug_utils import spec_masking, add_whitenoise
from sklearn.model_selection import train_test_split

seed_everything(42)

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