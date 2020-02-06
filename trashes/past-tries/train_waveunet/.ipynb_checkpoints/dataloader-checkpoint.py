import os
import numpy as np
from utils import seed_everything,loudness_normalize,add_whitenoise
import torch.utils.data

seed_everything(42)

class Dataset(torch.utils.data.Dataset):
    def __init__(self,X,n_frame=1024,stride = 512, is_train=True,aug = None):
        self.X = X
        self.n_frame = n_frame
        self.stride = stride
        self.is_train = is_train
        self.aug = aug
        
    def __len__(self):
        return (self.X.shape[1]-self.n_frame)//self.stride
    def __getitem__(self,idx):
        start_frame = idx*self.stride + np.random.randint(0,self.n_frame-1)
        X = np.array(self.X[0,start_frame : start_frame + self.n_frame].tolist())
        y = np.array(self.X[1,start_frame : start_frame + self.n_frame].tolist())
        X = loudness_normalize(X)
        if self.is_train:
            db = np.random.uniform(-5,30)
            X = add_whitenoise(X,db)
        
        if self.aug:
            X = self.aug(X,self.normal_noise,self.musical_noise)
        return np.expand_dims(X,axis=-1).astype('float32'),np.expand_dims(y,axis=-1).astype('float32')