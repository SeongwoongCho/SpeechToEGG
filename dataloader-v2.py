import os
import numpy as np
from tqdm import tqdm
from utils import *
import librosa
import random
import multiprocessing
import torch.utils.data

seed_everything(42)

def process_cmu(t): ## args : drt, file,window,step

    drt,file,window,step = t
    X,y = [],[]

    x,sr = librosa.load(drt+file,sr=16000,mono=False)
    itvs = librosa.effects.split(x[0],frame_length = int(window*0.75), hop_length = int(window*0.5),top_db = 10)

    x[0] = butter_lowpass_filter(x[0],2500,16000)
    for st_idx,end_idx in itvs:
        speech,egg = x[0][st_idx:end_idx],x[1][st_idx:end_idx]
        X.append(speech)
        X.append(egg)
    if file[7]=='a':
        return (True,X,y)
    elif file[7] =='b':
        return (False,X,y)
    
def process_saarbrucken(t):
    drt,file,window,step = t
    X,Y = [],[]

    x,sr = librosa.load(drt+file,sr=50000)
    y,sr = librosa.load(drt+file[:-4]+'-egg.wav',sr=50000)
    if len(x) > len(y):
        x = x[:len(y)]
    elif len(x) < len(y):
        y = y[:len(x)]
    x = librosa.resample(x, sr, 16000)
    y = librosa.resample(y, sr, 16000)
    itvs = librosa.effects.split(x,frame_length = int(window*0.75), hop_length = int(window*0.5),top_db = 10)
    x = butter_lowpass_filter(x,2500,16000)
    
    for st_idx,end_idx in itvs:
        speech,egg = x[st_idx:end_idx],y[st_idx:end_idx]
        X.append(speech)
        Y.append(egg)
    
    if np.random.uniform()<0.7:
        return (True,X,Y)
    else:
        return (False,X,Y)
    
def load_datas(n_frame,window,step):
    train_X = []
    train_y = []
    val_X = []
    val_y = []
   
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    
    print("load saarbrucken data sets")
    args = []
    files = os.listdir('./datasets/saarbrucken/export/')
    for file in files:
        if 'wav' in file and file[:-4]+'-egg.wav' in files:
            args.append(('./datasets/saarbrucken/export/',file,window,step))
            
    tmp1 = pool.map(process_saarbrucken,args)
#     tmp1 = []
    
    print("load cmu data sets")
    args = []
    for drt in ['./datasets/cmu_us_bdl_arctic/orig/','./datasets/cmu_us_jmk_arctic/orig/','./datasets/cmu_us_slt_arctic/orig/']:
        for file in os.listdir(drt):
            args.append((drt,file,window,step))
    tmp2 = pool.map(process_cmu,args)
    
    pool.close()
    pool.join()
    
    print("memory managing...")
    
    tmp1 = tmp1+tmp2
    release_list(tmp2)
    
    for b,x,y  in tqdm(tmp1):
        if len(x) > 0:
            if b:
                train_X += x
                train_y += y
            else:
                val_X += x
                val_y += y
                
    release_list(tmp1)
    
#     _train_X = np.array(train_X)
#     release_list(train_X)
#     _train_y = np.array(train_y)
#     release_list(train_y)
#     _val_X = np.array(val_X)
#     release_list(val_X)
#     _val_y = np.array(val_y)
#     release_list(val_y)
    
    return train_X, train_y, val_X, val_y

class Dataset(torch.utils.data.Dataset):
    def __init__(self,X,y,n_frame,stride,is_train,aug = None):
        self.X = X
        self.y = y
        self.n_frame = n_frame
        self.stride = stride
        self.is_train = is_train
        self.aug = aug
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
#         if len(self.X[idx]) == self.n_frame:
#             pi = 0
#         else:
#             pi = random.randint(0,len(self.X[idx])-self.n_frame)
#         _x,_y = self.X[idx][pi:pi+self.n_frame],self.y[idx][pi:pi+self.n_frame]
        #preprocess
        speech,egg = self.X[idx], self.y[idx]
        speech = normalize(speech)
        egg = normalize(egg)
        if self.aug:
            speech = self.aug(speech)
        speech,egg = np.expand_dims(speech,axis=-1),np.expand_dims(egg,axis=-1)
        speech, egg = (
            th.from_numpy(speech).unfold(0, self.n_frame, self.stride),
            th.from_numpy(egg).unfold(0, self.n_frame, self.stride),
        )

        return speech, egg