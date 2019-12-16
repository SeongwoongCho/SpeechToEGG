import os
import numpy as np
from tqdm import tqdm
from utils import *
import librosa
import random
import multiprocessing as mp
import torch.utils.data
import pandas as pd
from sklearn.model_selection import train_test_split

seed_everything(42)

########################################################################################
"""
밑에거는 옛날 버전. FFT로 안 쓸 때의 방식
"""

def process(t): ## args : drt, file,window,step
    drt,file,window,step,n_frame,top_db = t
    X,y = [],[]

    x,sr = librosa.load(drt+file,sr=16000,mono=False)
    itvs = librosa.effects.split(x[0],frame_length = n_frame+1, hop_length = step,top_db = top_db)
    
    for st_idx,end_idx in itvs:
        speech,egg = x[0][st_idx:end_idx],x[1][st_idx:end_idx]
        i=0
        while(i*step+window < len(speech)):
            tmp_speech = speech[i*step:i*step+window]
            tmp_egg = egg[i*step:i*step+window]
            if check_data(tmp_speech,tmp_egg,0.2,n_frame):
                X.append(tmp_speech)
                y.append(tmp_egg)
            i+=1
    return (X,y)
   
def load_datas(n_frame,window,step,top_db,is_test=False):
    X,y = [],[]
    pool = mp.Pool(mp.cpu_count()//4)
    
    print("load Train Data")
    args = []
    
    for drt in ['./datasets/TrainData/Alexis/','./datasets/TrainData/vietnam/','./datasets/TrainData/Childer/',
                './datasets/TrainData/CMU/','./datasets/TrainData/saarbrucken/']:
        for file in os.listdir(drt):
            if 'wav' in file:
                args.append((drt,file,window,step,n_frame,top_db))
    
    if is_test:
        args = args[:50]
    
    tmp = pool.map(process,args)
    for _x,_y in tmp:
        if len(_x) >0 and len(_y) > 0:
            X +=_x
            y +=_y
    pool.close()
    pool.join()
    
    release_list(tmp)
    X = np.array(X)
    y = np.array(y)
    train_X,val_X, train_y, val_y = train_test_split(X, y, test_size=0.3, random_state=42)
    return train_X, train_y, val_X, val_y

def load_datas_path(is_test = False):
    train = os.listdir('./datasets/TrainData/trainX/')
    val = os.listdir('./datasets/TrainData/valX/')
    
    if is_test:
        train = train[:100000]
        val = val[:100000]
    
    train_X = []
    train_y = []
    val_X = []
    val_y = []
    for path in train:
        train_X.append('./datasets/TrainData/trainX/'+path)
        train_y.append('./datasets/TrainData/trainy/'+path)
    for path in val:
        val_X.append('./datasets/TrainData/valX/'+path)
        val_y.append('./datasets/TrainData/valy/'+path)
        
    return train_X, train_y, val_X, val_y

def load_noise(n_frame = 192,top_db=20,is_test=False):
    musical_noise= []
    normal_noise = []
    
    normal_dir = './datasets/TrainData/normal_noise/'
    musical_dir = './datasets/TrainData/musical_noise/'
    
    normal_noise_files = os.listdir(normal_dir)
    musical_noise_files = os.listdir(musical_dir)
    
    if is_test:
        normal_noise_files = normal_noise_files[:10]
        musical_noise_files = musical_noise_files[:10]
    
    for file in normal_noise_files:
        tmp = np.array([])
        x,sr = librosa.load(normal_dir + file,sr=16000)
        itvs = librosa.effects.split(x,frame_length = n_frame, hop_length = n_frame//4,top_db = top_db)
        for st,ed in itvs:
            tmp = np.concatenate((tmp,x[st:ed]))
        normal_noise.append(tmp.astype('float32'))
        
    for file in musical_noise_files:
        tmp = np.array([])
        x,sr = librosa.load(musical_dir + file,sr=16000)
        itvs = librosa.effects.split(x,frame_length = n_frame, hop_length = n_frame//4,top_db = top_db)
        for st,ed in itvs:
            tmp = np.concatenate((tmp,x[st:ed]))
        musical_noise.append(tmp.astype('float32'))
    
    return normal_noise, musical_noise

class Dataset(torch.utils.data.Dataset):
    def __init__(self,X,y,normal_noise=None,musical_noise=None,n_frame=192,is_train=True,aug = None):
        self.X = X
        self.y = y
        self.n_frame = n_frame
        self.is_train = is_train
        self.aug = aug
        if is_train:
#             print("load noise")
            self.normal_noise, self.musical_noise = normal_noise,musical_noise
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        X = np.load(self.X[idx])
        Y = np.load(self.y[idx])
        
        if len(X) == self.n_frame:
            pi = 0
        else:
            pi = random.randint(0,len(X)-self.n_frame-1)
        _x,_y = X[pi:pi+self.n_frame],Y[pi:pi+self.n_frame]
        _x = normalize(_x)
        _y = normalize(_y)
        if self.aug:
            _x = self.aug(_x,self.normal_noise,self.musical_noise)
        return np.expand_dims(_x,axis=-1).astype('float32'),np.expand_dims(_y,axis=-1).astype('float32')
    
########################################################################################
########################################################################################
########################################################################################

def process_v2(t): ## args : drt, file, n_frame
    drt,file,n_frame = t
    X,Y = [],[]

    [x,y],sr = librosa.load(drt+file,sr=16000,mono=False)
    audio_length = len(x)
    padding = audio%n_frame
    x = np.pad(x,pad_width = (0,padding),mode='constant')
    y = np.pad(x,pad_width = (0,padding),mode='constant')
    for frame in range(audio_length//n_frame+1):
        X.append(x[frame*n_frame:(frame+1)*n_frame])
        Y.append(y[frame*n_frame:(frame+1)*n_frame])
    return (X,Y)

def load_datas_v2(n_frame,n_fft,hop_length,is_test=False):
    X,y = [],[]
    pool = mp.Pool(mp.cpu_count())
    
    print("load Train Data")
    args = []
    
    for drt in ['./datasets/TrainData/Alexis/','./datasets/TrainData/vietnam/','./datasets/TrainData/Childer/','./datasets/TrainData/CMU/','./datasets/TrainData/saarbrucken/']:
        for file in os.listdir(drt):
            if 'wav' in file:
                args.append((drt,file,n_frame))
    
    if is_test:
        args = args[:50]
    
    tmp = pool.map(process_v2,args)
    for _x,_y in tmp:
        if len(_x) >0 and len(_y) > 0:
            X +=_x
            y +=_y
    pool.close()
    pool.join()
    X = np.array(X)
    y = np.array(y)
    train_X,val_X, train_y, val_y = train_test_split(X, y, test_size=0.3, random_state=42)
    return train_X, train_y, val_X, val_y

def load_noise_v2(n_frame,n_fft,hop_length,is_test=False):
    normal_dir = './datasets/TrainData/normal_noise/'
    musical_dir = './datasets/TrainData/musical_noise/'
    normal_noise_files = os.listdir(normal_dir)
    musical_noise_files = os.listdir(musical_dir)
    
    if is_test:
        normal_noise_files = normal_noise_files[:10]
        musical_noise_files = musical_noise_files[:10]
    
    normal_noise = [librosa.load(normal_dir + file,sr=16000) for file in normal_noise_files]
    musical_noise = [librosa.load(musical_dir + file,sr=16000) for file in musical_noise_files]
    
    return normal_noise, musical_noise

class Dataset_v2(torch.utils.data.Dataset):
    def __init__(self,X,y,n_fft,hop_length,is_train=Truenormal_noise=None,musical_noise=None,aug = None):
        self.X = X
        self.y = y
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_frame = n_frame
        self.is_train = is_train
        self.aug = aug
        if is_train:
#             print("load noise")
            self.normal_noise, self.musical_noise = normal_noise,musical_noise
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        X = loudness_normalize(self.X[idx]).astype('float32')
        Y = loudness_normalize(self.y[idx]).astype('float32')
        if self.aug:
            X = self.aug(X,self.normal_noise,self.musical_noise)
        X = stft_with_phase(X,self.n_fft,self.hop_length,'channel')
        Y = stft_with_phase(Y,self.n_fft,self.hop_length,'channel')
        return X,Y