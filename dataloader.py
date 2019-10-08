import os
import numpy as np
from tqdm import tqdm
from utils import *
import librosa
import random
import multiprocessing
import torch.utils.data
from sklearn.model_selection import train_test_split

seed_everything(42)

def process(t): ## args : drt, file,window,step

    drt,file,window,step = t
    X,y = [],[]

    x,sr = librosa.load(drt+file,sr=16000,mono=False)
    itvs = librosa.effects.split(x[0],frame_length = int(window*0.5), hop_length = int(window*0.25),top_db = 25)
    
    ## EGG와 Speech 모두 잡음성분 제거
#     x[0] = butter_lowpass_filter(x[0],2500,16000)
#     x[1] = butter_lowpass_filter(x[1],2500,16000)
#   제거 하지 않음 -> pathological data의 정보를 최대한 줌.
    for st_idx,end_idx in itvs:
        speech,egg = x[0][st_idx:end_idx],x[1][st_idx:end_idx]
        i=0
        while(i*step+window < len(speech)):
            tmp_speech = speech[i*step:i*step+window]
            tmp_egg = egg[i*step:i*step+window]
            if check_data(tmp_speech,tmp_egg,0.2):
                X.append(tmp_speech)
                y.append(tmp_egg)
#             X.append(tmp_speech)
#             y.append(tmp_egg)
            i+=1
    return (X,y)
   
def load_datas(n_frame,window,step):
    X,y = [],[]
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    
    print("load Train Data")
    args = []
    for drt in ['./datasets/TrainData/Alexis/','./datasets/TrainData/vietnam/','./datasets/TrainData/CMU/','./datasets/TrainData/saarbrucken/']:
        for file in os.listdir(drt):
            if 'wav' in file:
                args.append((drt,file,window,step))

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

def load_noise():
    musical_noise= []
    normal_noise = []
    for file in tqdm(os.listdir('./datasets/TrainData/normal_noise/')):
        tmp = np.array([])
        x,sr = librosa.load('./datasets/TrainData/normal_noise/' + file,sr=16000)
        itvs = librosa.effects.split(x,frame_length = 96, hop_length = 48,top_db = 25)
        for st,ed in itvs:
            tmp = np.concatenate((tmp,x[st:ed]))
        normal_noise.append(tmp.astype('float32'))
        
    for file in tqdm(os.listdir('./datasets/TrainData/musical_noise/')):
        x,sr = librosa.load('./datasets/TrainData/musical_noise/' + file,sr=16000)
        musical_noise.append(x.astype('float32'))
    
    return normal_noise, musical_noise
        
class Dataset(torch.utils.data.Dataset):
    def __init__(self,X,y,n_frame,is_train,aug = None):
        self.X = X
        self.y = y
        self.n_frame = n_frame
        self.is_train = is_train
        self.aug = aug
        if is_train:
            print("load noise")
            self.normal_noise, self.musical_noise = load_noise()
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        if len(self.X[idx]) == self.n_frame:
            pi = 0
        else:
            pi = random.randint(0,len(self.X[idx])-self.n_frame-1)
        _x,_y = self.X[idx][pi:pi+self.n_frame],self.y[idx][pi:pi+self.n_frame]
        _x = normalize(_x)
        _y = normalize(_y)
        if self.aug:
            _x = self.aug(_x,self.normal_noise,self.musical_noise)
        return np.expand_dims(_x,axis=-1).astype('float32'),np.expand_dims(_y,axis=-1).astype('float32')