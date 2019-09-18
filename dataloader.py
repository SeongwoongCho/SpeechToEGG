import os
import numpy as np
from tqdm import tqdm
from utils import *
import librosa
import random

def load_datas(n_frame,window,step):
    train_X = []
    train_y = []
    val_X = []
    val_y = []

    # for idx,drt in tqdm_notebook(enumerate(os.listdir('./datasets/voco'))):

    #     for file in os.listdir('./datasets/voco/'+drt):
    #         if 'EGG' in file and ('wav' in file or 'WAV' in file):
    #             try:
    #                 x,sr = librosa.load('./datasets/voco/'+drt+'/'+file,sr=48000,mono=False)
    #                 x = librosa.resample(x, sr, 16000)
    #                 itvs = librosa.effects.split(x[0],frame_length = 1024, hop_length = 512)

    #                 for st_idx,end_idx in itvs:
    #                     speech,egg = x[0][st_idx:end_idx],x[1][st_idx:end_idx]
    #                     speech = butter_lowpass_filter(speech,2500,16000)
    #                     i=0
    #                     while(i*step+window < len(speech)):
    #                         tmp_speech = speech[i*step:i*step+window]
    #                         tmp_egg = egg[i*step:i*step+window]
    #                         if check_data(tmp_speech,tmp_egg,0.4):
    #                             tmp_egg = normalize(tmp_egg)
    #                             if idx<50:
    #                                 train_X.append(tmp_speech)
    #                                 train_y.append(tmp_egg)
    #                             else:
    #                                 val_X.append(tmp_speech)
    #                                 val_y.append(tmp_egg)
    #                         i+=1
    #             except:
    #                 print('nop')
    #                 continue


    for drt in ['./datasets/cmu_us_bdl_arctic/orig/','./datasets/cmu_us_jmk_arctic/orig/','./datasets/cmu_us_slt_arctic/orig/']:
        for file in tqdm(os.listdir(drt)):
            x,sr = librosa.load(drt+file,sr=16000,mono=False)
            itvs = librosa.effects.split(x[0],frame_length = 1024, hop_length = 512)

            for st_idx,end_idx in itvs:
                speech,egg = x[0][st_idx:end_idx],x[1][st_idx:end_idx]
                speech = butter_lowpass_filter(speech,2500,16000)
                i=0
                while(i*step+window < len(speech)):
                    tmp_speech = speech[i*step:i*step+window]
                    tmp_egg = egg[i*step:i*step+window]
                    if check_data(tmp_speech,tmp_egg,0.4):
                        if file[7]=='a':
                            train_X.append(tmp_speech)
                            train_y.append(tmp_egg)                   
                        if file[7]=='b':    
                            val_X.append(tmp_speech)
                            val_y.append(tmp_egg)
                    i+=1
    return np.array(train_X), np.array(train_y), np.array(val_X), np.array(val_y)

class Dataset(torch.utils.data.Dataset):
    def __init__(self,X,y,n_frame,is_train,aug = None):
        self.X = X
        self.y = y
        self.n_frame = n_frame
        self.is_train = is_train
        self.aug = aug
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        pi = random.randint(0,len(self.X[idx])-self.n_frame)        
        _x,_y = self.X[idx][pi:pi+self.n_frame],self.y[idx][pi:pi+self.n_frame]
        _x = normalize(_x)
        _y = normalize(_y)
        if self.aug:
            _x = self.aug(_x)
        return np.expand_dims(_x,axis=-1),np.expand_dims(_y,axis=-1)