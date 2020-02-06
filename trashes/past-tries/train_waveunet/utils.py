import os
import torch
from torch.nn import functional as F
import random
import numpy as np
import colorednoise
import librosa
from scipy.signal import butter, lfilter, freqz, filtfilt, medfilt, savgol_filter
import torch.distributed as dist
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(42)

def loudness_normalize(y):
    return y/(np.sqrt(np.mean(y**2)) + 1e-2)

def add_whitenoise(x,db):
    noise = np.random.normal(0,1,x.shape)
    return mix_db(x,noise,db)

def add_pinknoise(x,db):
    noise = colorednoise.powerlaw_psd_gaussian(1,x.shape[0])
    return mix_db(x,noise,db)

def add_brownnoise(x,db):
    noise = colorednoise.powerlaw_psd_gaussian(2,x.shape[0])
    return mix_db(x,noise,db)

def mix_db(x,y,db):
    E_x = np.mean(x**2)
    E_y = np.mean(y**2)
    
    a = E_x/(E_y*(10**(db/10)))
    lam = 1/(1+a)
    return lam*x+(1-lam)*y

def custom_aug(n_frame = 192):
    def _custom_aug(x,normal_noise,musical_noise):
        db =np.random.uniform(low=-5,high=35)
        p = np.random.uniform()
        if p<0.9 and db>0:
            if p<0.55:
                pi = random.randint(0,len(musical_noise)-1)
                pi2 = random.randint(0,len(musical_noise[pi])-n_frame-1)
                y = musical_noise[pi][pi2:pi2+n_frame]
                if np.max(y)-np.min(y)>0.1:
                    y = normalize(y)
            else:
                pi = random.randint(0,len(normal_noise)-1)
                pi2 = random.randint(0,len(normal_noise[pi])-n_frame-1)
                y = normal_noise[pi][pi2:pi2+n_frame]
                if np.max(y)-np.min(y)>0.1:
                    y = normalize(y)
        else:
            if p < 0.94:
                y = colorednoise.powerlaw_psd_gaussian(0,x.shape[0]) #whitenoise
            elif p < 0.98:
                y = colorednoise.powerlaw_psd_gaussian(1,x.shape[0]) #pinknoise
            else:
                y = colorednoise.powerlaw_psd_gaussian(2,x.shape[0]) # brownnoise
            y = normalize(y)
        return mix_db(x,y,db)
    return _custom_aug

def custom_aug_v2(n_frame = 192):
    def _custom_aug(x,normal_noise,musical_noise):
        db_1 = np.random.uniform(low=0,high=35)
        db_2 = np.random.uniform(low=-5,high=45)
        
        p_1 = np.random.uniform()
        p_2 = np.random.uniform()
  
        if 0.1 < p_1 < 0.6:
            pi = random.randint(0,len(normal_noise)-1)
            pi2 = random.randint(0,len(normal_noise[pi])-n_frame-1)
            y = normal_noise[pi][pi2:pi2+n_frame]
            if np.max(y)-np.min(y)>0.1:
                y = normalize(y)
            x = mix_db(x,y,db_1)
        elif 0.6 < p_1:
            pi = random.randint(0,len(musical_noise)-1)
            pi2 = random.randint(0,len(musical_noise[pi])-n_frame-1)
            y = musical_noise[pi][pi2:pi2+n_frame]
            if np.max(y)-np.min(y)>0.1:
                y = normalize(y)
            x = mix_db(x,y,db_1)
        
        if p_2 < 0.4:
            y = colorednoise.powerlaw_psd_gaussian(0,x.shape[0]) #whitenoise
        elif p_2 < 0.8:
            y = colorednoise.powerlaw_psd_gaussian(1,x.shape[0]) #pinknoise
        else:
            y = colorednoise.powerlaw_psd_gaussian(2,x.shape[0]) # brownnoise
        y = normalize(y)
        return mix_db(x,y,db_2)
    return _custom_aug

def custom_aug_v3(n_frame = 10000):
    def _custom_aug(x,normal_noise,musical_noise):
        db_1 = np.random.uniform(low=0,high=35)
        db_2 = np.random.uniform(low=-5,high=45)
        
        p_1 = np.random.uniform()
        p_2 = np.random.uniform()
  
        if 0.1 < p_1 < 0.6:
            pi = random.randint(0,len(normal_noise)-1)
            pi2 = random.randint(0,len(normal_noise[pi])-n_frame-1)
            y = normal_noise[pi][pi2:pi2+n_frame]
            x = mix_db(x,y,db_1)
        elif 0.6 < p_1:
            pi = random.randint(0,len(musical_noise)-1)
            pi2 = random.randint(0,len(musical_noise[pi])-n_frame-1)
            y = musical_noise[pi][pi2:pi2+n_frame]
            x = mix_db(x,y,db_1)
        
        if p_2 < 0.4:
            y = colorednoise.powerlaw_psd_gaussian(0,x.shape[0]) #whitenoise
        elif p_2 < 0.8:
            y = colorednoise.powerlaw_psd_gaussian(1,x.shape[0]) #pinknoise
        else:
            y = colorednoise.powerlaw_psd_gaussian(2,x.shape[0]) # brownnoise
        return mix_db(x,y,db_2)
    return _custom_aug

    
def CosineDistanceLoss():
    def f(pred,true):
        return torch.mean(torch.acos(F.cosine_similarity(pred,true,dim=1,eps = 1e-4)),dim=0)
    return f

def print_verbose(verbose):
    def _print_verbose(inp):
        if verbose:
            print(inp)
    return _print_verbose

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= 4
    return rt

def dynamic_loss(loss,ddp=True):
    if ddp:
        reduced_loss = reduce_tensor(loss.data)
        return reduced_loss.item()
    return loss.item()