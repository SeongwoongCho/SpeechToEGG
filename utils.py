import os
import torch
import random
import numpy as np
import colorednoise
from scipy.signal import butter, lfilter, freqz

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(42)

def check_data(x,y,threshold):
    if(np.max(x)-np.min(x)<0.05):
        return False
    if(np.max(y)-np.min(y)<threshold):
        return False
    return True

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def normalize(y):
    return -1 + 2*(y-np.min(y))/(np.max(y)-np.min(y))

def add_whitenoise(x,db):
    E_x = np.mean(x**2)
    noise = np.random.normal(0,1,x.shape)
    E_noise = np.mean(noise**2)
    
    a = E_x/(E_noise*(10**(db/10)))
    lam = 1/(1+a)
    
    return lam*x+(1-lam)*noise

def add_pinknoise(x,db):
    E_x = np.mean(x**2)
    noise = colorednoise.powerlaw_psd_gaussian(1,x.shape[0])
    E_noise = np.mean(noise**2)
    
    a = E_x/(E_noise*(10**(db/10)))
    lam = 1/(1+a)
    
    return lam*x+(1-lam)*noise

def add_brownnoise(x,db):
    E_x = np.mean(x**2)
    noise = colorednoise.powerlaw_psd_gaussian(1,x.shape[0])
    E_noise = np.mean(noise**2)
    
    a = E_x/(E_noise*(10**(db/10)))
    lam = 1/(1+a)
    
    return lam*x+(1-lam)*noise