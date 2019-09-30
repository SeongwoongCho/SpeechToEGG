import os
import torch
from torch.nn import functional as F
import random
import numpy as np
import colorednoise
from scipy.signal import butter, lfilter, freqz, filtfilt, medfilt, savgol_filter

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(42)

def check_data(x,y,threshold):
    if(np.max(x) - np.min(x) < 0.1):
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

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def positions2onehot(pos, shape):
    onehot = np.zeros(shape)
    onehot[pos] = 1
    return onehot

def smooth(s, window_len=10, window="hanning"):
    if window_len < 3:
        return s

    if window == "median":
        y = medfilt(s, kernel_size=window_len)
    elif window == "savgol":
        y = savgol_filter(s, window_len, 0)
    else:
        if window == "flat":  # moving average
            w = np.ones(window_len, "d")
        else:
            w = eval("np." + window + "(window_len)")

        y = np.convolve(w / w.sum(), s, mode="same")

    return y


def normalize(y):
    return -1 + 2*(y-np.min(y))/(np.max(y)-np.min(y))
#     return y/np.max(y)

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

def custom_aug(x):
    db =np.random.uniform(low=-5,high=45)
    p = np.random.uniform()
    if p<0.33:
        return add_whitenoise(x,db)
    elif p<0.66:
        return add_pinknoise(x,db)
    else:
        return add_brownnoise(x,db)
    
def release_list(a):
    a.clear()
    del a
    
def CosineDistanceLoss():
    def f(pred,true):
        return torch.mean(torch.acos(F.cosine_similarity(pred,true)))
    return f