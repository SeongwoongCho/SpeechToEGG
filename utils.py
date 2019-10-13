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
    if(np.max(x[:192]) - np.min(x[:192]) < 0.1):
        return False
    if(np.max(y[:192])-np.min(y[:192])<threshold):
        return False
    if(np.max(x[-192:]) - np.min(x[-192:]) < 0.1):
        return False
    if(np.max(y[-192:])-np.min(y[-192:])<threshold):
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

def custom_aug(x,normal_noise,musical_noise):
    db =np.random.uniform(low=-5,high=35)
    p = np.random.uniform()
    if p<0.9 and db>0:
        if p<0.55:
            pi = random.randint(0,len(musical_noise)-1)
            pi2 = random.randint(0,len(musical_noise[pi])-193)
            y = musical_noise[pi][pi2:pi2+192]
            if np.max(y)-np.min(y)>0.1:
                y = normalize(y)
        else:
            pi = random.randint(0,len(normal_noise)-1)
            pi2 = random.randint(0,len(normal_noise[pi])-193)
            y = normal_noise[pi][pi2:pi2+192]
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

def release_list(a):
    a.clear()
    del a
    
def CosineDistanceLoss():
    def f(pred,true):
        return torch.mean(torch.acos(F.cosine_similarity(pred,true)))
    return f