import os
import torch
from torch.nn import functional as F
import random
import numpy as np
import colorednoise ## whitenoise, pinknoise, brownnoise
import librosa
import torch.distributed as dist
from scipy.signal import butter, lfilter, freqz, filtfilt, medfilt, savgol_filter

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

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 

def print_verbose(verbose):
    def _print_verbose(inp):
        if verbose:
            print(inp)
    return _print_verbose    
    
def mix_db(x,y,db):
    E_x = np.mean(x**2)
    E_y = np.mean(y**2)
    
    a = E_x/(E_y*(10**(db/10)))
    lam = 1/(1+a)
    return lam*x + (1-lam)*y
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

def loudness_normalize(audio, criterion = 0.1):
    power = np.mean(audio**2)
    return audio*criterion/(power+1e-3)

def stft_with_phase(audio,n_fft,hop_length,mode = 'concat'):
    stft = librosa.core.stft(np.asfortranarray(audio),n_fft = n_fft,hop_length = hop_length)
    amp = stft.real #[F,T]
    phase = stft.imag # [F,T]
    amp = amp[np.newaxis,:,:] # [1,F,T]
    phase = phase[np.newaxis,:,:]# [1,F,T]
    
    if mode == 'concat':
        embedding = np.concatenate([amp,phase],axis=1)
    if mode == 'channel':
        embedding = np.concatenate([amp,phase],axis=0)
    return embedding

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= 4
    return rt

def istft(X,hop_length,n_fft):
    X = X[0] + 1j * X[1]## 2,F,T-> F,T
    return librosa.core.istft(X, hop_length=hop_length, win_length=n_fft)
