import numpy as np
import librosa
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

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return np.log(np.maximum(x,clip_val) * C)

def stft_to_mel(stft):
    yS = np.abs(stft)
    yS = librosa.feature.melspectrogram(S=yS,sr=16000,n_mels=80,n_fft=512, hop_length=128,fmax=8192,fmin=60)
    yS = -dynamic_range_compression(yS)
    return yS

'''
def loudness_normalize(audio, criterion = 0.1):
    power = np.mean(audio**2)
    return audio*criterion/(power+1e-3)

def stft_normalize(stft_magnitude,stft_phase):
    """
    torch Tensor B,F,T
    """
    return torch.log10(1+stft_magnitude),stft_phase/np.pi

def stft_denormalize(stft_magnitude,stft_phase):
    """
    torch Tensor B,F,T
    """
    return torch.pow(10,stft_magnitude)-1 , stft_phase*np.pi

def stft_with_phase(audio,n_fft,hop_length,mode = 'concat'):
    stft = librosa.core.stft(np.asfortranarray(audio),n_fft = n_fft,hop_length = hop_length)
    mag = np.log10(1+np.abs(stft)) # [F,T] 
    phase = np.angle(stft)/np.pi # [F,T] + normalize with PI 
    mag = mag[np.newaxis,:,:] # [1,F,T]
    phase = phase[np.newaxis,:,:]# [1,F,T]
    
    if mode == 'concat':
        embedding = np.concatenate([mag,phase],axis=1)
    if mode == 'channel':
        embedding = np.concatenate([mag,phase],axis=0)
    return embedding

def istft(X,hop_length,n_fft,length):
    stft = X[0]*(np.cos(X[1])+1j*np.sin(X[1]))## 2,F,T-> F,T
    return librosa.core.istft(stft, hop_length=hop_length, win_length=n_fft,length=length)
'''

def normalize_batch(batch):
    """
    input shape : B,1,F,T
    output shape : B,3,F,T with normalized batch
    """
    # normalize using imagenet mean and std
    vgg_batch = (1 + batch.clone())/2 ## normalize to [0,1]
    mean = vgg_batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = vgg_batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    vgg_batch = (vgg_batch - mean) / std
    return vgg_batch
    
