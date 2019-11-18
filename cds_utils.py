import os
import torch
from torch.nn import functional as F
import random
import numpy as np
import colorednoise ## whitenoise, pinknoise, brownnoise
import librosa
import torch.distributed as dist
from scipy.signal import butter, lfilter, freqz, filtfilt, medfilt, savgol_filter
from findpeaks.tests.libs import detect_peaks
import multiprocessing

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
    mag = np.log10(1+np.abs(stft)) # [F,T] 
    phase = np.angle(stft)/np.pi # [F,T] + normalize with PI 
    mag = mag[np.newaxis,:,:] # [1,F,T]
    phase = phase[np.newaxis,:,:]# [1,F,T]
    
    if mode == 'concat':
        embedding = np.concatenate([mag,phase],axis=1)
    if mode == 'channel':
        embedding = np.concatenate([mag,phase],axis=0)
    return embedding

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= 4
    return rt

def istft(X,hop_length,n_fft,length):
    stft = X[0]*(np.cos(X[1])+1j*np.sin(X[1]))## 2,F,T-> F,T
    return librosa.core.istft(stft, hop_length=hop_length, win_length=n_fft,length=length)

#TODO
def _get_CQSQ(arg):
        stft_pred,stft_target,hop_length,n_fft = arg
        _egg_pred = istft(stft_pred,hop_length,n_fft)
        _egg_true = istft(stft_target,hop_length,n_fft)
        _egg_pred = smooth(_egg_pred, 25)
        _degg_pred = np.gradient(_egg_pred,edge_order = 2)
        _degg_true = np.gradient(_egg_true,edge_order = 2)

        CQ_pred,SQ_pred = get_CQSQ(_egg_pred,_degg_pred)
        CQ_true,SQ_true = get_CQSQ(_egg_true,_degg_true)
        return (CQ_pred-CQ_true)**2,(SQ_pred-SQ_true)**2
    
def metric(stft_pred,stft_target,hop_length,n_fft,length,pool=None):
    """
    stft_pred : B,2,F,T ndarray
    stft_target : B,2,F,T ndarray
    return CQ_difference,SQ_difference values
    """
    B = stft_pred.shape[0]    
    
    stft_pred[:,0,:,:]=np.power(10,stft_pred[:,0,:,:])-1
    stft_pred[:,1,:,:]=stft_pred[:,1,:,:]*np.pi
    
    stft_target[:,0,:,:]=np.power(10,stft_target[:,0,:,:])-1
    stft_target[:,1,:,:]=stft_target[:,1,:,:]*np.pi
    
    CQ_diff,SQ_diff = [],[]
    if pool is not None:
        args = []
        for i in range(B):
            args.append((stft_pred[i],stft_target[i],hop_length,n_fft))
#         print(args)
        m = pool.map(_get_CQSQ,args)
        for x,y in m:
            CQ_diff.append(x)
            SQ_diff.append(y)
    else:
        for i in range(B):
            _egg_pred = istft(stft_pred[i],hop_length,n_fft,length)
            _egg_true = istft(stft_target[i],hop_length,n_fft,length)
            _egg_pred = smooth(_egg_pred, 25)
            _degg_pred = np.gradient(_egg_pred,edge_order = 2)
            _degg_true = np.gradient(_egg_true,edge_order = 2)

            CQ_pred,SQ_pred = get_CQSQ(_egg_pred,_degg_pred)
            CQ_true,SQ_true = get_CQSQ(_egg_true,_degg_true)

            CQ_diff.append(np.abs(CQ_pred-CQ_true))
            SQ_diff.append(np.abs(SQ_pred-SQ_true))
    
    CQ_diff_avg = np.mean(CQ_diff)
    SQ_diff_avg = np.mean(SQ_diff)
    CQ_diff_std = np.std(CQ_diff)
    SQ_diff_std = np.std(SQ_diff)
    
    return CQ_diff_avg,SQ_diff_avg,CQ_diff_std,SQ_diff_std

def frobenius_norm(x):
    """
    calculate frobenius_norm accros F,T
    x : B,F,T 
    
    return (B,)
    """
    return torch.mean(x**2, dim = (1,2))

def SC_loss(stft_magnitude_pred,stft_magnitude_target):
    """
    calculation of spectral convergence loss
    
    stft_magnitude_pred   : B,F,T Tensor
    stft_magnitude_target : B,F,T Tensor
    
    return SC_loss : constant
    """
    epsilon = 1e-4
    numerator = frobenius_norm(stft_magnitude_pred-stft_magnitude_target)
    denominator = frobenius_norm(stft_magnitude_target) + epsilon
    loss = torch.mean(numerator/denominator)
#     print(numerator)
#     print(denominator)
#     print(loss)
    
    ## issue 없음
#     if loss.item()>10:
#         print(numerator)
#         print(denominator)
#         print("SC:",loss.item())


    return loss

def LM_loss(stft_magnitude_pred,stft_magnitude_target):
    """
    calculation of Log-scale STFT-magnitude loss
    
    stft_magnitude_pred   : B,F,T Tensor
    stft_magnitude_target : B,F,T Tensor
    
    return LM_loss : constant
    """
    epsilon = 1e-4
#     print(stft_magnitude_pred.min())
    loss = torch.abs(stft_magnitude_pred-stft_magnitude_target) ## log 값이 씌어져 있는 상태임
#     print(loss.max())
    loss = torch.mean(loss,dim=(0,1,2))
#     print(loss)
    
    ## issue없음
    
#     if loss.item()>10:
#         print(stft_magnitude_pred)
#         print(stft_magnitude_target)
#         print("LM:",loss.item())
        
    return loss

def IF_loss(stft_phase_pred,stft_phase_target):
    """
    calculation of Instantaneous frequency loss
    
    stft_phase_pred   : B,F,T Tensor
    stft_phase_target : B,F,T Tensor
    
    return IF_loss : constant
    """
    phase_difference_pred = stft_phase_pred[:,:,1:] - stft_phase_pred[:,:,:-1]
    phase_difference_target = stft_phase_target[:,:,1:] - stft_phase_target[:,:,:-1]
    loss = torch.abs(phase_difference_pred - phase_difference_target)
    loss = torch.mean(loss, dim=(0,1,2))
    #     if loss.item()>10:
#         print("IF:",loss.item())
    
    ## issue없음
    
    return loss
    
def WP_loss(stft_pred,stft_target):
    """
    calculation of Weighted phase loss
    
    stft_pred   : B,2,F,T Tensor
    stft_target : B,2,F,T Tensor
    
    return WP_loss
    """
    stft_magnitude_pred = torch.sqrt(stft_pred[:,0,:,:]**2 + stft_pred[:,1,:,:]**2)
    stft_magnitude_target = torch.sqrt(stft_target[:,0,:,:]**2 + stft_target[:,1,:,:]**2)
    
    stft_amplitude_pred = stft_pred[:,0,:,:]
    stft_phase_pred = stft_pred[:,1,:,:]
    
    stft_amplitude_target = stft_target[:,0,:,:]
    stft_phase_target = stft_target[:,1,:,:]

    loss = stft_magnitude_pred*stft_magnitude_target - stft_amplitude_pred*stft_amplitude_target - stft_phase_pred*stft_amplitude_target
    loss = torch.mean(torch.abs(loss),dim=(0,1,2))
    
#     if loss.item()>1000:
#         print("WP:",loss.item())
    
    return loss   

def spectral_loss(coeff = [1,5,5,1]):
    def _spectral_loss(stft_pred,stft_target,coeff=coeff):
        stft_magnitude_pred = stft_pred[:,0,:,:]
        stft_magnitude_target = stft_target[:,0,:,:]
        stft_phase_pred = stft_pred[:,1,:,:]
        stft_phase_target = stft_target[:,1,:,:]

        coeff = coeff[:3]
        coeff = coeff/np.sum(coeff)
        
#         sc_loss = SC_loss(stft_magnitude_pred,stft_magnitude_target)
        lm_loss = LM_loss(stft_magnitude_pred,stft_magnitude_target)
        if_loss = IF_loss(stft_phase_pred,stft_phase_target)
#         loss= coeff[0]*sc_loss+coeff[1]*lm_loss+coeff[2]*if_loss
        loss = coeff[1]*lm_loss+coeff[2]*if_loss
#         print("sc:",sc_loss)
#         print("lm:",lm_loss)
#         print("if:",if_loss)
        
#         print(loss)
        
#         if loss > 10:
#             print(sc_loss)
#             print(lm_loss)
#             print(if_loss)
        
#         print(loss)
#         coeff[3]*WP_loss(stft_pred,stft_target)
        return loss
    return _spectral_loss

def get_CQSQ(EGG,DEGG):
    CQ,SQ = [],[]
    DEGG_low= DEGG.copy()
    DEGG_low[DEGG_low>0] =0
    
    DEGG_low = detect_peaks.detect_peaks(-DEGG_low,mph=0.01, mpd=45)
    DEGG_high = []
    for i in range(len(DEGG_low)-1):
        DEGG_high.append(DEGG_low[i] + np.argmax(DEGG[DEGG_low[i]:DEGG_low[i+1]]))
    DEGG_high = np.array(DEGG_high)
    
    for i in range(len(DEGG_high)-1):
        _CQ = (DEGG_low[i+1]-DEGG_high[i])/(DEGG_high[i+1]-DEGG_high[i])
        if 0.15<_CQ<0.85:
            CQ.append(_CQ)
    for i in range(len(DEGG_high)):
        _SQ = np.argmax(EGG[DEGG_high[i]:DEGG_low[i+1]])/(DEGG_low[i+1]-DEGG_high[i])
        _SQ = 1/(_SQ+1e-3) - 1
        if _SQ<10:
            SQ.append(_SQ)
    CQ = np.mean(CQ).astype('float32') if len(CQ)!=0 else 0
    SQ = np.mean(SQ).astype('float32') if len(SQ)!=0 else 0 
    return CQ,SQ

def MCWfilter(audio):
    """
    implementation of MultiChannel Wiener filter
    
    audio : ndarray
    
    return 
    """
    return

def SpecAug(audio):
    return