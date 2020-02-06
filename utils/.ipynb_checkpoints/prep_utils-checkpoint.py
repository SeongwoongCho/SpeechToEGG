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

def dynamic_range_decompression(x, C=1, torch_mode=False):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    if torch_mode:
        return torch.exp(x) / C
    return np.exp(x) / C

def loudness_normalize(x):
    return x/(np.mean(x**2)*10+1e-4)

def mag_normalize(mag):
    return 10*mag/(np.mean(mag**2) + 1e-4)

def stft_process(stft,mask=False):
    mag = np.abs(stft)
    phase = np.angle(stft)
    
    if not mask:
        mag = mag_normalize(mag) ## input만 normalize 시켜준다. mask는 output에 대해서만 구해준다.
    mag = dynamic_range_compression(mag)
    mag = mag[np.newaxis,:,:]
    phase = phase[np.newaxis,:,:]

    if mask:
        m = make_mask(mag)
        conc = np.concatenate([mag,phase,m],axis=0)
    else:
        conc = np.concatenate([mag,phase],axis=0)
    return conc

def make_mask(mag,torch_mode=False):
    if torch_mode:
        mask = torch.zeros_like(mag)
        m = torch.mean(mag)
        s = torch.std(mag)
    else:
        mask = np.zeros_like(mag)
        m = mag.mean()
        s = mag.std()
        
    mask[mag>m + 1*s] = 1
    return mask

'''
def stft_to_mel(stft):
    yS = np.abs(stft)
    yS = librosa.feature.melspectrogram(S=yS,sr=16000,n_mels=80,n_fft=512, hop_length=128,fmax=8192,fmin=60)
    yS = -dynamic_range_compression(yS)
    return yS

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
    
def diff(x, axis):
    """Take the finite difference of a tensor along an axis.
    Args:
    x: Input tensor of any dimension.
    axis: Axis on which to take the finite difference.
    Returns:
    d: Tensor with size less than x by 1 along the difference dimension.
    Raises:
    ValueError: Axis out of range for tensor.
    """
    shape = x.shape

    begin_back = [0 for unused_s in range(len(shape))]
#     print("begin_back",begin_back)
    begin_front = [0 for unused_s in range(len(shape))]

    begin_front[axis] = 1
#     print("begin_front",begin_front)

    size = list(shape)
    size[axis] -= 1
#     print("size",size)
    slice_front = x[begin_front[0]:begin_front[0]+size[0], begin_front[1]:begin_front[1]+size[1]]
    slice_back = x[begin_back[0]:begin_back[0]+size[0], begin_back[1]:begin_back[1]+size[1]]

#     slice_front = tf.slice(x, begin_front, size)
#     slice_back = tf.slice(x, begin_back, size)
#     print("slice_front",slice_front)
#     print(slice_front.shape)
#     print("slice_back",slice_back)

    d = slice_front - slice_back
    return d


def unwrap(p, discont=np.pi, axis=-1):
    """Unwrap a cyclical phase tensor.
    Args:
    p: Phase tensor.
    discont: Float, size of the cyclic discontinuity.
    axis: Axis of which to unwrap.
    Returns:
    unwrapped: Unwrapped tensor of same size as input.
    """
    dd = diff(p, axis=axis)
#     print("dd",dd)
    ddmod = np.mod(dd+np.pi,2.0*np.pi)-np.pi  # ddmod = tf.mod(dd + np.pi, 2.0 * np.pi) - np.pi
#     print("ddmod",ddmod)

    idx = np.logical_and(np.equal(ddmod, -np.pi),np.greater(dd,0)) # idx = tf.logical_and(tf.equal(ddmod, -np.pi), tf.greater(dd, 0))
#     print("idx",idx)
    ddmod = np.where(idx, np.ones_like(ddmod) *np.pi, ddmod) # ddmod = tf.where(idx, tf.ones_like(ddmod) * np.pi, ddmod)
#     print("ddmod",ddmod)
    ph_correct = ddmod - dd
#     print("ph_corrct",ph_correct)
    
    idx = np.less(np.abs(dd), discont) # idx = tf.less(tf.abs(dd), discont)
    
    ddmod = np.where(idx, np.zeros_like(ddmod), dd) # ddmod = tf.where(idx, tf.zeros_like(ddmod), dd)
    ph_cumsum = np.cumsum(ph_correct, axis=axis) # ph_cumsum = tf.cumsum(ph_correct, axis=axis)
#     print("idx",idx)
#     print("ddmod",ddmod)
#     print("ph_cumsum",ph_cumsum)
    
    
    shape = np.array(p.shape) # shape = p.get_shape().as_list()

    shape[axis] = 1
    ph_cumsum = np.concatenate([np.zeros(shape, dtype=p.dtype), ph_cumsum], axis=axis) 
    #ph_cumsum = tf.concat([tf.zeros(shape, dtype=p.dtype), ph_cumsum], axis=axis)
    unwrapped = p + ph_cumsum
#     print("unwrapped",unwrapped)
    return unwrapped
