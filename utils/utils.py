import os
import torch
import random
import numpy as np
import torch.distributed as dist

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

def MCWfilter(audio):
    """
    implementation of MultiChannel Wiener filter
    
    audio : ndarray
    
    return 
    """
    return

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h) / np.sqrt(ch * h * w) ## prevent overflow
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    return gram

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

'''
### CQSQ utils

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

def metric_egg(egg_pred,egg_target,hop_length,n_fft,length):
    """
    egg_pred : B,2,F,T ndarray
    egg_target : B,2,F,T ndarray
    return CQ_difference,SQ_difference values
    """
    B = egg_pred.shape[0]    
    
    CQ_diff,SQ_diff = [],[]

    for i in range(B):
#       _egg_pred = istft(stft_pred[i],hop_length,n_fft,length)
#       _egg_true = istft(stft_target[i],hop_length,n_fft,length)
        _egg_pred = egg_pred[i]
        _egg_true = egg_target[i]
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

'''