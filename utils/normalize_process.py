from __future__ import division, print_function
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from numpy.fft import rfft
from numpy import argmax, mean, diff, log, nonzero
from scipy.signal import blackmanharris, correlate
import math
import IPython.display as ipd
"""Detect peaks in data based on their amplitude and other features."""


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indexes of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indexes by their occurrence
        ind = np.sort(ind[~idel])
    return ind

def mask(mag):
    mask = np.zeros_like(mag)
    m = mag.mean()
    s = mag.std()
#     print(m,s)
    mask[mag>m+1*s] = 1
    return mask

def find_voice_interval(mask,threshold):
    intervals = []
    
    F,T = mask.shape
    mask_sum = np.sum(mask[:100,:],axis=0)
    voiced_region = mask_sum > threshold

    for i in range(1,len(voiced_region)-1):
#         if(voiced_region[i-1] == True and voiced_region[i] == False and voiced_region[i+1] == True):
#             voiced_region[i] = True
        if(voiced_region[i-1] == False and voiced_region[i] == True and voiced_region[i+1] == False):
            voiced_region[i] = False
    
    c = True
    for i in range(len(voiced_region)):
        c = c and voiced_region[i]
    if c:
        return [(0,len(voiced_region))]
    
    start = 10000
    if voiced_region[0]:
        start = 0
    for i in range(1,T-1):
        if voiced_region[i-1] == False and voiced_region[i] == True:
            start = i
        if start < i and voiced_region[i] == True and voiced_region[i+1] == False:
            end = i
            
            if end - start > 5: 
                intervals.append((start,end))
    if voiced_region[T-1]:
        intervals.append((start,T-1))
    return intervals

def freq_from_crossings(sig, fs):
    """
    Estimate frequency by counting zero crossings
    """
    # Find all indices right before a rising-edge zero crossing
    indices = nonzero((sig[1:] >= 0) & (sig[:-1] < 0))[0]

    # Naive (Measures 1000.185 Hz for 1000 Hz, for instance)
    # crossings = indices

    # More accurate, using linear interpolation to find intersample
    # zero-crossings (Measures 1000.000129 Hz for 1000 Hz, for instance)
    crossings = [i - sig[i] / (sig[i+1] - sig[i]) for i in indices]
    
    # Some other interpolation based on neighboring points might be better.
    # Spline, cubic, whatever
    return fs / mean(diff(crossings))

def normalize(segment):
    minmax = np.max(segment) - np.min(segment)
    
    if minmax < 0.05:
        return np.clip(segment,-1,1)
    
    segment_copy = -1 + 2*(segment-np.min(segment))/minmax
    return segment_copy

def periodic_normalize(signal,freq):
    _signal = signal.copy()
    _signal = savgol_filter(_signal,9,2)
    valley = _signal[1] - _signal[0] < 0
    
    mpd = min((16000/freq)*3.5/5,90)
    __peaks = detect_peaks(_signal,mpd=mpd, edge=False, valley = valley)
    _peaks = detect_peaks(signal, mpd=mpd,edge=False, valley = valley)
    
    peaks = []
    if len(_peaks) > 0 and len(__peaks) > 0:
        for peak in _peaks:
            if np.min(np.abs(__peaks - peak)) < mpd/5:
                peaks.append(peak)
        for i in range(len(peaks)-1):
            if valley:
                _peak = peaks[i] + np.argmax(_signal[peaks[i]:peaks[i+1]])
            else:
                _peak = peaks[i] + np.argmin(_signal[peaks[i]:peaks[i+1]])

            if _peak!=peaks[i]:
                _signal[peaks[i]:_peak] = normalize(signal[peaks[i]:_peak])
            if _peak!=peaks[i+1]:
                _signal[_peak:peaks[i+1]] = normalize(signal[_peak:peaks[i+1]])

    return _signal

def get_zeros(signal):
    idxs = []
    for i in range(len(signal)-1):
        if signal[i] *signal[i+1] < 0:
            idxs.append(i)
    return idxs
def process(signal):
#     1. signal
    _signal = signal.copy()
    _signal = signal - np.mean(signal)
    stft = librosa.stft(np.asfortranarray(signal),512,128)
    intervals = find_voice_interval(mask(np.abs(stft)), threshold = 3)
    ed_before = 0
    for idx,(_st,_ed) in enumerate(intervals):
        st = _st 
        ed = _ed 
        freq = freq_from_crossings(_signal[st*128:ed*128], 16000)
        if math.isnan(freq):
            if idx == 0:
                freq = 106
            else:
                pass
        _signal[st*128:min(ed*128,len(_signal))] = periodic_normalize(_signal[st*128:min(ed*128,len(_signal))],freq)
        
        zero_idxs = get_zeros(_signal[st*128:ed*128]) ### 요거 하기
        if len(zero_idxs) > 0:
            _signal[ed_before:st*128 + zero_idxs[0]] = 0
            ed_before = st*128 + zero_idxs[-1]
    
    _signal[ed_before:] = 0
    
    
    return _signal