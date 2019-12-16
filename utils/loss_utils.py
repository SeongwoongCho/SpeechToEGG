import os
import torch
import numpy as np

def frobenius_norm(x):
    """
    calculate frobenius_norm across F,T
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
    return loss

def LM_loss(stft_magnitude_pred,stft_magnitude_target):
    """
    calculation of Log-scale STFT-magnitude loss
    
    stft_magnitude_pred   : B,F,T Tensor
    stft_magnitude_target : B,F,T Tensor
    
    return LM_loss : constant
    """
    epsilon = 1e-4
    loss = torch.abs(stft_magnitude_pred-stft_magnitude_target) ## log 값이 씌어져 있는 상태임
    loss = torch.mean(loss,dim=(0,1,2))
        
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

    
    return loss   

def spectral_loss(coeff = [1,5,5,1]):
    def _spectral_loss(stft_pred,stft_target,coeff=coeff):
        stft_magnitude_pred = stft_pred[:,0,:,:]
        stft_magnitude_target = stft_target[:,0,:,:]
        stft_phase_pred = stft_pred[:,1,:,:]
        stft_phase_target = stft_target[:,1,:,:]

        coeff = coeff[:3]
        coeff = coeff/np.sum(coeff)
        
        lm_loss = LM_loss(stft_magnitude_pred,stft_magnitude_target)
        if_loss = IF_loss(stft_phase_pred,stft_phase_target)
        loss = coeff[1]*lm_loss+coeff[2]*if_loss
        return loss
    return _spectral_loss

def CosineDistanceLoss():
    def f(pred,true):
        return torch.mean(torch.acos(F.cosine_similarity(pred,true,dim=1,eps = 1e-4)),dim=0)
    return f