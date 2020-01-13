from utils.utils import seed_everything
import numpy as np
import random
import torch
# from SpecAug.sparse_image_warp_pytorch import sparse_image_warp

seed_everything(42)

def mix_db(x,y,db,torch_mode=False):
    if torch_mode:
        E_x = torch.mean(x**2)
        E_y = torch.mean(y**2)
    else:
        E_x = np.mean(x**2)
        E_y = np.mean(y**2)
    
    a = E_x/(E_y*(10**(db/10)))
    lam = 1/(1+a)
    
    return lam*x + (1-lam)*y

# def time_warp(spec, W=5):
#     num_rows = spec.shape[1] ##F
#     spec_len = spec.shape[2] ##T

#     y = num_rows // 2
#     horizontal_line_at_ctr = spec[0][y]
#     # assert len(horizontal_line_at_ctr) == spec_len

#     point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len-W)]
#     # assert isinstance(point_to_warp, torch.Tensor)

#     # Uniform distribution from (0,W) with chance to be up to W negative
#     dist_to_warp = random.randrange(-W, W)
    
#     src_pts = torch.tensor([[[y, point_to_warp]]])
#     dest_pts = torch.tensor([[[y, point_to_warp + dist_to_warp]]])
#     warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
#     return warped_spectro, dense_flows

def spec_masking(spec, F = 15, T = 10, num_masks = 1, prob = 0.7, replace_with_zero = True):
    def _freq_mask(spec, F=F, num_masks=num_masks, replace_with_zero=replace_with_zero):
#     cloned = spec.clone()
        _,num_mel_channels,_ = spec.shape
        
        for i in range(0, num_masks):        
            f = random.randrange(0, F)
            f_zero = random.randrange(0, num_mel_channels - f)
    
            # avoids randrange error if values are equal and range is empty
            if f_zero == f_zero + f:
                return spec
    
            mask_end = random.randrange(f_zero, f_zero + f) 
            if replace_with_zero:
                spec[:,f_zero:mask_end,:] = 0
            else:
                spec[:,f_zero:mask_end,:] = spec.mean()
        return spec

    def _time_mask(spec, T=T, num_masks=num_masks, replace_with_zero=replace_with_zero):
    #     cloned = spec.clone()
        _,_,len_spectro = spec.shape
      #  print("len",len_spectro)

        for i in range(0, num_masks):
            t = random.randrange(0, T)
            if len_spectro<=t:
                return spec
            t_zero = random.randrange(0, len_spectro - t)
            # avoids randrange error if values are equal and range is empty
            if t_zero == t_zero + t:
                return spec

            mask_end = random.randrange(t_zero, t_zero + t)
            if replace_with_zero:
                spec[:,:,t_zero:mask_end] = 0
            else:
                spec[:,:,t_zero:mask_end] = spec.mean()
        return spec
    
    if np.random.uniform()>prob:
        return spec
    else:
        spec = _freq_mask(spec)
        spec = _time_mask(spec)
    return spec

def add_whitenoise(x,torch_mode=False):
#     print(x,x.dtype)
    db = np.random.uniform(low=0,high=30)
    y = np.random.normal(size = x.shape)
    if torch_mode:
        y = torch.from_numpy(y).cuda()
    return mix_db(x,y,db,torch_mode)

def custom_stft_aug(n_frame = 64):
    def _custom_aug(x,normal_noise,musical_noise):
        db_1 = np.random.uniform(low=0,high=35)
        p_1 = np.random.uniform()
  
        if 0.1 < p_1 < 0.6:
            pi = random.randint(0,len(normal_noise)-1)
            pi2 = random.randint(0,len(normal_noise[pi])-n_frame-1)
            y = normal_noise[pi][:,pi2:pi2+n_frame]
            x = mix_db(x,y,db_1)
        elif 0.6 < p_1:
            pi = random.randint(0,len(musical_noise)-1)
            pi2 = random.randint(0,len(musical_noise[pi])-n_frame-1)
            y = musical_noise[pi][:,pi2:pi2+n_frame]
            x = mix_db(x,y,db_1)
    
        return x
    return _custom_aug
    