# exp 0

normalize(mel(log(S)))

lr : 3e-4
batch_size : 128 
n_frame : 64
n_epoch : 1000
lr_step : 400
gamma : 0.1
beta1 : 0.9
beta2 : 0.999
weight_decay : 1e-5

==model parameter == 
drop_rate = 0.2
bn_size = 4
k,l = 12,3

radam + lookahead(k=6,alpha=.5)
Noise augmentation
pixel L2Loss

# exp 1

normalize(log(mel(S)))

lr : 3e-4
batch_size : 128 
n_frame : 64
n_epoch : 1000
lr_step : 400
gamma : 0.1
beta1 : 0.9
beta2 : 0.999
weight_decay : 1e-5

==model parameter == 
drop_rate = 0.2
bn_size = 4
k,l = 12,3

radam + lookahead(k=6,alpha=.5)
exp 0 augmentation(but, padding -1 with random l,r) +  Time/freq masking(F = 10, T = 5, num_masks = 2, prob = 0.75) before to_mel
pixel L1 Loss

# exp 2

normalize(log(mel(S)))

lr : 3e-4
batch_size : 128 
n_frame : 64
n_epoch : 1000
lr_step : 400
gamma : 0.1
beta1 : 0.9
beta2 : 0.999
weight_decay : 1e-5

==model parameter == 
drop_rate = 0.2
bn_size = 4
k,l = 12,3

radam + lookahead(k=6,alpha=.5)
exp 0 augmentation(but, padding -1 with random l,r)+ Time/freq masking(F = 10, T = 5, num_masks = 2, prob = 0.75) before to_mel
pixel L2 loss(0.5) + VGG16 loss(0.25,0.25)

# exp 3 

normalize(log(mel(S)))

lr : 3e-4
batch_size : 108
n_frame : 64
n_epoch : 1000
lr_step : 400
gamma : 0.1
beta1 : 0.9
beta2 : 0.999
weight_decay : 1e-5

==model parameter == 
drop_rate = 0.25
bn_size = 4
k1,l1 = 10,3
k2,l2 = 14,4

radam + lookahead(k=6,alpha=.5)
exp 0 augmentation(but, padding -1 with random l,r)+ Time/freq masking(F = 10, T = 5, num_masks = 2, prob = 0.75) before to_mel
pixel L2 loss(0.5) + VGG16 loss(0.25,0.25)

# exp 4 : add Time warp, before mel to after mel from exp 3

normalize(log(mel(S)))

lr : 3e-4
batch_size : 108
n_frame : 64
n_epoch : 1000
lr_step : 400
gamma : 0.1
beta1 : 0.9
beta2 : 0.999
weight_decay : 1e-5

==model parameter == 
drop_rate = 0.25
bn_size = 4
k1,l1 = 10,3
k2,l2 = 14,4

radam + lookahead(k=6,alpha=.5)
exp 0 augmentation(but, padding -1 with random l,r) + SpecAug(W=10,F = 10, T = 5, num_masks = 2, prob = 1) after mel
pixel L2 loss(0.5) + VGG16 loss(0.25,0.25)

# exp 5 : remove Timewarp, add dtime,dfreq loss, replace VGG 16 to 19 from exp 4 

normalize(log(mel(S)))

lr : 3e-4
batch_size : 96
n_frame : 64
n_epoch : 1000
lr_step : 400
gamma : 0.1
beta1 : 0.9
beta2 : 0.999
weight_decay : 1e-5

==model parameter == 
drop_rate = 0.25
bn_size = 4
k1,l1 = 10,3
k2,l2 = 14,4

radam + lookahead(k=6,alpha=.5)
exp 0 augmentation(but, padding -1 with random l,r) +  Time/freq masking(F = 10, T = 5, num_masks = 2, prob = 1) after mel
pixel L2 loss(0.2) + time_derivative L2 loss(0.2) + freq_derivative L2 loss(0.2) + VGG19 loss(0.2,0.2)

# exp 6 : remove VGG loss from exp 5

normalize(log(mel(S)))

lr : 3e-4
batch_size : 96
n_frame : 64
n_epoch : 1000
lr_step : 400
gamma : 0.1
beta1 : 0.9
beta2 : 0.999
weight_decay : 1e-5

==model parameter == 
drop_rate = 0.25
bn_size = 4
k1,l1 = 10,3
k2,l2 = 14,4

radam + lookahead(k=6,alpha=.5)
exp 0 augmentation(but, padding -1 with random l,r) +  Time/freq masking(F = 10, T = 5, num_masks = 2, prob = 1) after mel
pixel L2 loss(0.34) + time_derivative L2 loss(0.33) + freq_derivative L2 loss(0.33)

# exp 7 ~ exp 10 학습 잘못했었음.(GAN implementation 실수)
#####
# exp 7 : Change model to CGAN(pix2pix)  

normalize(log(mel(S)))

lr : 3e-4
batch_size : 96
n_frame : 64
n_epoch : 1000
lr_step : 400
gamma : 0.1
beta1 : 0.9
beta2 : 0.999
weight_decay : 1e-5

==model parameter == 
Generator : MMDenseNet
    drop_rate = 0.25
    bn_size = 4
    k1,l1 = 10,3
    k2,l2 = 14,4
    
Discriminator : pixelGAN_discriminator (remove batch norm 2d becuase of some issue)
    no hyper parameter

radam + lookahead(k=6,alpha=.5)
exp 0 augmentation(but, padding -1 with random l,r) +  Time/freq masking(F = 10, T = 5, num_masks = 2, prob = 1) after mel

λ = 0.5
loss = cGAN loss + λ*recon_loss

recon loss : pixel L2 loss(0.34) + time_derivative L2 loss(0.33) + freq_derivative L2 loss(0.33)
cGAN loss : BCE(pred,0) + BCE(true,1)

# exp 8 : Change λ to 1  

normalize(log(mel(S)))

lr : 3e-4
batch_size : 96
n_frame : 64
n_epoch : 1500
lr_step : 600
gamma : 0.1
beta1 : 0.9
beta2 : 0.999
weight_decay : 1e-5

==model parameter == 
Generator : MMDenseNet
    drop_rate = 0.25
    bn_size = 4
    k1,l1 = 10,3
    k2,l2 = 14,4
    
Discriminator : pixelGAN_discriminator (remove batch norm 2d becuase of some issue)
    no hyper parameter

radam + lookahead(k=6,alpha=.5)
exp 0 augmentation(but, padding -1 with random l,r) +  Time/freq masking(F = 10, T = 5, num_masks = 2, prob = 1) after mel

λ = 1
loss = cGAN loss + λ*recon_loss

recon loss : pixel L2 loss(0.34) + time_derivative L2 loss(0.33) + freq_derivative L2 loss(0.33)
cGAN loss : (BCE(pred,0) + BCE(true,1))*0.5

# exp 9 : pixelGAN to PatchGAN

normalize(log(mel(S)))

lr : 3e-4
batch_size : 96
n_frame : 64
n_epoch : 1500
lr_step : 600
gamma : 0.1
beta1 : 0.9
beta2 : 0.999
weight_decay : 1e-5

==model parameter == 
Generator : MMDenseNet
    drop_rate = 0.25
    bn_size = 4
    k1,l1 = 10,3
    k2,l2 = 14,4
    
Discriminator : patchGAN_discriminator (remove batch norm 2d becuase of some issue)
    no hyper parameter

radam + lookahead(k=6,alpha=.5)
exp 0 augmentation(but, padding -1 with random l,r) +  Time/freq masking(F = 10, T = 5, num_masks = 2, prob = 1) after mel

λ = 1
loss = cGAN loss + λ*recon_loss

recon loss : pixel L2 loss(0.34) + time_derivative L2 loss(0.33) + freq_derivative L2 loss(0.33)
cGAN loss : (BCE(pred,0) + BCE(true,1))*0.5


# exp 10 : patchGAN to custom discriminator

normalize(log(mel(S)))

lr : 3e-4
batch_size : 96
n_frame : 64
n_epoch : 1500
lr_step : 600
gamma : 0.1
beta1 : 0.9
beta2 : 0.999
weight_decay : 1e-5

==model parameter == 
Generator : MMDenseNet
    drop_rate = 0.25
    bn_size = 4
    k1,l1 = 10,3
    k2,l2 = 14,4
    
Discriminator : customGAN_discriminator (remove batch norm 2d becuase of some issue)
    no hyper parameter

radam + lookahead(k=6,alpha=.5)
exp 0 augmentation(but, padding -1 with random l,r) +  Time/freq masking(F = 10, T = 5, num_masks = 2, prob = 1) after mel

λ = 0.5
gen loss = cgan loss + λ*recon_loss

recon loss : pixel L2 loss(0.34) + time_derivative L2 loss(0.33) + freq_derivative L2 loss(0.33)
cgan loss : BCE(pred,1)

disc loss = (BCE(pred,0) + BCE(true,1))*0.5

#####

# exp 11 : L1 loss, remove time_derivative loss

normalize(log(mel(S)))

lr : 3e-4
batch_size : 96
n_frame : 64
n_epoch : 1000
lr_step : 400
gamma : 0.1
beta1 : 0.9
beta2 : 0.999
weight_decay : 1e-5

==model parameter == 
Generator : MMDenseNet
    drop_rate = 0.25
    bn_size = 4
    k1,l1 = 10,3
    k2,l2 = 14,4
    
Discriminator : customGAN_discriminator (remove batch norm 2d becuase of some issue)
    no hyper parameter

radam + lookahead(k=6,alpha=.5)
exp 0 augmentation(but, padding -1 with random l,r) +  Time/freq masking(F = 10, T = 5, num_masks = 2, prob = 1) after mel

λ = 0.5
gen loss = cgan loss + λ*recon_loss

recon loss : pixel L1 loss(0.5) + freq_derivative L1 loss(0.5)
cgan loss : BCE(pred,1)

disc loss = (BCE(pred,0) + BCE(true,1))*0.5

# exp 12 : Change normalization!!!! (model tanh -> sigmoid) disriminator training policy, change lambda ==> overfitting

-normalize(log(mel(S)))

lr : 3e-4
batch_size : 96
n_frame : 64
n_epoch : 1000
lr_step : 400
gamma : 0.1
beta1 : 0.9
beta2 : 0.999
weight_decay : 1e-5

==model parameter == 
Generator : MMDenseNet
    drop_rate = 0.25
    bn_size = 4
    k1,l1 = 10,3
    k2,l2 = 14,4
    
Discriminator : customGAN_discriminator (remove batch norm 2d becuase of some issue)
    no hyper parameter

policy : if val_loss < 0.4 stop training, if val_loss > 0.5 restart training

radam + lookahead(k=6,alpha=.5)
exp 0 augmentation(but, padding -1 with random l,r) +  Time/freq masking(F = 10, T = 5, num_masks = 2, prob = 1) after mel

λ = 10
gen loss = cgan loss + λ*recon_loss

recon loss : pixel L1 loss(0.5) + freq_derivative L1 loss(0.5)
cgan loss : BCE(pred,1)

disc loss = (BCE(pred,0) + BCE(true,1))*0.5

# exp 13 : add noise/label smoothing/TTUR/Spectral normlization

-normalize(log(mel(S)))
label smoothing
discriminator noise

lr : 3e-4
batch_size : 96
n_frame : 64
n_epoch : 1000
lr_step : 400
gamma : 0.1
beta1 : 0.9
beta2 : 0.999
weight_decay : 1e-5
TTUR : 3

discriminator_whitenoise_sigma = 0.05

==model parameter == 
Generator : MMDenseNet
    drop_rate = 0.25
    bn_size = 4
    k1,l1 = 10,3
    k2,l2 = 14,4
    
Discriminator : customGAN_discriminator (spectral normalization)
    no hyper parameter

policy : if val_loss < 0.4 stop training, if val_loss > 0.5 restart training

radam + lookahead(k=6,alpha=.5)
exp 0 augmentation(but, padding -1 with random l,r) +  Time/freq masking(F = 10, T = 5, num_masks = 2, prob = 1) after mel

λ = 10

gen loss = cgan loss + λ*recon_loss

recon loss : pixel L1 loss(0.5) + freq_derivative L1 loss(0.5)
cgan loss : BCE(pred,1)

disc loss = (BCE(pred,0) + BCE(true,1))*0.5

# exp 14 : λ(10 -> 100)

-normalize(log(mel(S)))
label smoothing
discriminator noise

lr : 3e-4
batch_size : 96
n_frame : 64
n_epoch : 1000
lr_step : 400
gamma : 0.1
beta1 : 0.9
beta2 : 0.999
weight_decay : 1e-5
TTUR : 3

discriminator_whitenoise_sigma = 0.05

==model parameter == 
Generator : MMDenseNet
    drop_rate = 0.25
    bn_size = 4
    k1,l1 = 10,3
    k2,l2 = 14,4
    
Discriminator : customGAN_discriminator (spectral normalization)
    no hyper parameter

policy : if val_loss < 0.4 stop training, if val_loss > 0.5 restart training

radam + lookahead(k=6,alpha=.5)
exp 0 augmentation(but, padding -1 with random l,r) +  Time/freq masking(F = 10, T = 5, num_masks = 2, prob = 1) after mel

λ = 100

gen loss = cgan loss + λ*recon_loss

recon loss : pixel L1 loss(0.5) + freq_derivative L1 loss(0.5)
cgan loss : BCE(pred,1)

disc loss = (BCE(pred,0) + BCE(true,1))*0.5

# exp 15 : CBAM attention, modify policy

-normalize(log(mel(S)))
label smoothing
discriminator noise

lr : 3e-4
batch_size : 96
n_frame : 64
n_epoch : 1000
lr_step : 400
gamma : 0.1
beta1 : 0.9
beta2 : 0.999
weight_decay :1e-5
TTUR : 3

discriminator_whitenoise_sigma = 0.05

==model parameter == 
Generator : MMDenseNet
    drop_rate = 0.25
    bn_size = 4
    k1,l1 = 10,3
    k2,l2 = 14,4
    CBAM attention
    
Discriminator : customGAN_discriminator (spectral normalization)
    no hyper parameter

policy : if val_loss < 0.4 stop training, if val_loss > 0.5 restart training

radam + lookahead(k=6,alpha=.5)
exp 0 augmentation(but, padding -1 with random l,r) +  Time/freq masking(F = 10, T = 5, num_masks = 2, prob = 1) after mel

λ = 10

gen loss = cgan loss + λ*recon_loss

recon loss : pixel L1 loss(0.5) + freq_derivative L1 loss(0.5)
cgan loss : BCE(pred,1)

disc loss = (BCE(pred,0) + BCE(true,1))*0.5

# exp 16 : modify normalization --> change last activation sigmoid to Relu, λ = 3

-log(clip(mel(S))) 

label smoothing
discriminator noise

lr : 3e-4
batch_size : 96
n_frame : 64
n_epoch : 1000
lr_step : 400
gamma : 0.1
beta1 : 0.9
beta2 : 0.999
weight_decay :1e-5
TTUR : 3

discriminator_whitenoise_sigma = 0.05

==model parameter == 
Generator : MMDenseNet
    drop_rate = 0.25
    bn_size = 4
    k1,l1 = 10,3
    k2,l2 = 14,4
    CBAM attention
    
Discriminator : customGAN_discriminator (spectral normalization)
    no hyper parameter

policy : if val_loss < 0.4 stop training, if val_loss > 0.5 restart training

radam + lookahead(k=6,alpha=.5)
exp 0 augmentation(but, padding -1 with random l,r) +  Time/freq masking(F = 10, T = 5, num_masks = 2, prob = 1) after mel

λ = 3

gen loss = cgan loss + λ*recon_loss

recon loss : pixel L1 loss(0.5) + freq_derivative L1 loss(0.5)
cgan loss : BCE(pred,1)

disc loss = (BCE(pred,0) + BCE(true,1))*0.5

###### 위의 실험들은 augmentation이 전혀 진행되고 있지 않았음. return 해주지 않았었음.... ㅅㅂ ㅅㅂ 아 족같다! 

# exp 17 : No GAN, modify masking parameter, modify STFT data.., n_frame -> 128

-log(clip(mel(S))) 

lr : 3e-4
batch_size : 44
n_frame : 128
n_epoch : 600
lr_step : 250
gamma : 0.1
beta1 : 0.9
beta2 : 0.999
weight_decay :1e-5

==model parameter == 
Generator : MMDenseNet
    drop_rate = 0.25
    bn_size = 4
    k1,l1 = 10,3
    k2,l2 = 14,4
    CBAM attention

radam + lookahead(k=6,alpha=.5)
exp 0 augmentation(but, padding -1 with random l,r) +  Time/freq masking(F = 1, T = 1, num_masks = 10, prob = 1) after mel

recon loss : pixel L1 loss(0.5) + freq_derivative L1 loss(0.5)
