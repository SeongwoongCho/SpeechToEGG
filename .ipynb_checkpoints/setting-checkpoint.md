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
L2Loss

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
exp 0 augmentation(but, padding -1 with random l,r)+ freq masking
L1 Loss