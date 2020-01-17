import numpy as np
import random
import csv
import os
import subprocess

N_iter = 300

train_stride = 16
valid_stride = 64

batch_size = 512
epoch = 50

n_frame = 64

f = open('search.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow(["batch_size", "train_stride", "valid_stride", "epoch", "weight_decay", "optimizer", "Lookahead_alpha", "Lookahead_k", "maskloss_type", "pos_weight", "loss_lambda", "loss_gamma", "lr", "exp_num", "n_frame","best_val"])
f.flush()
for iter in range(N_iter):
    weight_decay = 10**np.random.uniform(-3,-6)
#     optimizer = random.choice(['RMSProp','amsgrad','RAdam','RAdamW'])
    optimizer = random.choice(['RAdam'])
    Lookahead_alpha = np.random.uniform(0.4,0.7)
    Lookahead_k = random.choice([4,5,6,7])
    maskloss_type = random.choice(['BCE','DiceLoss','BCEDICE'])
    pos_weight = int(10**np.random.uniform(0,1))
    loss_lambda = 10**np.random.uniform(-0.5,0.5)
    loss_gamma = 10**np.random.uniform(-0.5,0.5)
    lr = 10**np.random.uniform(-2,-4)
    exp_num = "TUNING_"+str(iter)
    
    command = "python -m torch.distributed.launch --nproc 4 train.py --batch_size {} --train_stride {} --valid_stride {} --epoch {} --weight_decay {} --optimizer {} --Lookahead_alpha {} --Lookahead_k {} --maskloss_type {} --pos_weight {} --loss_lambda {} --loss_gamma {} --lr {} --exp_num {} --n_frame {} --ddp --mixed".format(batch_size, train_stride, valid_stride, epoch, weight_decay, optimizer, Lookahead_alpha, Lookahead_k, maskloss_type, pos_weight, loss_lambda, loss_gamma, lr, exp_num, n_frame)
    
    
    x = subprocess.check_output(command.split(" "))
    best_val = float(x.decode("utf-8").split('\n')[-2])
    
    print("best val :",best_val)
    
    wr.writerow([batch_size, train_stride, valid_stride, epoch, weight_decay, optimizer, Lookahead_alpha, Lookahead_k, maskloss_type, pos_weight, loss_lambda, loss_gamma, lr, exp_num, n_frame,best_val])
    f.flush()