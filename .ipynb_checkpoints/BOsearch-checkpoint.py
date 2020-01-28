import numpy as np
import random
import csv
import os
import subprocess
from bayes_opt import BayesianOptimization
# from bayes_opt.logger import JSONLogger
# from bayes_opt.event import Events

train_stride = 16
valid_stride = 64
batch_size = 480
epoch = 50
n_frame = 64
global_iter = 0

f = open('BOsearch.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow(["batch_size", "train_stride", "valid_stride", "epoch", "weight_decay", "optimizer", "Lookahead_alpha", "Lookahead_k", "BCEDICE_ratio", "pos_weight", "loss_lambda", "loss_gamma", "lr", "exp_num", "n_frame","best_val"])
f.flush()

def train_validate(weight_decay_log, Lookahead_alpha,Lookahead_k_float,BCEDICE_ratio, pos_weight_log, loss_lambda_log, loss_gamma_log, lr_log):
    
    global train_stride
    global valid_stride
    global batch_size
    global epoch
    global n_frame
    global global_iter
    global wr
    global f
    
    weight_decay = 10**weight_decay_log
    optimizer = 'RAdam'
    Lookahead_k = int(np.round(Lookahead_k_float))
    pos_weight = int(10**pos_weight_log)
    loss_lambda = 10**loss_lambda_log
    loss_gamma = 10**loss_gamma_log
    lr = 10**lr_log
    exp_num = "TUNING/"+str(global_iter)
    
    command = "python -m torch.distributed.launch --nproc 4 train.py --batch_size {} --train_stride {} --valid_stride {} --epoch {} --weight_decay {} --optimizer {} --Lookahead_alpha {} --Lookahead_k {} --BCEDICE_ratio {} --pos_weight {} --loss_lambda {} --loss_gamma {} --lr {} --exp_num {} --n_frame {} --ddp --mixed".format(batch_size, train_stride, valid_stride, epoch, weight_decay, optimizer, Lookahead_alpha, Lookahead_k, BCEDICE_ratio, pos_weight, loss_lambda, loss_gamma, lr, exp_num, n_frame)
    
    x = subprocess.check_output(command.split(" "))
    best_val = float(x.decode("utf-8").split('\n')[-2])
    
    print("best val :",best_val)
    
    wr.writerow([batch_size, train_stride, valid_stride, epoch, weight_decay, optimizer, Lookahead_alpha, Lookahead_k, BCEDICE_ratio, pos_weight, loss_lambda, loss_gamma, lr, exp_num, n_frame,best_val])
    f.flush()
    
    global_iter +=1
    
    return -best_val

bayes_optimizer = BayesianOptimization(
    f=train_validate,
    pbounds={
        'weight_decay_log': (-3, -6),
        'Lookahead_alpha' : (0.3,0.7),
        'Lookahead_k_float' : (3.5,7.49),
        'BCEDICE_ratio' : (0,1),
        'pos_weight_log' : (0,1),
        'loss_lambda_log' : (-0.5,0.5),
        'loss_gamma_log' : (-0.5,0.5),
        'lr_log' : (-2,-4)
    },
    random_state=42,
    verbose=2
)

# logger = JSONLogger(path="./BO_logs.json")
# optimizer.subscribe(Events.OPTMIZATION_STEP, logger)
bayes_optimizer.maximize(init_points=12, n_iter=150, acq='ei', xi=0.01)

for i, res in enumerate(bayes_optimizer.res):
    print('Iteration {}: \n\t{}'.format(i, res))
print('Final result: ', bayes_optimizer.max)
