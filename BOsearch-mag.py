import numpy as np
import random
import csv
import os
import subprocess
from bayes_opt import BayesianOptimization

global_iter = 0

f = open('BOsearch-mag.csv', 'a', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow(["weight_decay","lr","momentum","best_val"])
f.flush()

def train_validate(weight_decay_log,lr_log,momentum_log):
    global global_iter
    global wr
    global f
    
    weight_decay = 10**weight_decay_log
    lr = 10**lr_log
    momentum = 1 - 2**momentum_log 
    exp_num = "TUNING/"+str(global_iter)
    
    command = "python -m torch.distributed.launch --nproc 6 train-mag.py --batch_size 1200 --train_stride 2048 --valid_stride 4096 --n_sample 4096 --window_length 512 --hop_length 128 --window hann --epoch 100 --weight_decay {} --lr {} --patience 100 --momentum {} --exp_num {} --ddp --mixed".format(weight_decay,lr,momentum,exp_num)
    
    x = subprocess.check_output(command.split(" "))
    best_val = float(x.decode("utf-8").split('\n')[-2])
    
    print("best val :",best_val)
    
    wr.writerow([weight_decay,lr,momentum,best_val])
    f.flush()
    
    global_iter +=1
    
    return -best_val

bayes_optimizer = BayesianOptimization(
    f=train_validate,
    pbounds={
        'weight_decay_log': (-5, -1),
        'lr_log' : (-5,-1),
        'momentum_log' : (-5,-1)
    },
    random_state=42,
    verbose=2
)

bayes_optimizer.maximize(init_points=5, n_iter=15, acq='ei', xi=0.01)

for i, res in enumerate(bayes_optimizer.res):
    print('Iteration {}: \n\t{}'.format(i, res))
print('Final result: ', bayes_optimizer.max)