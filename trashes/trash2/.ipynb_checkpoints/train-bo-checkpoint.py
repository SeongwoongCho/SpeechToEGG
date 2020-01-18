import csv
import time
import multiprocessing as mp
import argparse
import warnings
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
import torch.distributed as dist

from radam import RAdam,Lookahead
from utils import *
from dataloader import *
from models import Resv2Unet,ULSTM, UEDAttention

import apex
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from bayes_opt import BayesianOptimization

def train_and_validate(init_lr_log,weight_decay_log,max_norm_log,beta1_log,beta2_log):
    seed_everything(42)
    
    torch.cuda.empty_cache()
    model = Resv2Unet(5,64,15,5)
#     model = apex.parallel.convert_syncbn_model(model)
    model.cuda()
    
    init_lr = 10**init_lr_log
    weight_decay = 10**weight_decay_log
    max_norm = 10**max_norm_log
    beta1 = 1-10**beta1_log
    beta2 = 1-10**beta2_log
    
    optimizer = RAdam(model.parameters(), lr=init_lr, betas=(beta1, beta2), eps=1e-8, weight_decay=weight_decay)
    model,optimizer = amp.initialize(model,optimizer,opt_level = 'O1')
    model = torch.nn.DataParallel(model)
    
    for epoch in range(1):
        train_loss = 0.
        optimizer.zero_grad()
        model.train()

        for idx,(_x,_y) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            x_train,y_train = _x.cuda(),_y.cuda()
            pred = model(x_train)
            loss = criterion(pred,y_train)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()    
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm)
    #         loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(train_loader)

    #     torch.cuda.synchronize()
        val_loss = 0.
        model.eval()
        for idx,(_x,_y) in enumerate(tqdm(valid_loader)):
            x_val,y_val = _x.cuda(),_y.cuda()
            with torch.no_grad():
                pred = model(x_val)
                loss = criterion(pred,y_val)
            val_loss += loss.item()/len(valid_loader)
    return -val_loss

seed_everything(42)
###Hyper parameters

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default = 8192)
parser.add_argument('--n_frame',type=int,default=192)
parser.add_argument('--window_ratio', type = float, default = 1.25)
parser.add_argument('--step_ratio',type = float, default = 4)
parser.add_argument('--top_db',type=int,default = 20)
parser.add_argument('--test',action='store_true')

args = parser.parse_args()

##train ? or test? 
is_test = args.test

##training parameters
batch_size = args.batch_size

##data preprocessing parameters##
n_frame = args.n_frame
window_ratio = args.window_ratio
step_ratio = args.step_ratio
top_db = args.top_db

assert (window_ratio-1)*step_ratio >=1, "window ratio * step ratio should be >=1"
window = int(n_frame*window_ratio)
step = int(n_frame/step_ratio)

print("[*] load data ...")

st = time.time()
# train_X,train_y,val_X,val_y = load_datas(n_frame,window,step,top_db,is_test,hvd.local_rank())
train_X,train_y,val_X,val_y = load_datas_path(is_test)
normal_noise,musical_noise = load_noise(n_frame,top_db,is_test)

print(len(train_X))
print(len(val_X))

train_dataset = Dataset(train_X,train_y,normal_noise,musical_noise,n_frame = n_frame, is_train = True, aug = custom_aug_v2(n_frame))
valid_dataset = Dataset(val_X,val_y,n_frame = n_frame, is_train = False)
train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               num_workers=mp.cpu_count(),
                               pin_memory=True)

valid_loader = data.DataLoader(dataset=valid_dataset,
                               batch_size=batch_size,
                               num_workers=mp.cpu_count(),
                               pin_memory=True)
print("Load duration : {}".format(time.time()-st))
print("[!] load data end")

criterion = CosineDistanceLoss()
bayes_optimizer = BayesianOptimization(
    f=train_and_validate,
    pbounds={
        'init_lr_log': (-5, -1),
        'weight_decay_log': (-6, -1),
        'max_norm_log':(-1,2),
        'beta1_log':(-2,-0.7),
        'beta2_log':(-4,-2)
    },
    random_state=42,
    verbose=2
)

bayes_optimizer.minimize(init_points=6, n_iter=27, acq='ei', xi=0.01)

for i, res in enumerate(bayes_optimizer.res):
    print('Iteration {}: \n\t{}'.format(i, res))
print('Final result: ', bayes_optimizer.max)