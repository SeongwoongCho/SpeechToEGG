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
from models import Resv2Unet, ULSTM, UEDAttention

import apex
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= 4
    return rt

seed_everything(42)
###Hyper parameters

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default = 8192)
parser.add_argument('--epoch',type=int, default = 100)
parser.add_argument('--max_norm',type=float,default=-1)
parser.add_argument('--teacher_force_ratio',type=float,default=0)
parser.add_argument('--beta1',type=float,default = 0.9)
parser.add_argument('--beta2',type=float,default = 0.999)
parser.add_argument('--weight_decay',type=float,default = 1e-5)
parser.add_argument('--lr', type=float, default = 1e-2)
parser.add_argument('--lr_step', type=int, default = 65)
parser.add_argument('--exp_num',type=str,default=0)
parser.add_argument('--n_frame',type=int,default=192)
parser.add_argument('--window_ratio', type = float, default = 1.25)
parser.add_argument('--step_ratio',type = float, default = 4)
parser.add_argument('--top_db',type=int,default = 20)
parser.add_argument('--scheduler_gamma',type=float,default=0.1)
parser.add_argument('--test',action='store_true')
parser.add_argument('--local_rank',type=int,default=0)
parser.add_argument('--mode',type=str,default = 'ddp')

args = parser.parse_args()

##train ? or test? 
is_test = args.test
mode = args.mode

##training parameters
n_epoch = args.epoch if not is_test else 1
batch_size = args.batch_size//4 if mode == 'ddp' else args.batch_size
max_norm = args.max_norm
teacher_force_ratio = args.teacher_force_ratio
beta1 = args.beta1
beta2 = args.beta2
weight_decay = args.weight_decay
##model parameters

##data preprocessing parameters##
n_frame = args.n_frame
window_ratio = args.window_ratio
step_ratio = args.step_ratio
top_db = args.top_db

assert (window_ratio-1)*step_ratio >=1, "window ratio * step ratio should be >=1"
window = int(n_frame*window_ratio)
step = int(n_frame/step_ratio)

##optimizer parameters##
learning_rate = args.lr

##scheduler parameters## 
step_size = args.lr_step
scheduler_gamma = args.scheduler_gamma

##saving path
save_path = './models/exp{}/'.format(args.exp_num)
os.makedirs(save_path,exist_ok=True)

##Distributed Data Parallel
if mode == 'ddp':
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    args.world_size = torch.distributed.get_world_size()
    
verbose = 1 if args.local_rank ==0 else 0

if not verbose:
    warnings.filterwarnings(action='ignore')
logging = print_verbose(verbose)
logging("[*] load data ...")

st = time.time()
# train_X,train_y,val_X,val_y = load_datas(n_frame,window,step,top_db,is_test,hvd.local_rank())
train_X,train_y,val_X,val_y = load_datas_path(is_test)
normal_noise,musical_noise = load_noise(n_frame,top_db,is_test)

logging(len(train_X))
logging(len(val_X))
logging(batch_size)
train_dataset = Dataset(train_X,train_y,normal_noise,musical_noise,n_frame = n_frame, is_train = True, aug = custom_aug_v2(n_frame))
valid_dataset = Dataset(val_X,val_y,n_frame = n_frame, is_train = False)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=4, rank=args.local_rank)
valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas=4, rank=args.local_rank)
train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               num_workers=mp.cpu_count()//4 if mode == 'ddp' else mp.cpu_count(),
                               sampler = train_sampler if mode == 'ddp' else None,
                               pin_memory=True)
valid_loader = data.DataLoader(dataset=valid_dataset,
                               batch_size=batch_size,
                               num_workers=mp.cpu_count()//4 if mode == 'ddp' else mp.cpu_count(),
                               sampler = valid_sampler if mode == 'ddp' else None,
                               pin_memory=True)
logging("Load duration : {}".format(time.time()-st))
logging("[!] load data end")

criterion = CosineDistanceLoss()

model = UEDAttention(nlayers = 5, nefilters = 64,filter_size = 15,merge_filter_size = 5,hidden_size = 30, num_layers = 1)
# model = ULSTM(nlayers = 5, nefilters = 64,filter_size = 15,merge_filter_size = 5,hidden_size = 128, num_layers = 1)
# model = Resv2Unet(5,64,15,5)
if mode == 'ddp':
    model = apex.parallel.convert_syncbn_model(model)

## pretrained model
pretrained_state_dict = torch.load('./models/exp19_continue/best.pth')
model_state_dict = model.state_dict()

# print(pretrained_state_dict.keys())
# print(model_state_dict.keys())
for key in pretrained_state_dict:
    key_model = key.replace("module","UBlock")
    key_model_2 = ".".join(key.split(".")[1:])
    if key in model_state_dict:
        logging(key)
        model_state_dict[key] = pretrained_state_dict[key]        
    elif key_model in model_state_dict:
        logging(key_model)
        model_state_dict[key_model] = pretrained_state_dict[key]
    elif key_model_2 in model_state_dict:
        logging(key_model_2)
        model_state_dict[key_model_2] = pretrained_state_dict[key]
model.load_state_dict(model_state_dict)
model.cuda()

optimizer = RAdam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=1e-8, weight_decay=weight_decay)
optimizer = Lookahead(optimizer,alpha=0.5,k=6)
model,optimizer = amp.initialize(model,optimizer,opt_level = 'O1')
scheduler = StepLR(optimizer,step_size=step_size,gamma = scheduler_gamma)

if mode == 'ddp':
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.local_rank],
                                                      output_device=0)
#     model = torch.nn.parallel.DistributedDataParallel(model)
else:
    model = torch.nn.DataParallel(model)

logging("[*] training ...")

if verbose and not is_test:
    log = open(os.path.join(save_path,'log.csv'), 'a+', encoding='utf-8', newline='')
    log_writer = csv.writer(log)
    best_val = np.inf

for param in model.module.UBlock.parameters():
    param.requires_grad = False
    
for epoch in range(n_epoch):
    if epoch == 5 or is_test:
        for param in model.module.UBlock.parameters():
            param.requires_grad = True
#         learning rate를 decay시킬 때 CNN layer를 unfreeze 해준다.
            
    st = time.time()
    train_sampler.set_epoch(epoch)
    train_loss = 0.
    optimizer.zero_grad()
    model.train()
    
    for idx,(_x,_y) in enumerate(tqdm(train_loader,disable=(verbose==0))):
        optimizer.zero_grad()
        x_train,y_train = _x.cuda(),_y.cuda()
        pred = model(x_train)
        loss = criterion(pred,y_train)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm)
        optimizer.step()
        
        if mode == 'ddp':
            reduced_loss = reduce_tensor(loss.data)
            train_loss += reduced_loss.item() / len(train_loader)
        else:
            train_loss += loss.item() / len(train_loader)
    
#     torch.cuda.synchronize()
    val_loss = 0.
    model.eval()
    for idx,(_x,_y) in enumerate(tqdm(valid_loader,disable=(verbose==0))):
        x_val,y_val = _x.cuda(),_y.cuda()
        with torch.no_grad():
            pred = model(x_val)
            loss = criterion(pred,y_val)
        
        if mode == 'ddp':
            reduced_loss = reduce_tensor(loss.data)
            val_loss += reduced_loss.item()/len(valid_loader)
        else:
            val_loss += loss.item()/len(valid_loader)
            
    scheduler.step()
    
    if verbose and not is_test:
        if val_loss < best_val:
            torch.save(model.state_dict(), os.path.join(save_path,'best.pth'))
            best_val = val_loss
        log_writer.writerow([epoch,train_loss,val_loss])
        log.flush()
    logging("Epoch [%d]/[%d] train_loss %.6f valid_loss %.6f duration : %.4f"%
        (epoch,n_epoch,train_loss,val_loss,time.time()-st))
    
if verbose and not is_test:
    log.close()
logging("[!] training end")