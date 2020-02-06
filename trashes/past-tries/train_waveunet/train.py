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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
import torch.distributed as dist

from train_unet.utils.radam import RAdam,Lookahead
from utils import *
from dataloader import *
from models import Resv2Unet
from tensorboardX import SummaryWriter

import apex
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

seed_everything(42)
###Hyper parameters

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default = 8192)
parser.add_argument('--epoch',type=int, default = 100)
parser.add_argument('--weight_decay',type=float,default = 1e-5)
parser.add_argument('--lr', type=float, default = 1e-2)
parser.add_argument('--patience', type=int, default = 65)
parser.add_argument('--exp_num',type=str,default=0)
parser.add_argument('--n_frame',type=int,default=1024)
parser.add_argument('--train_stride',type=int,default=256)
parser.add_argument('--valid_stride',type=int,default=1024)
parser.add_argument('--scheduler_gamma',type=float,default=0.1)
parser.add_argument('--test',action='store_true')
parser.add_argument('--local_rank',type=int,default=0)
parser.add_argument('--ddp',action='store_true')

args = parser.parse_args()

##train ? or test? 
is_test = args.test
ddp = args.ddp

##training parameters
n_epoch = args.epoch
batch_size = args.batch_size//4 if ddp else args.batch_size
weight_decay = args.weight_decay

##data preprocessing parameters
n_frame = args.n_frame
train_stride = args.train_stride
valid_stride = args.valid_stride

##optimizer parameters##
learning_rate = args.lr
patience = args.patience
scheduler_gamma = args.scheduler_gamma

##saving path
save_path = './models/{}/'.format(args.exp_num)
os.makedirs(save_path,exist_ok=True)

##Distributed Data Parallel
if ddp:
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
train = np.load('../eggdata/TrainData/train_processing.npy',mmap_mode='r')
val = np.load('../eggdata/TrainData/valid_processing.npy',mmap_mode='r')

logging(train.shape)
logging(val.shape)
logging(batch_size)

train_dataset = Dataset(train,n_frame = n_frame,stride = train_stride, is_train=True)
valid_dataset = Dataset(val,n_frame = n_frame, stride =valid_stride, is_train=False)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=4, rank=args.local_rank)
valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas=4, rank=args.local_rank)

train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               num_workers=mp.cpu_count()//4 if ddp else mp.cpu_count(),
                               sampler = train_sampler if ddp else None,
                               shuffle = True if not ddp else False,
                               pin_memory=True)
valid_loader = data.DataLoader(dataset=valid_dataset,
                               batch_size=batch_size,
                               num_workers=mp.cpu_count()//4 if ddp else mp.cpu_count(),
                               sampler = valid_sampler if ddp else None,
                               pin_memory=True)

logging("Load duration : {}".format(time.time()-st))
logging("[!] load data end")

criterion = CosineDistanceLoss()
MSE = nn.MSELoss()
model = Resv2Unet(6,64,15,5)
if ddp:
    model = apex.parallel.convert_syncbn_model(model)
model.cuda()

optimizer = RAdam(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=weight_decay)
optimizer = Lookahead(optimizer,alpha=0.5,k=6)
model,optimizer = amp.initialize(model,optimizer,opt_level = 'O1')
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_gamma, patience= patience, verbose=True)

if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.local_rank],
                                                      output_device=0)
else:
    model = torch.nn.DataParallel(model)

logging("[*] training ...")

if verbose and not is_test:
    best_val = np.inf
    writer = SummaryWriter('../eggdata/log/%s/'%args.exp_num)

for epoch in range(n_epoch):
    st = time.time()
    train_sampler.set_epoch(epoch)
    valid_sampler.set_epoch(epoch)
    
    train_loss = 0.
    train_signal_loss = 0.
    train_mse_loss = 0.
    optimizer.zero_grad()
    model.train()
    
    for idx,(_x,_y) in enumerate(tqdm(train_loader,disable=(verbose==0))):
        optimizer.zero_grad()
        x_train,y_train = _x.cuda(),_y.cuda()
        pred = model(x_train)
        signal_loss = criterion(pred,y_train)
        MSE_loss = MSE(pred,y_train)
        
        loss = 0.6*signal_loss + 0.4*MSE_loss
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        
        optimizer.step()
        train_loss += dynamic_loss(loss,ddp)/len(train_loader)
        train_signal_loss += dynamic_loss(signal_loss,ddp)/len(train_loader)
        train_mse_loss += dynamic_loss(MSE_loss,ddp)/len(train_loader)
    
    
    val_loss = 0.
    val_signal_loss = 0.
    val_mse_loss = 0.
    model.eval()
    for idx,(_x,_y) in enumerate(tqdm(valid_loader,disable=(verbose==0))):
        x_val,y_val = _x.cuda(),_y.cuda()
        with torch.no_grad():
            pred = model(x_val)
            signal_loss = criterion(pred,y_val)
            MSE_loss = MSE(pred,y_val)

            loss = 0.6*signal_loss + 0.4*MSE_loss
            val_loss += dynamic_loss(loss,ddp)/len(valid_loader)
            val_signal_loss += dynamic_loss(signal_loss,ddp)/len(valid_loader)
            val_mse_loss += dynamic_loss(MSE_loss,ddp)/len(valid_loader)
    scheduler.step(val_loss)
    
    if verbose and not is_test:
        if val_loss < best_val:
            torch.save(model.module.state_dict(), os.path.join(save_path,'best.pth'))
            best_val = val_loss
            writer.add_scalar('total_loss/train', train_loss, epoch)
            writer.add_scalar('total_loss/val',val_loss,epoch)
            writer.add_scalar('signal_loss/train', train_signal_loss, epoch)
            writer.add_scalar('signal_loss/val',val_signal_loss,epoch)
            writer.add_scalar('mse_loss/train', train_mse_loss, epoch)
            writer.add_scalar('mse_loss/val',val_mse_loss,epoch)
    
    logging("Epoch [%d]/[%d] train_loss %.6f valid_loss %.6f train_signal_loss %.6f valid_signal_loss %.6f train_mse_loss %.6f valid_mse_loss %.6f duration : %.4f"%(epoch,n_epoch,train_loss,val_loss,train_signal_loss,val_signal_loss, train_mse_loss,val_mse_loss,time.time()-st))
    
logging("[!] training end")