import time
import multiprocessing as mp
import argparse
import warnings
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from utils.stft_utils.stft import STFT
from utils.utils import *
from utils.loss_utils import CosineDistanceLoss

from dataloader import *
from efficientunet import *

import apex
import sys
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

seed_everything(42)
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default = 8192)
parser.add_argument('--train_stride', type=int, default = 192)
parser.add_argument('--valid_stride', type=int, default = 64)
parser.add_argument('--n_sample',type = int, default = 4096)

parser.add_argument('--window_length', type = int, default = 512)
parser.add_argument('--hop_length', type = int, default = 256)
parser.add_argument('--window', type = str, default = 'hann')

parser.add_argument('--epoch',type=int, default = 100)
parser.add_argument('--weight_decay',type=float,default = 1e-5)
parser.add_argument('--lr', type=float, default = 3e-4)
parser.add_argument('--patience',type=int,default=30)
parser.add_argument('--momentum',type=float,default=0.9)
parser.add_argument('--exp_num',type=str,default='0')
parser.add_argument('--local_rank',type=int,default=0)
parser.add_argument('--ddp',action='store_true')
parser.add_argument('--mixed',action='store_true')

args = parser.parse_args()

##train ? or test? 
mixed = args.mixed
ddp = args.ddp

## stft config
stft_config = {'window_length':args.window_length,
               'hop_length':args.hop_length,
               'window':args.window}
n_sample = args.n_sample

##training parameters
n_epoch = args.epoch
batch_size = args.batch_size//6 if ddp else args.batch_size

##optimizer parameters##
learning_rate = args.lr
weight_decay = args.weight_decay
patience = args.patience
momentum = args.momentum

## data
train_stride = args.train_stride
valid_stride = args.valid_stride

##saving path
save_path = './models/sep/mag-only/{}/'.format(args.exp_num)
os.makedirs(save_path,exist_ok=True)

##Distributed Data Parallel
if ddp:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    args.world_size = torch.distributed.get_world_size()
    
verbose = 1 if args.local_rank ==0 else 0


# print(args.local_rank)
if not verbose:
    warnings.filterwarnings(action='ignore')

logging = print_verbose(verbose)
logging("[*] load data ...")

st = time.time()
train = np.load('../eggdata/TrainData/train_processing_0205.npy',mmap_mode='r')
val = np.load('../eggdata/TrainData/valid_processing_0205.npy',mmap_mode='r')
# musical_noise = None
# normal_noise = None
musical_noise = np.load('../eggdata/TrainData/musical_0212.npy',mmap_mode='r')
normal_noise = np.load('../eggdata/TrainData/normal_0212.npy',mmap_mode='r')

logging(train.shape)
logging(val.shape)

train_dataset = Dataset(train,n_sample, train_stride,stft_config, is_train = True,musical_noise = musical_noise, normal_noise = normal_noise)
valid_dataset = Dataset(val,n_sample, valid_stride,stft_config, is_train = False)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=6, rank=args.local_rank)
valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas=6, rank=args.local_rank)
train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               num_workers=mp.cpu_count()//6 if ddp else mp.cpu_count(),
                               sampler = train_sampler if ddp else None,
                               shuffle = True if not ddp else False,
                               pin_memory=True)
valid_loader = data.DataLoader(dataset=valid_dataset,
                               batch_size=batch_size,
                               num_workers=mp.cpu_count()//6 if ddp else mp.cpu_count(),
                               sampler = valid_sampler if ddp else None,
                               pin_memory=True)

logging("Load duration : {}".format(time.time()-st))
logging("[!] load data end")
mag_criterion = nn.L1Loss(reduction='sum')
cosine_distance_criterion = CosineDistanceLoss()
"""
model definition
"""

model = get_efficientunet_b4(out_channels=1, concat_input=True, pretrained=False,mode = 'mag', bn = BatchNorm2dSync)
model.cuda()

optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=True)

if mixed:
    model,optimizer = amp.initialize(model,optimizer,opt_level = 'O1')
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)

if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.local_rank],
                                                      output_device=0)
else:
    model = torch.nn.DataParallel(model)

stftTool = STFT(filter_length=stft_config["window_length"], hop_length=stft_config["hop_length"],window=stft_config["window"]).cuda()

"""
Training
"""

logging("[*] training ...")
if verbose:
    best_val = np.inf
    writer = SummaryWriter('../logs/mag-only/%s/'%args.exp_num)   
    
for epoch in range(n_epoch): 
    st = time.time()
    train_sampler.set_epoch(epoch)
    valid_sampler.set_epoch(epoch)
    train_loss = 0.
    train_signal_distance = 0.
    
    model.train()
    
    for idx,(_x,_y) in enumerate(tqdm(train_loader,disable=(verbose==0))):
        x_train,y_train = _x.cuda(),_y.cuda()
        B,_,F,T = x_train.shape
        
        pred = model(x_train)[:,0,:,:]
        y_train_mag = y_train[:,0,:,:]
        y_train_phase = y_train[:,1,:,:]
        y_train_mask = y_train[:,2,:,:]
        
        optimizer.zero_grad()
        
        loss = mag_criterion(y_train_mask*pred,y_train_mask*y_train_mag)/(torch.sum(y_train_mask) + 1e-5)
        if mixed:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        
        pred_signal_recon = stftTool.inverse(y_train_mask*torch.exp(pred),y_train_phase,n_sample)
        y_train_signal_recon = stftTool.inverse(y_train_mask*torch.exp(y_train_mag),y_train_phase,n_sample)
        signal_distance = cosine_distance_criterion(pred_signal_recon[:,30:-30],y_train_signal_recon[:,30:-30])
        
        train_loss += dynamic_loss(loss,ddp)/len(train_loader)
        train_signal_distance += dynamic_loss(signal_distance,ddp)/len(train_loader)
        
    val_loss = 0.
    val_signal_distance = 0.
    
    model.eval()
    for idx,(_x,_y) in enumerate(tqdm(valid_loader,disable=(verbose==0))):
        x_val,y_val = _x.cuda(),_y.cuda()
        B,_,F,T = x_val.shape
        
        with torch.no_grad():
            pred = model(x_val)[:,0,:,:]
            y_val_mag = y_val[:,0,:,:]
            y_val_phase = y_val[:,1,:,:]
            y_val_mask = y_val[:,2,:,:]

            loss = mag_criterion(y_val_mask*pred,y_val_mask*y_val_mag)/(torch.sum(y_val_mask) + 1e-5)
            pred_signal_recon = stftTool.inverse(y_val_mask*torch.exp(pred),y_val_phase,n_sample)
            y_val_signal_recon = stftTool.inverse(y_val_mask*torch.exp(y_val_mag),y_val_phase,n_sample)
            signal_distance = cosine_distance_criterion(pred_signal_recon[:,30:-30],y_val_signal_recon[:,30:-30])

            val_loss += dynamic_loss(loss,ddp)/len(valid_loader)
            val_signal_distance += dynamic_loss(signal_distance,ddp)/len(valid_loader)
        
    scheduler.step(val_loss)
    
    if verbose:
        if val_loss < best_val:
            best_val = val_loss
    
    if verbose:
        torch.save(model.module.state_dict(), os.path.join(save_path,'best_%d.pth'%epoch))
        writer.add_scalar('total_loss/train', train_loss, epoch)
        writer.add_scalar('total_loss/val',val_loss,epoch)
        writer.add_scalar('signal_distance/train',train_signal_distance, epoch)
        writer.add_scalar('signal_distance/val',val_signal_distance, epoch)
        
    logging("Epoch [%d]/[%d] Metrics([train][valid]) are shown below "%(epoch,n_epoch))
    logging("Total loss [%.6f][%.6f] Signal distance [%.4f][%.4f]"%(train_loss,val_loss,train_signal_distance, val_signal_distance))
    
if verbose:
    writer.close()
logging("[!] training end")
if verbose:
    print(best_val)