import csv
import time
import multiprocessing as mp
import argparse
import warnings
import numpy as np
import librosa
from tqdm import tqdm

import torch
from torch import nn
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from utils.radam import RAdam,Lookahead
from utils.aug_utils import custom_stft_aug
from utils.stft_utils.stft import STFT
from utils.utils import *
from utils.loss_utils import CosineDistanceLoss

from dataloader import *
from efficientunet import *

import apex
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

ddp = True
is_test = False
n_frame = 64
batch_size = 384

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank',type=int,default=0)
args = parser.parse_args()

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
train,val = load_stft_datas_path(is_test)
unlabel = load_stft_unlabel_datas_path(is_test)[:50000]
normal_noise,musical_noise = load_stft_noise(is_test)

logging(len(train))
logging(len(unlabel))
logging(len(val))

train_dataset = SSLDataset(train,unlabel,n_frame,normal_noise=normal_noise,musical_noise=musical_noise,is_train = True, aug = custom_stft_aug(n_frame))
valid_dataset = Dataset(val,n_frame, is_train = False)
if ddp:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=4, rank=args.local_rank)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas=4, rank=args.local_rank)
else:
    valid_sampler = None
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

t = time.time()
for idx,((x1,x2),y,labeled) in enumerate(tqdm(train_loader,disable=(verbose==0))):
    print(args.local_rank,time.time()-t)
    t = time.time()