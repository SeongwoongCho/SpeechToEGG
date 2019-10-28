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
from models import Unet,Resv2Unet,ULSTM

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
parser.add_argument('--lr', type=float, default = 1e-2)
parser.add_argument('--lr_step', type=int, default = 65)
parser.add_argument('--exp_num',type=int,default=0)
parser.add_argument('--n_frame',type=int,default=192)
parser.add_argument('--window_ratio', type = float, default = 1.25)
parser.add_argument('--step_ratio',type = float, default = 4)
parser.add_argument('--top_db',type=int,default = 20)
parser.add_argument('--scheduler_gamma',type=float,default=0.1)
parser.add_argument('--test',action='store_true')
parser.add_argument('--local_rank',type=int,default=0)

args = parser.parse_args()

##train ? or test? 
is_test = args.test

##training parameters
n_epoch = args.epoch if not is_test else 1
batch_size = args.batch_size
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
save_path = './models/exp%d/'%args.exp_num
os.makedirs(save_path,exist_ok=True)

##Distributed Data Parallel

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

train_dataset = Dataset(train_X,train_y,normal_noise,musical_noise,n_frame = n_frame, is_train = True, aug = custom_aug_v2(n_frame))
valid_dataset = Dataset(val_X,val_y,n_frame = n_frame, is_train = False)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=4, rank=args.local_rank)
valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas=4, rank=args.local_rank)

train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               num_workers=mp.cpu_count()//4,
                               sampler = train_sampler,
                               pin_memory=True)
valid_loader = data.DataLoader(dataset=valid_dataset,
                               batch_size=batch_size,
                               num_workers=mp.cpu_count()//4,
                               sampler = valid_sampler,
                               pin_memory=True)
# train_loader = data.DataLoader(dataset=train_dataset,
#                                batch_size=batch_size,
#                                num_workers=mp.cpu_count(),
#                                pin_memory=True)

# valid_loader = data.DataLoader(dataset=valid_dataset,
#                                batch_size=val_batch_size,
#                                num_workers=mp.cpu_count(),
#                                pin_memory=True)

logging("Load duration : {}".format(time.time()-st))
logging("[!] load data end")

criterion = CosineDistanceLoss()

model = ULSTM(nlayers = 3, nefilters = 64,filter_size = 15,merge_filter_size = 5,hidden_size = 20, num_layers = 2)
# model = Resv2Unet(5,64,15,5)
model.cuda()

optimizer = RAdam(model.parameters(), lr= learning_rate)
model,optimizer = amp.initialize(model,optimizer,opt_level = 'O1')
scheduler = StepLR(optimizer,step_size=step_size,gamma = scheduler_gamma)
# model = DDP(model,delay_allreduce=True)
# model = DDP(model)

model = torch.nn.parallel.DistributedDataParallel(model,
                                                  device_ids=[args.local_rank],
                                                  output_device=args.local_rank)


logging("[*] training ...")

if verbose and not is_test:
    log = open(os.path.join(save_path,'log.csv'), 'w', encoding='utf-8', newline='')
    log_writer = csv.writer(log)

    best_val = np.inf

for epoch in range(n_epoch):
    train_sampler.set_epoch(epoch)
    st = time.time()
    
    train_loss = 0.
#     optimizer.zero_grad()
    model.train()
    
    for idx,(_x,_y) in enumerate(tqdm(train_loader,disable=(verbose==0))):
        optimizer.zero_grad()
        x_train,y_train = _x.cuda(),_y.cuda()
        pred = model(x_train)
        loss = criterion(pred,y_train)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
#         loss.backward()
        optimizer.step()
        reduced_loss = reduce_tensor(loss.data)
        train_loss += reduced_loss.item() / len(train_loader)
    
#     torch.cuda.synchronize()
    val_loss = 0.
    model.eval()
    for idx,(_x,_y) in enumerate(tqdm(valid_loader,disable=(verbose==0))):
        x_val,y_val = _x.cuda(),_y.cuda()
        with torch.no_grad():
            pred = model(x_val)
            loss = criterion(pred,y_val)
            reduced_loss = reduce_tensor(loss.data)
        val_loss += reduced_loss.item()/len(valid_loader)
            
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