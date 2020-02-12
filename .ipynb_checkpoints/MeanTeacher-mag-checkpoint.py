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
from efficientunet.layers import BatchNorm2d,BatchNorm2dSync

import apex
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
# from utils.sync_batchnorm import convert_model

# from gpuinfo import GPUInfo

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

parser.add_argument('--consistency_weight', type=float, default = 100)
parser.add_argument('--consistency_rampup', type=int, default = 5)
parser.add_argument('--ema_decay',type=float,default = 0.999)
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

## mean teacher hyperparam
consistency_weight = args.consistency_weight
consistency_rampup = args.consistency_rampup
ema_decay = args.ema_decay

##saving path
save_path = './models/sep/mag-only-mean-teacher/{}/'.format(args.exp_num)
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

global_step = 0

def create_model_optimizer(ema=False):
    if ema:
        model = get_efficientunet_b4(out_channels=1, concat_input=True, pretrained=False,mode = 'mag', bn = BatchNorm2d)
    else:
        model = get_efficientunet_b4(out_channels=1, concat_input=True, pretrained=False,mode = 'mag', bn = BatchNorm2dSync)
    model.load_state_dict(torch.load('./models/sep/mag_only/4/best_871.pth',map_location=lambda storage, loc: storage))
    model.cuda()

    if ema:
        optimizer = None
        for param in model.parameters():
            param.requires_grad = False
            param.detach_()
    else:
        optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=True)
        if mixed:
            model,optimizer = amp.initialize(model,optimizer,opt_level = 'O1')

        if ddp:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[args.local_rank],
                                                            output_device=0)
        else:
            model = torch.nn.DataParallel(model)
    return model,optimizer

def create_maskInference_model():
    model = get_efficientunet_b4(out_channels=1, concat_input=True, pretrained=False,mode = 'mask', bn = BatchNorm2d)
    model.load_state_dict(torch.load('./models/sep/mask-only/0/best_171.pth',map_location=lambda storage, loc: storage))
    model.cuda()
    for param in model.parameters():
        param.requires_grad = False
        param.detach_()
    return model
        
st = time.time()
train = np.load('../eggdata/TrainData/train_processing_0205.npy',mmap_mode='r')
val = np.load('../eggdata/TrainData/valid_processing_0205.npy',mmap_mode='r')
unlabeled = np.load('../eggdata/TrainData/unlabeled_v1_0212.npy',mmap_mode='r')
musical_noise = np.load('../eggdata/TrainData/musical_0212.npy',mmap_mode='r')
normal_noise = np.load('../eggdata/TrainData/normal_0212.npy',mmap_mode='r')

logging(train.shape)
logging(val.shape)
logging(unlabeled.shape)

train_dataset = SSLDataset(train,n_sample, train_stride,stft_config, is_train = True,musical_noise = musical_noise, normal_noise = normal_noise)
valid_dataset = Dataset(val,n_sample, valid_stride,stft_config, is_train = False)
Unlabeled_dataset = UnlabelDataset(unlabeled,n_sample,train_stride,stft_config, is_train = True,musical_noise = musical_noise, normal_noise = normal_noise)

train_dataset = torch.utils.ConcatDataset([train_dataset,Unlabeled_dataset])
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

######################## SET OF LOSS CRITERION ############################
mag_criterion = nn.L1Loss(reduction='sum')
cosine_distance_criterion = CosineDistanceLoss()

########################    MODEL DEFINITION   ############################
model,optimizer = create_model_optimizer()
ema_model,_ = create_model_optimizer(ema=True)

maskInference_model = create_maskInference_model()

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)
stftTool = STFT(filter_length=stft_config["window_length"], hop_length=stft_config["hop_length"],window=stft_config["window"]).cuda()

########################        Training       ############################
logging("[*] training ...")
if verbose and not is_test:
    best_val = np.inf
    writer = SummaryWriter('../log/%s/'%args.exp_num)
        
for epoch in range(n_epoch):
    if ddp:
        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)
    train_loss = 0.
    train_supervised_loss = 0.
    train_consistency_loss = 0.
    train_signal_distance = 0.
    
    for idx,((x1,x2),y,labeled) in enumerate(tqdm(train_loader,disable=(verbose==0))):
        model.train()
        ema_model.train()
        x1_train,x2_train,y_train = x1.cuda(),x2.cuda(),y.cuda()
        
        pred = model(x1_train)[:,0,:,:]
        with torch.no_grad():
            ema_pred = ema_model(x2_train)[:,0,:,:].detach()
            teacher_mask = torch.round(torch.sigmoid(maskInference_model(x1_train))/2 + torch.sigmoid(maskInference_model(x2_train))/2)
            teacher_mask = teacher_mask[:,0,:,:].detach()
            
        y_train_mag = y_train[:,0,:,:]
        y_train_phase = y_train[:,1,:,:]
        y_train_mask = y_train[:,2,:,:]
        
        labeled = (labeled == 1)
        optimizer.zero_grad()
        
        ### define supervised loss
        if torch.sum(labeled).item() > 0:
            supervised_loss = L1_criterion(y_train_mask[labeled]*pred[labeled],y_train_mask[labeled]*y_train_mag[labeled])/(torch.sum(y_train_mask[labeled]) + 1e-5)
        else:
            supervised_loss = torch.Tensor([0]).cuda()
        
        ### define consistency loss
        current_consistency_weight = get_current_consistency_weight(consistency_weight,epoch,consistency_rampup)
        consistency_loss = current_consistency_weight*L1_criterion(teacher_mask*pred,teacher_mask*ema_pred)/(torch.sum(teacher_mask) +1e-5)
        ## loss
        loss = supervised_loss + consistency_loss
        
        if mixed:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        
        global_step +=1
        update_ema_variables(model,ema_model,ema_decay,global_step)
        train_loss += dynamic_loss(loss,ddp)/len(train_loader)
        train_supervised_loss += dynamic_loss(supervised_loss,ddp)/len(train_loader)
        train_consistency_loss += dynamic_loss(consistency_loss,ddp)/len(train_loader)

    scheduler.step()
    
    val_loss = 0.
    val_signal_distance = 0.
    model.eval()
    ema_model.eval()
    for idx,(_x,_y) in enumerate(tqdm(valid_loader,disable=(verbose==0))):
        x_val,y_val = _x.cuda(),_y.cuda()
        
        B,_,F,T = x_val.shape
        
        with torch.no_grad():
            pred = model(x_val)
            pred = pred[:,0,:,:]
            y_val_mag = y_val[:,0,:,:]
            y_val_phase = y_val[:,1,:,:]
            y_val_mask = y_val[:,2,:,:]

            loss = L1_criterion(y_val_mask*pred,y_val_mask*y_val_mag)/(torch.sum(y_val_mask) + 1e-5)
            pred_signal_recon = stftTool.inverse(y_val_mask*torch.exp(pred),y_val_phase,n_sample)
            y_val_signal_recon = stftTool.inverse(y_val_mask*torch.exp(y_val_mag),y_val_phase,n_sample)
            signal_distance = cosine_distance_criterion(pred_signal_recon[:,30:-30],y_val_signal_recon[:,30:-30])
        val_loss += dynamic_loss(loss,ddp)/len(loader)
        val_signal_distance += dynamic_loss(signal_distance,ddp)/len(loader)
        
    ### save model and logging onto writer
    if verbose and not is_test:
        torch.save(model.module.state_dict(), os.path.join(save_path,'best_%d.pth'%epoch))
        torch.save(ema_model.state_dict(), os.path.join(save_path,'ema_best_%d.pth'%epoch))
        writer.add_scalar('total_mag_loss/train', train_supervised_loss, epoch)
        writer.add_scalar('total_mag_loss/val',val_loss,epoch)
        writer.add_scalar('consistency_loss/train', train_consistency_loss, epoch)
        writer.add_scalar('signal_distance/val',val_signal_distance, epoch)
    logging("Epoch [%d]/[%d] Metrics([train][valid]) are shown below "%(epoch,n_epoch))
    logging("Total loss [%.6f][%.6f] Consistency loss [%.6f] Signal distance [%.4f]"%(train_loss,val_loss,train_consistency_loss,val_signal_distance))
    
if verbose and not is_test:
    writer.close()
logging("[!] training end")