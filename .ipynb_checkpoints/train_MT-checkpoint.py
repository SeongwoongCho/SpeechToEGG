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
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
# from utils.sync_batchnorm import convert_model

seed_everything(42)
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default = 8192)
parser.add_argument('--consistency_weight', type=float, default = 10)
parser.add_argument('--ema_decay',type=float,default = 0.999)
parser.add_argument('--epoch',type=int, default = 100)
parser.add_argument('--weight_decay',type=float,default = 1e-5)
parser.add_argument('--lr', type=float, default = 3e-4)
parser.add_argument('--exp_num',type=str,default='0')
parser.add_argument('--n_frame',type=int,default=64)
parser.add_argument('--scheduler_gamma',type=float,default=0.1)
parser.add_argument('--local_rank',type=int,default=0)
parser.add_argument('--test',action='store_true')
parser.add_argument('--ddp',action='store_true')
parser.add_argument('--mixed',action='store_true')

args = parser.parse_args()

##train ? or test? 
is_test = args.test
mixed = args.mixed
ddp = args.ddp

##training parameters
n_epoch = args.epoch if not is_test else 1
batch_size = args.batch_size//4 if ddp else args.batch_size
weight_decay = args.weight_decay
init_consistency_weight = args.consistency_weight
ema_decay = args.ema_decay

##data preprocessing parameters##
n_frame = args.n_frame

##optimizer parameters##
learning_rate = args.lr

##scheduler parameters##
scheduler_gamma = args.scheduler_gamma

##saving path
save_path = './models/masked/exp{}/'.format(args.exp_num)
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
train,val = load_stft_datas_path(is_test)
unlabel = load_stft_unlabel_datas_path(is_test)
normal_noise,musical_noise = load_stft_noise(is_test)

logging(len(train))
logging(len(unlabel))
logging(len(val))

train_dataset = SSLDataset(train,unlabel,n_frame,normal_noise=normal_noise,musical_noise=musical_noise,is_train = True, aug = custom_stft_aug(n_frame))
valid_dataset = Dataset(val,n_frame, is_train = False)
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

BCE_criterion = nn.BCEWithLogitsLoss()
L1_criterion = nn.L1Loss(reduction='sum')
L2_criterion = nn.MSELoss(reduction='sum')
cosine_distance_criterion = CosineDistanceLoss()

"""
model definition
"""

def create_model_optimizer(ema=False):
    model = get_efficientunet_b2(out_channels=3, concat_input=True, pretrained=False)
    model.load_state_dict(torch.load('./models/masked/exp6-re/best_5761.pth')) 
    model.cuda()
    
    if ema:
        optimizer = None
    else:
        optimizer = RAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer = Lookahead(optimizer,alpha=0.5,k=6)
        if mixed:
            model,optimizer = amp.initialize(model,optimizer,opt_level = 'O1')
    
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=0)
    else:
        model = torch.nn.DataParallel(model)
    
    if ema:
        for param in model.parameters():
            param.detach_()
    
    return model,optimizer

model,optimizer = create_model_optimizer()
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=500, verbose=True)
scheduler = CosineAnnealingLR(optimizer, n_epoch, eta_min=0, last_epoch=-1)

model_ema,_ = create_model_optimizer(ema=True)
    
stftTool = STFT(filter_length=512, hop_length=128,window='hann').cuda()
"""
Training
"""
    
logging("[*] training ...")
if verbose and not is_test:
    best_val = np.inf
    writer = SummaryWriter('../log/%s/'%args.exp_num)   

global_step = 0

for epoch in range(n_epoch): 
    st = time.time()
    train_sampler.set_epoch(epoch)
    valid_sampler.set_epoch(epoch)
    train_loss = 0.
    train_mask_loss = 0.
    train_mag_loss = 0.
    train_phase_loss = 0.
    
    train_mask_accuracy = 0.
    train_false_positive = 0.
    train_false_negative = 0.
    train_signal_distance = 0. ## just for monitering
    
    consistency_loss = 0.
    consistency_mask_loss = 0.
    consistency_mag_loss = 0.
    consistency_phase_loss = 0.
    
    model.train()
    ema_model.train()
    
    for idx,((x1,x2),y,labeled) in enumerate(tqdm(train_loader,disable=(verbose==0))):
        x1_train,x2_train,y_train = x1.cuda(),x2.cuda(),y.cuda()
        
        B,_,F,T = x_train.shape
        
        pred = model(x1_train)
        pred_mag = pred[:,0,:,:].unsqueeze(1)
        pred_phase = pred[:,1,:,:].unsqueeze(1)
        pred_mask = pred[:,2,:,:].unsqueeze(1)
        
        with torch.no_grad():
            ema_pred = model_ema(x2_train).detach()
            ema_pred_mag = ema_pred[:,0,:,:].unsqueeze(1)
            ema_pred_phase = ema_pred[:,1,:,:].unsqueeze(1)
            ema_pred_mask = ema_pred[:,2,:,:].unsqueeze(1)
        
        y_train_mag = y_train[:,0,:,:].unsqueeze(1)
        y_train_phase = y_train[:,1,:,:].unsqueeze(1)
        y_train_mask = y_train[:,2,:,:].unsqueeze(1)
        
        optimizer.zero_grad()
        
        ### define supervised loss
        mask_loss = BCE_criterion(pred_mask[labeled==1],y_train_mask[labeled==1])
        mag_loss = L1_criterion(y_train_mask[labeled==1]*pred_mag[labeled==1],y_train_mask[labeled==1]*y_train_mag[labeled==1])/(torch.sum(y_train_mask[labeled==1]) + 1e-5)
        phase_loss = L1_criterion(y_train_mask[labeled==1]*pred_phase[labeled==1],y_train_mask[labeled==1]*y_train_phase[labeled==1])/(torch.sum(y_train_mask[labeled==1]) + 1e-5)
        supervised_loss = mask_loss + mag_loss + phase_loss
        
        ### define consistency loss
        consistency_weight = get_current_consistency_weight(init_consistency_weight,epoch,n_epoch)
        consistency_mask_loss = L2_criterion(torch.sigmoid(pred_mask),torch.sigmoid(ema_pred_mask))/(torch.sum(y_train_mask))
        
        teacher_mask = torch.round(torch.sigmoid(ema_pred_mask))
        consistency_mag_loss = L2_criterion(teacher_mask*pred_mag,teacher_mask*ema_pred_mag)
        consistency_phase_loss = L2_criterion(teacher_mask*pred_phase,teacher_mask*ema_pred_phase)
        consistency_loss = consistency_weight*(consistency_mask_loss + consistency_mag_loss + consistency_phase_loss)
        
        ## loss
        loss = supervised_loss + consistency_loss
        
        if mixed:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        
        global_step +=1
        update_ema_variables(model, ema_model, ema_decay, global_step)
        
        ## evaluate on labeled image
        zero_mask = torch.zeros_like(pred_mask[labeled==1])
        pred_mask = torch.round(torch.sigmoid(pred_mask[labeled==1]))
        mask_diff = pred_mask - y_train_mask[labeled==1]
        mask_accuracy = torch.mean(torch.eq(mask_diff,zero_mask).type(torch.cuda.FloatTensor))
        false_negative = torch.mean(torch.eq(mask_diff,zero_mask-1).type(torch.cuda.FloatTensor)) ## voice(1)인데 unvoice(0)로 mask한 경우
        false_positive = torch.mean(torch.eq(mask_diff,zero_mask+1).type(torch.cuda.FloatTensor)) ## unvoice(0)인데 voice(1)로 mask한 경우
        
        pred_mask = pred_mask[:,0,:,:]
        pred_mag = pred_mag[labeled==1][:,0,:,:]
        pred_phase = pred_phase[labeled==1][:,0,:,:]
        
        y_train_mask = y_train_mask[labeled==1][:,0,:,:]
        y_train_mag = y_train_mag[labeled==1][:,0,:,:]
        y_train_phase = y_train_phase[labeled==1][:,0,:,:]
        
        pred_signal_recon = stftTool.inverse(pred_mask*torch.exp(pred_mag),pred_phase,n_frame*128)
        y_train_signal_recon = stftTool.inverse(y_train_mask*torch.exp(y_train_mag),y_train_phase,n_frame*128)
        signal_distance = cosine_distance_criterion(pred_signal_recon[:,30:-30],y_train_signal_recon[:,30:-30])

        train_loss += dynamic_loss(loss,ddp)/len(train_loader)
        train_mask_loss += dynamic_loss(mask_loss,ddp)/len(train_loader)
        train_mag_loss += dynamic_loss(mag_loss,ddp)/len(train_loader)
        train_phase_loss += dynamic_loss(phase_loss,ddp)/len(train_loader)
        
        consistency_loss += dynamic_loss(consistency_loss,ddp)/len(train_loader)
        consistency_mask_loss += dynamic_loss(consistency_mask_loss,ddp)/len(train_loader)
        consistency_mag_loss += dynamic_loss(consistency_mag_loss,ddp)/len(train_loader)
        consistency_phase_loss += dynamic_loss(consistency_phase_loss,ddp)/len(train_loader)
        
        train_mask_accuracy += dynamic_loss(mask_accuracy,ddp)/len(train_loader)
        train_false_negative += dynamic_loss(false_negative,ddp)/len(train_loader)
        train_false_positive += dynamic_loss(false_positive,ddp)/len(train_loader)
        train_signal_distance += dynamic_loss(signal_distance,ddp)/len(train_loader)
        
    val_loss = 0.
    val_mask_loss = 0.
    val_mag_loss = 0.
    val_phase_loss = 0.
    
    val_mask_accuracy = 0.
    val_false_positive = 0.
    val_false_negative = 0.
    val_signal_distance = 0. ## just for monitering
    
    saving_image_pred_mask = []
    saving_image_pred_mag = []
    saving_image_pred_phase = []
    saving_image_true_mask = []
    saving_image_true_mag = []
    saving_image_true_phase = []
    
    model.eval()
    ema_model.eval()
    
    for idx,(_x,_y) in enumerate(tqdm(valid_loader,disable=(verbose==0))):
        x_val,y_val = _x.cuda(),_y.cuda()
        
        B,_,F,T = x_val.shape
        
        with torch.no_grad():
            pred = model(x_val)
        
            pred_mag = pred[:,0,:,:].unsqueeze(1)
            pred_phase = pred[:,1,:,:].unsqueeze(1)
            pred_mask = pred[:,2,:,:].unsqueeze(1)
            y_val_mag = y_val[:,0,:,:].unsqueeze(1)
            y_val_phase = y_val[:,1,:,:].unsqueeze(1)
            y_val_mask = y_val[:,2,:,:].unsqueeze(1)

            mask_loss = BCE_criterion(pred_mask,y_val_mask)
            mag_loss = L1_criterion(y_val_mask*pred_mag,y_val_mask*y_val_mag)/(torch.sum(y_val_mask) + 1e-5)
            phase_loss = L1_criterion(y_val_mask*pred_phase,y_val_mask*y_val_phase)/(torch.sum(y_val_mask) + 1e-5)
            loss = mask_loss + mag_loss + phase_loss
        
            zero_mask = torch.zeros_like(pred_mask)
            pred_mask = torch.round(torch.sigmoid(pred_mask))
            mask_diff = pred_mask - y_val_mask
            mask_accuracy = torch.mean(torch.eq(mask_diff,zero_mask).type(torch.cuda.FloatTensor))
            false_negative = torch.mean(torch.eq(mask_diff,zero_mask-1).type(torch.cuda.FloatTensor)) ## voice(1)인데 unvoice(0)로 mask한 경우
            false_positive = torch.mean(torch.eq(mask_diff,zero_mask+1).type(torch.cuda.FloatTensor)) ## unvoice(0)인데 voice(1)로 mask한 경우
        
            if verbose:
                saving_image_pred_mask.append(pred_mask.cpu().detach())
                saving_image_pred_mag.append(pred_mag.cpu().detach())
                saving_image_pred_phase.append(pred_phase.cpu().detach())
                saving_image_true_mask.append(y_val_mask.cpu().detach())
                saving_image_true_mag.append(y_val_mag.cpu().detach())
                saving_image_true_phase.append(y_val_phase.cpu().detach())

            pred_mask = pred_mask[:,0,:,:]
            pred_mag = pred_mag[:,0,:,:]
            pred_phase = pred_phase[:,0,:,:]

            y_val_mask = y_val_mask[:,0,:,:]
            y_val_mag = y_val_mag[:,0,:,:]
            y_val_phase = y_val_phase[:,0,:,:]

            pred_signal_recon = stftTool.inverse(pred_mask*torch.exp(pred_mag),pred_phase,n_frame*128)
            y_val_signal_recon = stftTool.inverse(y_val_mask*torch.exp(y_val_mag),y_val_phase,n_frame*128)
            signal_distance = cosine_distance_criterion(pred_signal_recon[:,30:-30],y_val_signal_recon[:,30:-30])
        
        val_loss += dynamic_loss(loss,ddp)/len(valid_loader)
        val_mask_loss += dynamic_loss(mask_loss,ddp)/len(valid_loader)
        val_mag_loss += dynamic_loss(mag_loss,ddp)/len(valid_loader)
        val_phase_loss += dynamic_loss(phase_loss,ddp)/len(valid_loader)
        val_mask_accuracy += dynamic_loss(mask_accuracy,ddp)/len(valid_loader)
        val_false_negative += dynamic_loss(false_negative,ddp)/len(valid_loader)
        val_false_positive += dynamic_loss(false_positive,ddp)/len(valid_loader)
        val_signal_distance += dynamic_loss(signal_distance,ddp)/len(valid_loader)
        
#     scheduler.step(val_signal_distance)
    scheduler.step()
    
    if verbose and not is_test:
        torch.save(model.module.state_dict(), os.path.join(save_path,'best_%d.pth'%epoch))
        torch.save(ema_model.module.state_dict(), os.path.join(save_path,'ema_best_%d.pth'%epoch))
        
        if epoch%50 == 0:
            saving_image_pred_mask = torch.cat(saving_image_pred_mask,dim=0)
            saving_image_true_mask = torch.cat(saving_image_true_mask,dim=0)
            saving_image_pred_mag = torch.cat(saving_image_pred_mag,dim=0)
            saving_image_true_mag = torch.cat(saving_image_true_mag,dim=0)
            saving_image_pred_phase = torch.cat(saving_image_pred_phase,dim=0)
            saving_image_true_phase = torch.cat(saving_image_true_phase,dim=0)

            saving_image_pred_mask = torch.cat([saving_image_pred_mask,saving_image_pred_mask,saving_image_pred_mask],dim=1)
            saving_image_true_mask = torch.cat([saving_image_true_mask,saving_image_true_mask,saving_image_true_mask],dim=1)
            saving_image_pred_mag = torch.cat([saving_image_pred_mag,saving_image_pred_mag,saving_image_pred_mag],dim=1)
            saving_image_true_mag = torch.cat([saving_image_true_mag,saving_image_true_mag,saving_image_true_mag],dim=1)
            saving_image_pred_phase = torch.cat([saving_image_pred_phase,saving_image_pred_phase,saving_image_pred_phase],dim=1)
            saving_image_true_phase = torch.cat([saving_image_true_phase,saving_image_true_phase,saving_image_true_phase],dim=1)

            saving_image_pred_mask = make_grid(saving_image_pred_mask, normalize=True, scale_each=True)
            saving_image_true_mask = make_grid(saving_image_true_mask, normalize=True, scale_each=True)
            saving_image_pred_mag = make_grid(saving_image_pred_mag, normalize=True, scale_each=True)
            saving_image_true_mag = make_grid(saving_image_true_mag, normalize=True, scale_each=True)
            saving_image_pred_phase = make_grid(saving_image_pred_phase, normalize=True, scale_each=True)
            saving_image_true_phase = make_grid(saving_image_true_phase, normalize=True, scale_each=True)

            writer.add_image('img_true_mag/stft', saving_image_true_mag, epoch)
            writer.add_image('img_true_phase/stft', saving_image_true_phase, epoch)
            writer.add_image('img_true_mask/stft', saving_image_true_mask, epoch)
            writer.add_image('img_pred_mag/stft', saving_image_pred_mag, epoch)
            writer.add_image('img_pred_phase/stft', saving_image_pred_phase, epoch)
            writer.add_image('img_pred_mask/stft', saving_image_pred_mask, epoch)

        writer.add_scalar('total_loss/train', train_loss, epoch)
        writer.add_scalar('total_loss/val',val_loss,epoch)
        writer.add_scalar('mask_loss/train', train_mask_loss, epoch)
        writer.add_scalar('mask_loss/val',val_mask_loss,epoch)
        writer.add_scalar('mag_loss/train', train_mag_loss, epoch)
        writer.add_scalar('mag_loss/val',val_mag_loss,epoch)
        writer.add_scalar('phase_loss/train', train_phase_loss, epoch)
        writer.add_scalar('phase_loss/val',val_phase_loss,epoch)
        
        writer.add_scalar('consistency_loss/train', consistency_loss, epoch)
        writer.add_scalar('consistency_mask_loss/train', consistency_mask_loss, epoch)
        writer.add_scalar('consistency_mag_loss/train', consistency_mag_loss, epoch)
        writer.add_scalar('consistency_phase_loss/train', consistency_phase_loss, epoch)
        
        writer.add_scalar('mask_accuracy/train',train_mask_accuracy, epoch)
        writer.add_scalar('mask_accuracy/val',val_mask_accuracy, epoch)
        writer.add_scalar('false_positive/train',train_false_positive, epoch)
        writer.add_scalar('false_positive/val',val_false_positive, epoch)
        writer.add_scalar('false_negative/train',train_false_negative, epoch)
        writer.add_scalar('false_negative/val',val_false_negative, epoch)
        writer.add_scalar('signal_distance/train',train_signal_distance, epoch)
        writer.add_scalar('signal_distance/val',val_signal_distance, epoch)
        
    logging("Epoch [%d]/[%d] Metrics([train][valid]) are shown below "%(epoch,n_epoch))
    logging("Total loss [%.6f][%.6f] Mask loss [%.6f][%.6f] Mag loss [%.6f][%.6f] Phase loss [%.6f][%.6f] Mask accuracy [%.4f][%.4f] False Positive [%.4f][%.4f] False Negative [%.4f][%.4f] Signal distance [%.4f][%.4f]"%(train_loss,val_loss,train_mask_loss,val_mask_loss,train_mag_loss,val_mag_loss,train_phase_loss,val_phase_loss,train_mask_accuracy,val_mask_accuracy,train_false_positive,val_false_positive,train_false_negative,val_false_negative,train_signal_distance,val_signal_distance))
    logging("Consistency loss [%.6f] Mask loss [%.6f] Mag loss [%.6f] Phase loss [%.6f]"%
            (consistency_loss, consistency_mask_loss, consistency_mag_loss, consistency_phase_loss))
if verbose and not is_test:
    writer.close()
logging("[!] training end")

