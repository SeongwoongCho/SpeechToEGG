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
from torch.autograd import Variable
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from stft_utils.stft import STFT
import torch.distributed as dist

from radam import RAdam,Lookahead
from utils import *
from utils import _get_CQSQ
from dataloader import *
from models import MMDenseNet, Vgg16, Vgg19, pixelGAN_discriminator, patchGAN_discriminator, customGAN_discriminator,InterpolateWrapper
import segmentation_models_pytorch as smp

import apex
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from sync_batchnorm import convert_model

seed_everything(42)
###Hyper parameters

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default = 8192)
parser.add_argument('--epoch',type=int, default = 100)
parser.add_argument('--beta1',type=float,default = 0.9)
parser.add_argument('--beta2',type=float,default = 0.999)
parser.add_argument('--weight_decay',type=float,default = 1e-5)
parser.add_argument('--loss_ratio',type=float,default = 10)
parser.add_argument('--TTUR',type=float,default=1)
parser.add_argument('--lr', type=float, default = 3e-4)
parser.add_argument('--lr_step', type=int, default = 65)
parser.add_argument('--exp_num',type=str,default='0')
parser.add_argument('--n_frame',type=int,default=10000)
parser.add_argument('--n_fft',type=int,default=512)
parser.add_argument('--hop_length',type=int,default=128)
parser.add_argument('--scheduler_gamma',type=float,default=0.1)
parser.add_argument('--test',action='store_true')
parser.add_argument('--start_epoch',type=int,default=0)
parser.add_argument('--mode',type=str,default ='ddp')
parser.add_argument('--spectral',action='store_true')
parser.add_argument('--mixed',action='store_true')
parser.add_argument('--d_epoch',type=int,default=1)
parser.add_argument('--local_rank',type=int,default=0)

args = parser.parse_args()

##train ? or test? 
is_test = args.test
mixed = args.mixed
mode = args.mode
start_epoch = args.start_epoch
##training parameters
n_epoch = args.epoch if not is_test else start_epoch+1
batch_size = args.batch_size//4 if mode == 'ddp' else args.batch_size
beta1 = args.beta1
beta2 = args.beta2
weight_decay = args.weight_decay
loss_ratio = args.loss_ratio
d_epoch = args.d_epoch
spectral = args.spectral
# e_epoch = args.e_epoch

##data preprocessing parameters##
n_frame = args.n_frame
n_fft = args.n_fft
hop_length = args.hop_length

##optimizer parameters##
learning_rate = args.lr
TTUR = args.TTUR

assert TTUR>=1, "TTUR ratio should >= 1, generator's learning rate < discriminator's learning rate"

##scheduler parameters## 
step_size = args.lr_step - start_epoch
scheduler_gamma = args.scheduler_gamma

##saving path
save_path = './models/fft/exp{}/'.format(args.exp_num)
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
train,val = load_stft_datas_path(is_test)
normal_noise,musical_noise = load_stft_noise(is_test)

logging(len(train))
logging(len(val))

# train_dataset = Dataset(train_X,train_y,n_fft,hop_length,n_frame,normal_noise=normal_noise,musical_noise=musical_noise,is_train = True, aug = custom_aug_v3(n_frame))
# valid_dataset = Dataset(val_X,val_y,n_fft,hop_length,n_frame, is_train = False)
train_dataset = STFTDataset(train,n_frame,normal_noise=normal_noise,musical_noise=musical_noise,is_train = True, aug = custom_stft_aug(n_frame))
valid_dataset = STFTDataset(val,n_frame, is_train = False)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=4, rank=args.local_rank)
valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas=4, rank=args.local_rank)
train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               num_workers=mp.cpu_count()//4 if mode == 'ddp' else mp.cpu_count(),
                               sampler = train_sampler if mode == 'ddp' else None,
                               shuffle = True if mode !='ddp' else False,
                               pin_memory=True)
valid_loader = data.DataLoader(dataset=valid_dataset,
                               batch_size=batch_size,
                               num_workers=mp.cpu_count()//4 if mode == 'ddp' else mp.cpu_count(),
                               sampler = valid_sampler if mode == 'ddp' else None,
                               pin_memory=True)
logging("Load duration : {}".format(time.time()-st))
logging("[!] load data end")

MSE_criterion = nn.MSELoss(reduction='mean')
BCE_criterion = nn.BCEWithLogitsLoss()
# spec_criterion = spectral_loss(coeff = [0,1,1,0])
# cdl_criterion = CosineDistanceLoss()
L1_criterion = nn.L1Loss()

"""
Define generator/discriminator
"""
# ENCODER = 'resnet18'
# ENCODER_WEIGHTS = None
# CLASSES = ['voice']
# ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation

# # create segmentation model with pretrained encoder
# model = smp.Unet(
#     encoder_name=ENCODER,
#     encoder_weights=None,
#     activation = ACTIVATION,
# #     encoder_depth=2,
#     in_channels = 1
# )

# model = InterpolateWrapper(model)
# model.encoder.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=model.encoder.conv1.out_channels, kernel_size=model.encoder.conv1.kernel_size, stride=model.encoder.conv1.stride, padding=model.encoder.conv1.padding, bias=model.encoder.conv1.bias)

model = MMDenseNet(drop_rate=0.25,bn_size=4,k1=10,l1=3,k2=14,l2=4,attention = 'CBAM')
# discriminator = pixelGAN_discriminator(2) ## input channel numbers = 2
# discriminator = patchGAN_discriminator(2) ## input channel numbers = 2
discriminator = customGAN_discriminator(2,drop_rate=0.25,spectral = spectral)

if mode == 'ddp':
#     model = apex.parallel.convert_syncbn_model(model)
#     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = convert_model(model)
    discriminator = convert_model(discriminator)

model.cuda()
discriminator.cuda()

## adjust TTUR
optimizer = RAdam(model.parameters(), lr=learning_rate/TTUR, betas=(beta1, beta2), eps=1e-8, weight_decay=weight_decay)
optimizer = Lookahead(optimizer,alpha=0.5,k=6)
optimizer_d = RAdam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=1e-8, weight_decay=0 if spectral else weight_decay)
optimizer_d = Lookahead(optimizer_d,alpha=0.5,k=6)
if mixed:
    model,optimizer = amp.initialize(model,optimizer,opt_level = 'O1')
    discriminator,optimizer_d = amp.initialize(discriminator,optimizer_d,opt_level = 'O1')

scheduler = StepLR(optimizer,step_size=step_size,gamma = scheduler_gamma)
scheduler_d = StepLR(optimizer_d,step_size=step_size,gamma = scheduler_gamma)

if mode == 'ddp':
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.local_rank],
                                                      output_device=0)
    discriminator = torch.nn.parallel.DistributedDataParallel(discriminator,
                                                      device_ids=[args.local_rank],
                                                      output_device=0)    
else:
    model = torch.nn.DataParallel(model)
    discriminator = torch.nn.DataParallel(discriminator)
    
# """
# vgg model for computing composite spectrogram loss
# """
# vgg =  Vgg16(requires_grad=False)
# if mode == 'ddp':
# #     model = apex.parallel.convert_syncbn_model(model)
# #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
#     vgg = convert_model(vgg)
# vgg.cuda()

# if mode == 'ddp':
# #     vgg = torch.nn.parallel.DistributedDataParallel(vgg,
# #                                                       device_ids=[args.local_rank],
# #                                                       output_device=0)
#     pass
# else:
#     vgg = torch.nn.DataParallel(vgg)    

"""
Training
"""
    
logging("[*] training ...")
if verbose and not is_test:
#     log = open(os.path.join(save_path,'log.csv'), 'a+', encoding='utf-8', newline='')
#     log_writer = csv.writer(log)
    writer = SummaryWriter('../log/%s/'%args.exp_num)
#     hparams = {'batch_size' : batch_size, 'n_frame' : n_frame, 'lr' : learning_rate,'lr_step' : step_size,'gamma':scheduler_gamma,'beta1':beta1,'beta2':beta2, 'weight_decay' : weight_decay}
    
#     writer.add_hparams(hparams,{})
    
    best_val = np.inf
    best_metric = np.inf
# stft = STFT(filter_length = n_fft, hop_length = hop_length,window='hann').cuda()
# pool = mp.Pool(mp.cpu_count()//4 if mode == 'ddp' else mp.cpu_count())

if start_epoch >0:
    model.load_state_dict(torch.load(save_path+"best.pth"))

train_discriminator = True    
    
for epoch in range(start_epoch,n_epoch): 
    st = time.time()
    train_sampler.set_epoch(epoch)
    train_loss = 0.
    train_L1_loss = 0.
    train_dtime_loss = 0.
    train_dfreq_loss = 0.
    train_disc_loss = 0.
    train_disc_accuracy_real = 0.
    train_disc_accuracy_fake = 0.
    train_gen_accuracy = 0.
#     train_content_loss = 0.
#     train_style_loss = 0.
    
#     train_CQ_diff = 0.
#     train_SQ_diff = 0.
#     optimizer.zero_grad()
    model.train()
    discriminator.train()
    
    for idx,(_x,_y) in enumerate(tqdm(train_loader,disable=(verbose==0))):
        x_train,y_train = _x.cuda(),_y.cuda()
        pred = model(x_train)

#         vgg_input_pred = normalize_batch(pred)
#         vgg_input_y = normalize_batch(y_train)
#         pred_feature = vgg(vgg_input_pred)
#         true_feature = vgg(vgg_input_y)
#         gram_style = [gram_matrix(y) for y in true_feature]
        
        ## ===================== Train Generator =====================#

        optimizer.zero_grad()
        L1_loss = L1_criterion(pred,y_train)
        dfreq_loss = L1_criterion(pred[:,:,1:,:]-pred[:,:,:-1,:],y_train[:,:,1:,:]-y_train[:,:,:-1,:])
#         L2_loss = MSE_criterion(pred,y_train)
#         dtime_loss = MSE_criterion(pred[:,:,:,1:]-pred[:,:,:,:-1],y_train[:,:,:,1:]-y_train[:,:,:,:-1])
#         dfreq_loss = MSE_criterion(pred[:,:,1:,:]-pred[:,:,:-1,:],y_train[:,:,1:,:]-y_train[:,:,:-1,:])
#         content_loss = MSE_criterion(pred_feature[1],true_feature[1]) ##relu2_2
#         style_loss = 0.
#         for ft_pred,gm_s in zip(pred_feature,gram_style):
#             gm_y = gram_matrix(ft_pred)
#             style_loss += MSE_criterion(gm_y,gm_s)
        
#         loss = 0.2*L2_loss + 0.2*dtime_loss + 0.2*dfreq_loss + 0.2*content_loss + 0.2*style_loss
#         recon_loss = 0.34*L2_loss + 0.33*dtime_loss + 0.33*dfreq_loss
        recon_loss = 0.5*L1_loss + 0.5*dfreq_loss
    
        disc_input_fake = torch.cat([x_train,pred],dim=1)
        disc_output_fake = discriminator(disc_input_fake)
        disc_label_real = torch.ones_like(disc_output_fake)
        B,_,W,H = disc_label_real.shape
        gen_loss = BCE_criterion(disc_output_fake,disc_label_real) + loss_ratio*recon_loss
#         logging(torch.round(torch.sigmoid(disc_output_fake)))
        gen_accuracy = torch.round(torch.sigmoid(disc_output_fake)).eq(disc_label_real).sum().type(torch.cuda.FloatTensor)/(B*W*H) ##얼마나 잘 속이냐
        
        if mixed:
            with amp.scale_loss(gen_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            gen_loss.backward()
        optimizer.step()
        
        # ===================== Train discriminator =====================#
        if train_discriminator:
            optimizer_d.zero_grad()
            
            ### TODO: adjust tricks for training stablizing : noise/label smoothing/TTUR/Spectral normlization + FID SCORE
            ## add noise while training discriminator
            disc_input_fake = torch.cat([x_train,pred.detach() + 0.05*torch.randn_like(pred)],dim=1)
            disc_input_real = torch.cat([x_train,y_train + 0.05*torch.randn_like(y_train)],dim=1) ## 2sigma = 0.1 (8% 범주)
            disc_output_fake = discriminator(disc_input_fake)
            disc_output_real = discriminator(disc_input_real)
            
            
            disc_label_real = Variable(torch.ones_like(disc_output_real),requires_grad=False)
            disc_label_fake = Variable(torch.zeros_like(disc_output_fake),requires_grad=False)
            
            ## adjust one-sided label smoothing
            disc_loss = (BCE_criterion(disc_output_real,disc_label_real-0.1) + BCE_criterion(disc_output_fake,disc_label_fake+0.1))*0.5

            B,_,W,H = disc_label_fake.shape
            disc_accuracy_real = torch.round(torch.sigmoid(disc_output_real)).eq(disc_label_real).sum().type(torch.cuda.FloatTensor)/(B*W*H) 
            disc_accuracy_fake = torch.round(torch.sigmoid(disc_output_fake)).eq(disc_label_fake).sum().type(torch.cuda.FloatTensor)/(B*W*H)

            if mixed:
                with amp.scale_loss(disc_loss, optimizer_d) as scaled_loss:
                    scaled_loss.backward()
            else:
                disc_loss.backward()
            optimizer_d.step()
        
        if mode == 'ddp':
            reduced_loss = reduce_tensor(gen_loss.data)
            train_loss += reduced_loss.item() / len(train_loader)
            reduced_L1_loss = reduce_tensor(L1_loss.data)
            train_L1_loss += reduced_L1_loss.item() / len(train_loader)
#             reduced_content_loss = reduce_tensor(cntent_loss.data)
#             train_content_loss += reduced_content_loss.item() / len(train_loader)
#             reduced_style_loss = reduce_tensor(style_loss.data)
#             train_style_loss += reduced_style_loss.item() / len(train_loader)
#             reduced_dtime_loss = reduce_tensor(dtime_loss.data)
#             train_dtime_loss += reduced_dtime_loss.item() / len(train_loader)
            reduced_dfreq_loss = reduce_tensor(dfreq_loss.data)
            train_dfreq_loss += reduced_dfreq_loss.item() / len(train_loader)
            reduced_gen_accuracy = reduce_tensor(gen_accuracy.data)
            train_gen_accuracy += reduced_gen_accuracy.item()/len(train_loader)
            
            if train_discriminator:
                reduced_disc_loss = reduce_tensor(disc_loss.data)
                train_disc_loss += reduced_disc_loss.item()/len(train_loader)
                reduced_disc_accuracy_real = reduce_tensor(disc_accuracy_real.data)
                train_disc_accuracy_real += reduced_disc_accuracy_real.item()/len(train_loader)
                reduced_disc_accuracy_fake = reduce_tensor(disc_accuracy_fake.data)
                train_disc_accuracy_fake += reduced_disc_accuracy_fake.item()/len(train_loader)

        else:
            train_loss += gen_loss.item() / len(train_loader)
            train_L1_loss += L1_loss.item() / len(train_loader)
#             train_content_loss += content_loss.item() / len(train_loader)
#             train_style_loss += style_loss.item() / len(train_loader)
#             train_dtime_loss += dtime_loss.item() / len(train_loader)
            train_dfreq_loss += dfreq_loss.item() / len(train_loader)
            train_disc_loss += disc_loss.item() / len(train_loader)
            train_disc_accuracy += disc_accuracy.item() / len(train_loader)
            train_gen_accuracy += gen_accuracy.item() / len(train_loader)
            
#         CQ_diff,SQ_diff = metric(pred.cpu().detach().numpy(),y_train.cpu().detach().numpy(),hop_length,n_fft)
#         if mode == 'ddp':
#             CQ_diff = reduce_tensor(torch.Tensor([CQ_diff]).cuda())
#             SQ_diff = reduce_tensor(torch.Tensor([SQ_diff]).cuda())
#         train_CQ_diff += CQ_diff.item()/len(train_loader)
#         train_SQ_diff += SQ_diff.item()/len(train_loader)
        
    val_loss = 0.
    val_L1_loss = 0.
    val_dtime_loss = 0.
    val_dfreq_loss = 0.
    val_disc_loss = 0.
    val_disc_accuracy_real = 0.
    val_disc_accuracy_fake = 0.
    val_gen_accuracy = 0.
#     val_content_loss = 0.
#     val_style_loss = 0.
#     valid_CQ_diff_avg =0
#     valid_CQ_diff_std = 0
#     valid_SQ_diff_avg = 0
#     valid_SQ_diff_std = 0
    
    saving_image_input = []
    saving_image_pred = []
    saving_image_true = []
    
    model.eval()
    discriminator.eval()

    for idx,(_x,_y) in enumerate(tqdm(valid_loader,disable=(verbose==0))):
        x_val,y_val = _x.cuda(),_y.cuda()
        
        with torch.no_grad():
            pred = model(x_val)

            ## ===================== Generator ===================== #
            L1_loss = L1_criterion(pred,y_val)
            dfreq_loss = L1_criterion(pred[:,:,1:,:]-pred[:,:,:-1,:],y_val[:,:,1:,:]-y_val[:,:,:-1,:])
#             L2_loss = MSE_criterion(pred,y_val)
#             dtime_loss = MSE_criterion(pred[:,:,:,1:]-pred[:,:,:,:-1],y_val[:,:,:,1:]-y_val[:,:,:,:-1])
#             dfreq_loss = MSE_criterion(pred[:,:,1:,:]-pred[:,:,:-1,:],y_val[:,:,1:,:]-y_val[:,:,:-1,:])
    #         content_loss = MSE_criterion(pred_feature[1],true_feature[1]) ##relu2_2
    #         style_loss = 0.
    #         for ft_pred,gm_s in zip(pred_feature,gram_style):
    #             gm_y = gram_matrix(ft_pred)
    #             style_loss += MSE_criterion(gm_y,gm_s)

    #         loss = 0.2*L2_loss + 0.2*dtime_loss + 0.2*dfreq_loss + 0.2*content_loss + 0.2*style_loss
#             recon_loss = 0.34*L2_loss + 0.33*dtime_loss + 0.33*dfreq_loss
            recon_loss = 0.5*L1_loss + 0.5*dfreq_loss
            
            disc_input_fake = torch.cat([x_val,pred],dim=1)
            disc_output_fake = discriminator(disc_input_fake)
            disc_label_real = torch.ones_like(disc_output_fake)
            B,_,W,H = disc_label_real.shape
            gen_loss = BCE_criterion(disc_output_fake,disc_label_real) + loss_ratio*recon_loss
            gen_accuracy = torch.round(torch.sigmoid(disc_output_fake)).eq(disc_label_real).sum().type(torch.cuda.FloatTensor)/(B*W*H) ##얼마나 잘 속이냐
            
            ## ===================== discriminator ================= #
            disc_input_fake = torch.cat([x_val,pred.detach()],dim=1)
            disc_input_real = torch.cat([x_val,y_val],dim=1)
            disc_output_fake = discriminator(disc_input_fake)
            disc_output_real = discriminator(disc_input_real)
            disc_label_real = Variable(torch.ones_like(disc_output_real),requires_grad=False)
            disc_label_fake = Variable(torch.zeros_like(disc_output_fake),requires_grad=False)

            disc_loss = (BCE_criterion(disc_output_real,disc_label_real) + BCE_criterion(disc_output_fake,disc_label_fake))*0.5

            B,_,W,H = disc_label_fake.shape
            disc_accuracy_real = torch.round(torch.sigmoid(disc_output_real)).eq(disc_label_real).sum().type(torch.cuda.FloatTensor)/(B*W*H) 
            disc_accuracy_fake = torch.round(torch.sigmoid(disc_output_fake)).eq(disc_label_fake).sum().type(torch.cuda.FloatTensor)/(B*W*H)
            
            saving_image_input.append(x_val)
            saving_image_pred.append(pred)
            saving_image_true.append(y_val)
            
        if mode == 'ddp':
            reduced_loss = reduce_tensor(gen_loss.data)
            val_loss += reduced_loss.item()/len(valid_loader)
            
            reduced_L1_loss = reduce_tensor(L1_loss.data)
            val_L1_loss += reduced_L1_loss.item() / len(valid_loader)
#             reduced_dtime_loss = reduce_tensor(dtime_loss.data)
#             val_dtime_loss += reduced_dtime_loss.item() / len(valid_loader)
            reduced_dfreq_loss = reduce_tensor(dfreq_loss.data)
            val_dfreq_loss += reduced_dfreq_loss.item() / len(valid_loader)
            reduced_disc_loss = reduce_tensor(disc_loss.data)
            val_disc_loss += reduced_disc_loss.item()/len(valid_loader)
            reduced_disc_accuracy_real = reduce_tensor(disc_accuracy_real.data)
            val_disc_accuracy_real += reduced_disc_accuracy_real.item()/len(valid_loader)
            reduced_disc_accuracy_fake = reduce_tensor(disc_accuracy_fake.data)
            val_disc_accuracy_fake += reduced_disc_accuracy_fake.item()/len(valid_loader)
            
            reduced_gen_accuracy = reduce_tensor(gen_accuracy.data)
            val_gen_accuracy += reduced_gen_accuracy.item()/len(valid_loader)
        else:
            val_loss += gen_loss.item()/len(valid_loader)
            val_loss += L1_loss.item()/len(valid_loader)
#             val_content_loss += content_loss.item() / len(valid_loader)
#             val_style_loss += style_loss.item() / len(valid_loader)
#             val_dtime_loss += dtime_loss.item() / len(valid_loader)
            val_dfreq_loss += dfreq_loss.item() / len(valid_loader)
            val_disc_loss += disc_loss.item() / len(valid_loader)
            val_disc_accuracy += disc_accuracy.item() / len(valid_loader)
            val_gen_accuracy += gen_accuracy.item() / len(valid_loader)

#         pred_egg = pred_egg.cpu().detach().numpy()
#         y_val = y_val.cpu().detach().numpy()
        
#         CQ_diff_avg,SQ_diff_avg,CQ_diff_std,SQ_diff_std = metric_egg(pred_egg,y_val,hop_length,n_fft,n_frame)
#         if mode == 'ddp':
#             CQ_diff_avg = reduce_tensor(torch.Tensor([CQ_diff_avg]).cuda())
#             CQ_diff_std = reduce_tensor(torch.Tensor([CQ_diff_std]).cuda())
#             SQ_diff_avg = reduce_tensor(torch.Tensor([SQ_diff_avg]).cuda())
#             SQ_diff_std = reduce_tensor(torch.Tensor([SQ_diff_std]).cuda())
#         valid_CQ_diff_avg += CQ_diff_avg.item()/len(valid_loader)
#         valid_SQ_diff_avg += SQ_diff_avg.item()/len(valid_loader)
#         valid_CQ_diff_std += CQ_diff_std.item()/len(valid_loader)
#         valid_SQ_diff_std += SQ_diff_std.item()/len(valid_loader)
    
    scheduler.step()
    scheduler_d.step()
    if verbose and not is_test:
#         if val_loss < best_val:
        torch.save(model.state_dict(), os.path.join(save_path,'generator_best_%d.pth'%epoch))
        torch.save(discriminator.state_dict(), os.path.join(save_path,'discriminator_best_%d.pth'%epoch))
#         best_val = val_loss
        
        saving_image_input = torch.cat(saving_image_input,dim=0)
        saving_image_pred = torch.cat(saving_image_pred,dim=0)
        saving_image_true = torch.cat(saving_image_true,dim=0)
        
        saving_image_input = torch.cat([saving_image_input,saving_image_input,saving_image_input],dim=1)
        saving_image_pred = torch.cat([saving_image_pred,saving_image_pred,saving_image_pred],dim=1)
        saving_image_true = torch.cat([saving_image_true,saving_image_true,saving_image_true],dim=1)
        
        saving_image_pred = make_grid(saving_image_pred, normalize=True, scale_each=True)
        saving_image_input = make_grid(saving_image_input, normalize=True, scale_each=True)
        saving_image_true = make_grid(saving_image_true, normalize=True, scale_each=True)
        if epoch ==0:
            writer.add_image('img_input/mel', saving_image_input,epoch)
            writer.add_image('img_true/mel', saving_image_true, epoch)
        writer.add_image('img_pred/mel', saving_image_pred, epoch)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val',val_loss,epoch)
        writer.add_scalar('pixel_L1_Loss/train', train_L1_loss, epoch)
        writer.add_scalar('pixel_L1_Loss/val',val_L1_loss,epoch)
#         writer.add_scalar('derivative Time_L1_Loss/train', train_dtime_loss, epoch)
#         writer.add_scalar('derivative Time_L1_Loss/val',val_dtime_loss,epoch)
        writer.add_scalar('derivative Freq_L1_Loss/train', train_dfreq_loss, epoch)
        writer.add_scalar('derivative Freq_L1_Loss/val',val_dfreq_loss,epoch)
        
        if train_discriminator:
            writer.add_scalar('discriminator_Loss/train',train_disc_loss,epoch)
            writer.add_scalar('discriminator_accuracy_real/train',train_disc_accuracy_real,epoch)
            writer.add_scalar('discriminator_accuracy_fake/train',train_disc_accuracy_fake,epoch)
        
        writer.add_scalar('discriminator_Loss/val',val_disc_loss,epoch)
        writer.add_scalar('discriminator_accuracy_real/val',val_disc_accuracy_real,epoch)
        writer.add_scalar('discriminator_accuracy_fake/val',val_disc_accuracy_fake,epoch)
        
        
        writer.add_scalar('generator_accuracy/train',train_gen_accuracy,epoch)
        writer.add_scalar('generator_accuracy/val',val_gen_accuracy,epoch)
#         writer.add_scalar('pixel_content_Loss/train', train_content_loss, epoch)
#         writer.add_scalar('pixel_content_Loss/val',val_content_loss,epoch)
#         writer.add_scalar('pixel_style_Loss/train', train_style_loss, epoch)
#         writer.add_scalar('pixel_style_Loss/val',val_style_loss,epoch)
#         writer.add_scalar('CQ_diff_avg/val',valid_CQ_diff_avg)
#         writer.add_scalar('CQ_diff_std/val',valid_CQ_diff_std)
#         writer.add_scalar('SQ_diff_avg/val',valid_SQ_diff_avg)
#         writer.add_scalar('SQ_diff_std/val',valid_SQ_diff_std)
#     logging("Epoch [%d]/[%d] train_loss %.6f valid_loss %.6f valid_CQ_diff_avg %.6f valid_CQ_diff_std %.6f valid_SQ_diff_avg %.6fvalid_SQ_diff_std %.6f duration : %.4f"%
#         (epoch,n_epoch,train_loss,val_loss,valid_CQ_diff_avg,valid_CQ_diff_std,valid_SQ_diff_avg,valid_SQ_diff_std,time.time()-st))
    logging("Epoch [%d]/[%d] train_loss %.6f valid_loss %.6f train_L1_loss %.6f valid_L1_loss %.6f train_gen_accuracy %.6f val_gen_accuracy %.6f train_disc_loss %.6f valid_disc_loss %.6f train_disc_accuracy_real %.6f valid_disc_accuracy_real %.6f train_disc_accuracy_fake %.6f valid_disc_accuracy_fake %.6f duration : %.4f"%(epoch,n_epoch,train_loss,val_loss,train_L1_loss,val_L1_loss,train_gen_accuracy,val_gen_accuracy,train_disc_loss,val_disc_loss,train_disc_accuracy_real,val_disc_accuracy_real, train_disc_accuracy_fake,val_disc_accuracy_fake, time.time()-st))
    
    if train_discriminator == True and val_disc_loss < 0.4 :
        train_discriminator = False
        logging("============Stop training discriminator==========")
    if train_discriminator == False and val_disc_loss > 0.5 :
        train_discriminator = True
        logging("============Restart training discriminator============")
    
if verbose and not is_test:
#     log.close()
    writer.close()
logging("[!] training end")

