import csv
import numpy as np
import time
from tqdm import tqdm

import torch
from torch import nn
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F

from utils import *
from dataloader import *
from model_AAE import *
from apex import amp
# from apex.parallel import DistributedDataParallel as DDP

import multiprocessing as mp

seed_everything(42)
###Hyper parameters

save_path = './models/AAI/'
os.makedirs(save_path,exist_ok=True)
n_epoch = 100
batch_size = 40000
n_frame = 192
# (1.4 -1)*3 = 1.2 --> Can see all possible datas
window = int(n_frame*1.4 )
step = int(n_frame/2.5)

gen_lr = 2e-3
reg_lr = 2e-3
EPS = 1e-12

print("[*] load data ...")
st = time.time()
train_X,train_y,val_X,val_y = load_datas(n_frame,window,step)

print(train_X.shape)
print(val_X.shape)

train_dataset = Dataset(train_X,train_y,n_frame = n_frame, is_train = True, aug = custom_aug)
valid_dataset = Dataset(val_X,val_y,n_frame = n_frame, is_train = False)
train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               num_workers=mp.cpu_count(),
                               shuffle=True)
valid_loader = data.DataLoader(dataset=valid_dataset,
                               batch_size=batch_size,
                               num_workers=mp.cpu_count(),
                              shuffle=False)

print("Load duration : {}".format(time.time()-st))
print("[!] load data end")

# torch.cuda.set_device(3)
# torch.distributed.init_process_group(backend='nccl',
#                                      init_method='env://',
#                                      world_size=4, rank=1)

EGG_prior = FCAE()
EGG_prior.cuda()
EGG_prior = nn.DataParallel(EGG_prior)
EGG_prior.load_state_dict(torch.load('./models/AAI_EGGAE/best_val-cosloss.pth'))
EGG_prior.eval()

STZ = FCEncoder()
STZ.cuda()
ZTE = FCDecoder()
ZTE.cuda()

DC = SimpleDiscriminator()
DC.cuda()

#encoder/decoder optimizers
STZ_optimizer_gen = torch.optim.Adam(STZ.parameters(), lr = gen_lr)
ZTE_optimizer = torch.optim.Adam(ZTE.parameters(), lr = gen_lr)

#regularizing optimizer

STZ_optimizer_enc = torch.optim.Adam(STZ.parameters(), lr = reg_lr)
DC_optimizer = torch.optim.Adam(DC.parameters(),lr = reg_lr)

# criterion = nn.MSELoss()
criterion = CosineDistanceLoss()
# criterion.cuda()

# opt_level = 'O1'
# assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
# STZ,STZ_optimizer_enc = amp.initialize(STZ,STZ_optimizer_enc,opt_level = opt_level)
# STZ,STZ_optimizer_gen = amp.initialize(STZ,STZ_optimizer_gen,opt_level = opt_level)
# ZTE,ZTE_optimizer = amp.initialize(ZTE,ZTE_optimizer,opt_level = opt_level)
# DC,DC_optimizer = amp.initialize(DC,DC_optimizer,opt_level = opt_level)

# [STZ,ZTE,DC],[STZ_optimizer_enc,STZ_optimizer_gen,ZTE_optimizer,DC_optimizer] = amp.initialize([STZ,ZTE,DC],[STZ_optimizer_enc,STZ_optimizer_gen,ZTE_optimizer,DC_optimizer],opt_level = opt_level)

scheduler_STZ_enc = StepLR(STZ_optimizer_enc,step_size=10,gamma = 0.9)
scheduler_STZ_gen = StepLR(STZ_optimizer_gen,step_size=10,gamma = 0.9)
scheduler_ZTE = StepLR(ZTE_optimizer,step_size=10,gamma = 0.9)
scheduler_DC = StepLR(DC_optimizer,step_size=10,gamma = 0.9)

# STZ = DDP(STZ)
# ZTE = DDP(ZTE)
# DC = DDP(DC)

STZ = nn.DataParallel(STZ)
ZTE = nn.DataParallel(ZTE)
DC = nn.DataParallel(DC)

print("[*] training ...")

log = open(os.path.join(save_path,'log.csv'), 'w', encoding='utf-8', newline='')
log_writer = csv.writer(log)
best_val = 100.
for epoch in range(n_epoch):
    train_recon_loss = 0.
    train_dc_loss = 0.
    train_gen_loss = 0.
    
    for idx,(_x,_y) in enumerate(tqdm(train_loader)):
        #reconstruction loss
        STZ.train()
        ZTE.train()
        DC.train()
        
        STZ.zero_grad()
        ZTE.zero_grad()
        DC.zero_grad()

        x_train,y_train = _x.float().cuda(),_y.float().cuda()
        
        z = STZ(x_train)
        recon = ZTE(z)
        recon_loss = criterion(recon,y_train)
#         with amp.scale_loss(recon_loss, [STZ_optimizer_enc,ZTE_optimizer]) as scaled_loss:
#             scaled_loss.backward()
        recon_loss.backward()
    
        STZ_optimizer_enc.step()
        ZTE_optimizer.step()
        
        #Discriminator
        STZ.eval()
        
        STZ.zero_grad()
        ZTE.zero_grad()
        DC.zero_grad()

        z_real = EGG_prior(y_train,extract = True)
        z_fake = STZ(x_train)
        DC_real = DC(z_real)
        DC_fake = DC(z_fake)
        
        DC_loss = -torch.mean(torch.log(DC_real + EPS) + torch.log(1-DC_fake + EPS))
#         DC_loss = DC_loss.type(torch.cuda.HalfTensor)
        
#         with amp.scale_loss(DC_loss, DC_optimizer) as scaled_loss:
#             scaled_loss.backward()
        DC_loss.backward()
        DC_optimizer.step()
        
        #Generator
        STZ.train()

        STZ.zero_grad()
        ZTE.zero_grad()
        DC.zero_grad()
        
        z_fake = STZ(x_train)
        DC_fake = DC(z_fake)
        
        G_loss = -torch.mean(torch.log(DC_fake + EPS))
#         G_loss = G_loss.type(torch.cuda.HalfTensor)
        
#         with amp.scale_loss(G_loss, STZ_optimizer_reg) as scaled_loss:
#             scaled_loss.backward()
        G_loss.backward()
    
        STZ_optimizer_gen.step()
        
        train_recon_loss += recon_loss.item()/len(train_loader)
        train_dc_loss += DC_loss.item()/len(train_loader)
        train_gen_loss += G_loss.item()/len(train_loader)  
    
    ###validation
    val_recon_loss = 0.
    
    STZ.eval()
    ZTE.eval()
    DC.eval()
    with torch.no_grad():
        for idx,(_x,_y) in enumerate(tqdm(valid_loader)):
            x_val,y_val = _x.float().cuda(),_y.float().cuda()
            z = STZ(x_val)
            recon = ZTE(z)
            recon_loss= torch.mean(torch.acos(F.cosine_similarity(recon,y_val)))
            val_recon_loss+=recon_loss.item()/len(valid_loader)
    
    ## update scheduler, save model, write logs
    scheduler_DC.step()
    scheduler_ZTE.step()
    scheduler_STZ_enc.step()
    scheduler_STZ_gen.step()
    
    if val_recon_loss < best_val:
        torch.save(STZ.state_dict(), os.path.join(save_path,'STZ-cosloss.pth'))
        torch.save(ZTE.state_dict(), os.path.join(save_path,'ZTE-cosloss.pth'))
        torch.save(DC.state_dict(), os.path.join(save_path,'DC-cosloss.pth'))
        best_val = val_recon_loss
    
    log_writer.writerow([epoch,train_recon_loss,train_dc_loss,train_gen_loss,val_recon_loss])
    log.flush()
    print("Epoch [%d]/[%d] train_recon_loss %.6f valid_recon_loss %.6f "%
          (epoch,n_epoch,train_recon_loss,val_recon_loss))

log.close()
print("[!] training end")