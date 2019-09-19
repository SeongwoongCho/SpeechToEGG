import csv
import numpy as np
import time
from tqdm import tqdm

import torch
from torch import nn
from torch.utils import data
from torch.optim.lr_scheduler import StepLR

from utils import *
from dataloader import *
from model import Unet,Resv2Unet
from apex import amp

def custom_aug(x):
    db =np.random.uniform(low=-5,high=45)
    p = np.random.uniform()
    if p<0.33:
        return add_whitenoise(x,db)
    elif p<0.66:
        return add_pinknoise(x,db)
    else:
        return add_brownnoise(x,db)

seed_everything(42)
###Hyper parameters

save_path = './models/Resv2Unet_Many/'
os.makedirs(save_path,exist_ok=True)
n_epoch = 200
batch_size =8192
n_frame = 576
window = int(n_frame*1.1)
step = int(n_frame/3)
learning_rate = 1e-2

print("[*] load data ...")
st = time.time()
train_X,train_y,val_X,val_y = load_datas(n_frame,window,step)

print(train_X.shape)
print(val_X.shape)

train_dataset = Dataset(train_X,train_y,n_frame = n_frame, is_train = True, aug = custom_aug)
valid_dataset = Dataset(val_X,val_y,n_frame = n_frame, is_train = False)
train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               num_workers=40,
                               shuffle=True)
valid_loader = data.DataLoader(dataset=valid_dataset,
                               batch_size=batch_size,
                               num_workers=40,
                              shuffle=False)

print("Load duration : {}".format(time.time()-st))
print("[!] load data end")

model = Resv2Unet(nlayers = 6, nefilters = 15)
model.cuda()

criterion = nn.MSELoss()
criterion.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

opt_level = 'O1'
model,optimizer = amp.initialize(model,optimizer,opt_level = opt_level)
scheduler = StepLR(optimizer,step_size=80,gamma = 0.3)

model = nn.DataParallel(model)


print("[*] training ...")

log = open(os.path.join(save_path,'log.csv'), 'w', encoding='utf-8', newline='')
log_writer = csv.writer(log)

best_val_loss = 100
for epoch in range(n_epoch):
    avg_loss = 0.
    optimizer.zero_grad()
    model.train()
    for idx,(_x,_y) in enumerate(tqdm(train_loader)):
        x_train,y_train = _x.float().cuda(),_y.float().cuda()
        pred = model(x_train)

        loss = criterion(pred,y_train)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
#         loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        avg_loss += loss.item() / len(train_loader)
        
    val_loss = 0.
    model.eval()
    with torch.no_grad():
        for idx,(_x,_y) in enumerate(tqdm(valid_loader)):
            x_val,y_val = _x.float().cuda(),_y.float().cuda()
            pred = model(x_val)
            loss= criterion(pred,y_val)
            val_loss += loss.item()/len(valid_loader)
    scheduler.step()
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), os.path.join(save_path,'best_val.pth'))
        best_val_loss = val_loss
    
    log_writer.writerow([epoch,avg_loss,val_loss])
    log.flush()
    print("Epoch [%d]/[%d] train_loss %.6f valid_loss %.6f "%
          (epoch,n_epoch,avg_loss,val_loss))

log.close()
print("[!] training end")