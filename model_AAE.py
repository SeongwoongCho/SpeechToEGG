import torch
import torch.nn as nn
from torch.nn import functional as F

class FCEncoder(nn.Module):
    def __init__(self):
        super(FCEncoder, self).__init__()
        self.module_list = nn.ModuleList()
        self.batches = nn.ModuleList()
#         self.drop_rate = 0.5
        tmp = [192,175,125,100]
        for i in range(3):
            self.module_list.append(nn.Linear(tmp[i],tmp[i+1]))
            self.batches.append(nn.BatchNorm1d(num_features=tmp[i+1]))
    def forward(self,x):
        x = x.squeeze(-1)
        for batch,lin in zip(self.batches,self.module_list):
            x = F.relu(batch(lin(x)))
#             x = F.dropout(x,self.drop_rate,self.training)
# TODO : 여기 마지막 단 Tanh넣어서 다시 돌리기
        x = x.unsqueeze(-1)
        return x
        
class FCDecoder(nn.Module):
    def __init__(self):
        super(FCDecoder, self).__init__()
        self.module_list = nn.ModuleList()
        self.batches = nn.ModuleList()
#         self.drop_rate = 0.5
        tmp = [100,125,175,192]
        for i in range(3):
            self.module_list.append(nn.Linear(tmp[i],tmp[i+1]))
            self.batches.append(nn.BatchNorm1d(num_features=tmp[i+1]))
    def forward(self,x):
        x = x.squeeze(-1)
        for i,(batch,lin) in enumerate(zip(self.batches,self.module_list)):
            if i!=len(self.module_list)-1:
                x = F.relu(batch(lin(x)))
#                 x = F.dropout(x,self.drop_rate,self.training)
            else:
                x = F.tanh(batch(lin(x)))
        x = x.unsqueeze(-1)
        return x
    
class FCAE(nn.Module):
    def __init__(self):
        super(FCAE,self).__init__()
        self.encoder = FCEncoder()
        self.decoder = FCDecoder()
    def forward(self,x,extract = False):
        x = self.encoder(x)
        if extract:
            return x
        x = self.decoder(x)
        return x
    
class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super(SimpleDiscriminator,self).__init__()
#         self.drop_rate = 0.3
        self.lin1 = nn.Linear(100,50)
        self.lin2 = nn.Linear(50,25)
        self.lin3 = nn.Linear(25,1)
        
        self.batch1 = nn.BatchNorm1d(50)
        self.batch2 = nn.BatchNorm1d(25)
        self.batch3 = nn.BatchNorm1d(1)
    def forward(self,x):
        x = x.squeeze(-1)
        x = F.relu(self.batch1(self.lin1(x)))
        x = F.relu(self.batch2(self.lin2(x)))
        x = F.sigmoid(self.batch3(self.lin3(x)))
#         x = F.dropout(x,self.drop_rate,self.training)
#         x = F.relu
#         x = F.sigmoid(self.lin2(x))
        x = x.unsqueeze(-1)
        return x