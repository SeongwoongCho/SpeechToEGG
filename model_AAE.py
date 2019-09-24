import torch
import torch.nn as nn
from torch.nn import functional as F

class FCEncoder(nn.Module):
    def __init__(self):
        super(FCEncoder, self).__init__()
        self.module_list = nn.ModuleList()
        self.batches = nn.ModuleList()
        self.drop_rate = 0.5
        tmp = [192,144,121,100,81,64,49]
        for i in range(6):
            self.module_list.append(nn.Linear(tmp[i],tmp[i+1]))
            self.batches.append(nn.BatchNorm1d(num_features=tmp[i+1]))
    def forward(self,x):
        x = x.squeeze(-1)
        for batch,lin in zip(self.batches,self.module_list):
            x = F.relu(batch(lin(x)))
            x = F.dropout(x,self.drop_rate,self.training)
        x = x.unsqueeze(-1)
        return x
        
class FCDecoder(nn.Module):
    def __init__(self):
        super(FCDecoder, self).__init__()
        self.module_list = nn.ModuleList()
        self.batches = nn.ModuleList()
        self.drop_rate = 0.5
        tmp = [49,64,81,100,121,144,192]
        for i in range(6):
            self.module_list.append(nn.Linear(tmp[i],tmp[i+1]))
            self.batches.append(nn.BatchNorm1d(num_features=tmp[i+1]))
    def forward(self,x):
        x = x.squeeze(-1)
        for i,(batch,lin) in enumerate(zip(self.batches,self.module_list)):
            if i!=len(self.module_list)-1:
                x = F.relu(batch(lin(x)))
                x = F.dropout(x,self.drop_rate,self.training)
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
        self.drop_rate = 0.3
        self.lin1 = nn.Linear(49,16)
        self.lin2 = nn.Linear(16,1)
        
    def forward(self,x):
        x = x.squeeze(-1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x,self.drop_rate,self.training)
        x = self.lin2(x)
        x = x.unsqueeze(-1)
        return x