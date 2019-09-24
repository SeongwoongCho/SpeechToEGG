import torch
import torch.nn as nn
from torch.nn import functional as F

class BottleneckV2(nn.Module):
    def __init__(self, in_channels,channels,ks, stride=1,upsample=False,downsample=False):
        super(BottleneckV2, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, channels//4, 1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels//4)
        self.conv2 = nn.Conv1d(channels//4,channels//4, ks, stride=stride, padding=ks//2,bias=False)
        self.bn3 = nn.BatchNorm1d(channels//4)
        self.conv3 = nn.Conv1d(channels//4,channels, 1, stride=1, bias=False)
        if downsample:
            self.downsample = nn.Conv1d(in_channels,channels, 1, stride, bias=False)
        else:
            self.downsample = None
        if upsample:
            self.upsample = nn.Conv1d(in_channels, channels, 1, stride, bias=False)
        else:
            self.upsample = None

    def forward(self,x):
        residual = x
        x = self.bn1(x)
        x = F.relu(x)
        if self.downsample:residual = self.downsample(x)
        if self.upsample:residual = self.upsample(x)

        x = self.conv1(x)

        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)

        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv3(x)
        #print(x.shape,residual.shape)
        return x + residual


class BasicBlockV2(nn.Module):
    def __init__(self, in_channels,channels,ks, stride=1,upsample=False,downsample=False):
        super(BasicBlockV2, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels,channels, ks, stride=stride, padding=ks//2,bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels,channels, ks, stride=stride, padding=ks//2,bias=False)
        if downsample:
            self.downsample = nn.Conv1d(in_channels,channels, 1, stride, bias=False)
        else:
            self.downsample = None
        if upsample:
            self.upsample = nn.Conv1d(in_channels, channels, 1, stride, bias=False)
        else:
            self.upsample = None

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = F.relu(x)
        if self.downsample:
            residual = self.downsample(x)
        if self.upsample:
            residual = self.upsample(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + residual

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class SEBasicBlock(nn.Module):
    def __init__(self, in_channels,channels,ks, stride=1,upsample=False,downsample=False):
        super(SEBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels,channels, ks, stride=stride, padding=ks//2,bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels,channels, ks, stride=stride, padding=ks//2,bias=False)
        self.se = SELayer(channels, reduction=4)
        if downsample:
            self.downsample = nn.Conv1d(in_channels,channels, 1, stride, bias=False)
        else:
            self.downsample = None
        if upsample:
            self.upsample = nn.Conv1d(in_channels, channels, 1, stride, bias=False)
        else:
            self.upsample = None

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = F.relu(x)
        if self.downsample:
            residual = self.downsample(x)
        if self.upsample:
            residual = self.upsample(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.se(x)
        return x + residual
    
class Unet(nn.Module):
    def __init__(self,nlayers = 12,nefilters=24, filter_size = 15, merge_filter_size = 5):
        super(Unet, self).__init__()
        self.num_layers = nlayers
        self.nefilters = nefilters
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.ebatch = nn.ModuleList()
        self.dbatch = nn.ModuleList()
        echannelin = [1] + [(i + 1) * nefilters for i in range(nlayers-1)]
        echannelout = [(i + 1) * nefilters for i in range(nlayers)]
        dchannelout = echannelout[::-1]
        dchannelin = [dchannelout[0]*2]+[(i) * nefilters + (i - 1) * nefilters for i in range(nlayers,1,-1)]
        for i in range(self.num_layers):
            self.encoder.append(nn.Conv1d(echannelin[i],echannelout[i],filter_size,padding=filter_size//2))
            self.decoder.append(nn.Conv1d(dchannelin[i],dchannelout[i],merge_filter_size,padding=merge_filter_size//2))
            self.ebatch.append(nn.BatchNorm1d(echannelout[i]))
            self.dbatch.append(nn.BatchNorm1d(dchannelout[i]))

        self.middle = nn.Sequential(
            nn.Conv1d(echannelout[-1],echannelout[-1],filter_size,padding=filter_size//2),
            nn.BatchNorm1d(echannelout[-1]),
            nn.LeakyReLU(0.1)
        )
        self.out = nn.Sequential(
            nn.Conv1d(nefilters + 1, 1, 1),
            nn.Tanh()
        )
    def forward(self,x):
        ## x = [bs,n_frame,1]
        encoder = list()
        x = x.squeeze(-1).unsqueeze(1)
        
        # x = [bs,1,n_frame]
        
        input = x
        for i in range(self.num_layers):
            x = self.encoder[i](x)
            x = self.ebatch[i](x)
            x = F.leaky_relu(x,0.1)
            encoder.append(x)
            x = x[:,:,::2]

        x = self.middle(x)

        for i in range(self.num_layers):
            x = F.upsample(x,scale_factor=2,mode='linear')
            x = torch.cat([x,encoder[self.num_layers - i - 1]],dim=1)
            x = self.decoder[i](x)
            x = self.dbatch[i](x)
            x = F.leaky_relu(x,0.1)
        x = torch.cat([x,input],dim=1)

        x = self.out(x)
        x = x.squeeze(1).unsqueeze(-1)
        return x

class Resv2Unet(nn.Module):
    def __init__(self, nlayers = 14,nefilters=24,filter_size = 9,merge_filter_size = 5):
        super(Resv2Unet, self).__init__()

        self.num_layers = nlayers
        self.nefilters = nefilters
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        #self.downsample=nn.ModuleList()
        #self.upsample = nn.ModuleList()
        #echannelin = [24, 24, 48, 72, 96, 124, 144, 168, 192, 216, 240, 264]
        #echannelout = [24, 48, 72, 96, 124, 144, 168, 192, 216, 240, 264, 288]
        echannelin = [24] + [(i + 1) * nefilters for i in range(nlayers - 1)]
        echannelout = [(i + 1) * nefilters for i in range(nlayers)]
        dchannelout = echannelout[::-1]
        upsamplec = [dchannelout[0]] + [(i) * nefilters for i in range(nlayers, 1, -1)]
        dchannelin = [dchannelout[0] * 2] + [(i) * nefilters + (i - 1) * nefilters for i in range(nlayers, 1, -1)]
        for i in range(self.num_layers):
            self.encoder.append(SEBasicBlock(echannelin[i],echannelout[i],filter_size,downsample=True))
            #self.downsample.append(SEBasicBlock(echannelout[i],echannelout[i],filter_size))
            #self.upsample.append(SEBasicBlock(upsamplec[i], upsamplec[i],merge_filter_size))
            self.decoder.append(SEBasicBlock(dchannelin[i], dchannelout[i],merge_filter_size,upsample=True))
        self.first = nn.Conv1d(1,24,filter_size,padding=filter_size//2)
        self.middle = SEBasicBlock(echannelout[-1],echannelout[-1],filter_size)
        self.outbatch = nn.BatchNorm1d(nefilters+1)
        self.out = nn.Sequential(
            nn.Conv1d(nefilters + 1, 1, 1),
            nn.Tanh()
        )
    def forward(self,x):
        encoder = list()
        
        x = x.squeeze(-1).unsqueeze(1)
        
        input = x
        x = self.first(x)
        for i in range(self.num_layers):
            x = self.encoder[i](x)
            encoder.append(x)
            x = x[:, :, ::2]
            #x = self.downsample[i](x)
        x = self.middle(x)
        for i in range(self.num_layers):
            x = F.upsample(x,scale_factor=2,mode='linear')
            #x = self.upsample[i](x)
            x = torch.cat([x,encoder[self.num_layers - i - 1]],dim=1)
            x = self.decoder[i](x)
        x = torch.cat([x,input],dim=1)
        x = self.outbatch(x)
        x = F.leaky_relu(x)
        x = self.out(x)

        x = x.squeeze(1).unsqueeze(-1)
        return x
    