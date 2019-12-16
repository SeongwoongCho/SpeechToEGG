import torch
# import encoding ## pip install torch-encoding . For synchnonized Batch norm in pytorch 1.0.0
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from attention_model import EncoderRNN,Attn,BahdanauAttnDecoderRNN
######################################################
####################Trash Can ########################
######################################################
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
            x = F.interpolate(x,scale_factor=2,mode='linear')
            x = torch.cat([x,encoder[self.num_layers - i - 1]],dim=1)
            x = self.decoder[i](x)
            x = self.dbatch[i](x)
            x = F.leaky_relu(x,0.1)
        x = torch.cat([x,input],dim=1)

        x = self.out(x)
        x = x.squeeze(1).unsqueeze(-1)
        return x
    
######################################################
######################################################
######################################################
######################################################

def swish(x):
    return x * torch.sigmoid(x)

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
    def __init__(self, in_channels,channels,ks, stride=1,act = 'relu',sample=True):
        super(SEBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels,channels, ks, stride=stride, padding=ks//2,bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels,channels, ks, stride=stride, padding=ks//2,bias=False)
        self.se = SELayer(channels, reduction=4)
        self.act = act
        self.sample = None
        if sample:
            self.sample = nn.Conv1d(in_channels,channels, 1, stride, bias=False)
    def forward(self, x):
        residual = x
        if self.sample:
            residual = self.sample(residual)    
        x = self.bn1(x)
        x = F.relu(x) if self.act == 'relu' else swish(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x) if self.act == 'relu' else swish(x)
        x = self.conv2(x)
        x = self.se(x)
        return x + residual
      
class Resv2Unet(nn.Module):
    def __init__(self, nlayers = 14,nefilters=24,filter_size = 9,merge_filter_size = 5,act = 'swish',extract = False):
        super(Resv2Unet, self).__init__()

        self.num_layers = nlayers
        self.nefilters = nefilters
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.extract = extract
        echannelin = [nefilters] + [(i + 1) * nefilters for i in range(nlayers - 1)]
        echannelout = [(i + 1) * nefilters for i in range(nlayers)]
        dchannelout = echannelout[::-1]
        dchannelin = [dchannelout[0] * 2] + [(i) * nefilters + (i - 1) * nefilters for i in range(nlayers, 1, -1)]
        for i in range(self.num_layers):
            self.encoder.append(SEBasicBlock(echannelin[i],echannelout[i],filter_size,act=act,sample=True))
            self.decoder.append(SEBasicBlock(dchannelin[i], dchannelout[i],merge_filter_size,act=act,sample=True))
        
        self.first = nn.Conv1d(1,nefilters,filter_size,padding=filter_size//2)
        self.middle = SEBasicBlock(echannelout[-1],echannelout[-1],filter_size)
#         self.outbatch = encoding.nn.BatchNorm1d(nefilters+1)
        self.outbatch = nn.BatchNorm1d(nefilters+1)
        
        if not extract:
            self.out = nn.Sequential(
                nn.Conv1d(nefilters + 1, 1, 1),
                nn.Tanh()
            )
        
    def forward(self,x):
        encoder = list()
        
        x = x.squeeze(-1).unsqueeze(1) ## bs,1,n_frame
        
        input = x
        x = self.first(x)
        for i in range(self.num_layers):
            x = self.encoder[i](x)
            encoder.append(x)
            x = x[:, :, ::2]
            
        x = self.middle(x)
        for i in range(self.num_layers):
            x = F.interpolate(x,scale_factor=2,mode='linear')
            x = torch.cat([x,encoder[self.num_layers - i - 1]],dim=1)
            x = self.decoder[i](x)
        x = torch.cat([x,input],dim=1)
        
        x = self.outbatch(x)
        x = F.leaky_relu(x)
        if self.extract:
            return x.transpose(1,2)  ## bs,n_frame,nefilters+1
        
        x = self.out(x)
        return x.squeeze(1).unsqueeze(-1)


class UEDSimple(nn.Module): ## Unet Encoder Decoder Simple
    def __init__(self, nlayers = 5,nefilters=32,filter_size = 15,merge_filter_size = 5,hidden_size = 10, num_layers = 1):
        super(UEDSimple, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.UBlock = Resv2Unet(nlayers = nlayers,nefilters=nefilters,filter_size = filter_size,merge_filter_size = merge_filter_size,act='swish',extract = True)

        self.encoder = nn.GRU(nefilters+1,hidden_size,num_layers,batch_first=True,bidirectional=False)
        self.decoder = nn.GRU(hidden_size,hidden_size,num_layers,batch_first=True,bidirectional=False)
        self.out = nn.Sequential(
            nn.Linear(hidden_size,1),
            nn.Tanh()
        )
        
    def forward(self,x):
        """
        x : B,1,T
        inputs : B,T,nefilters+1
        """
        
        inputs = self.UBlock(x)
        T = inputs.shape[1]

#         output, (hidden,cell) = self.encoder(inputs)
        output, hidden = self.encoder(inputs)
        ## output : B,T,hidden_size
        
        outputs = []
        for t in range(T):
#             output,(hidden,cell) = self.decoder(output[:,-1:,:],(hidden,cell))
            output,hidden = self.decoder(output[:,-1:,:],hidden)
            
            ## output : B,1,hidden_size
            outputs.append(output)
        outputs = torch.stack(outputs) #T,B,1,H
        outputs = outputs.squeeze(2).transpose(0,1)
        # Many-to-Many
        outputs = self.out(outputs) # B,T,H -> B,T,1

        return outputs
    
class ULSTM(nn.Module):
    def __init__(self, nlayers = 5,nefilters=32,filter_size = 15,merge_filter_size = 5,hidden_size = 10, num_layers = 1):
        super(ULSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.UBlock = Resv2Unet(nlayers = nlayers,nefilters=nefilters,filter_size = filter_size,merge_filter_size = merge_filter_size,act='swish',extract = True)

        self.gru = nn.GRU(nefilters+1,hidden_size,num_layers,batch_first=True,bidirectional=True)
        self.rnn_out = nn.Sequential(
            nn.Linear(2*hidden_size,1),
            nn.Tanh()
        )
    def forward(self,x):
        """
        x : B,1,T
        inputs : B,T,nefilters+1
        """
        
        inputs = self.UBlock(x)
        self.gru.flatten_parameters()
        output, hidden = self.gru(inputs)
        ## output : B,T,hidden_size
        output = self.rnn_out(output) # B,T,H -> B,T,1

        return output

class UEDAttention(nn.Module):
    def __init__(self,nlayers = 5,nefilters=32,filter_size = 15,merge_filter_size = 5,hidden_size = 64,num_layers = 1,dropout_e = 0.5, dropout_p = 0.1):
        super(UEDAttention, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_directions = 1
        self.UBlock = Resv2Unet(nlayers = nlayers,nefilters=nefilters,filter_size = filter_size,merge_filter_size = merge_filter_size,act='swish',extract = True)

        self.encoder = EncoderRNN(nefilters+1,hidden_size,num_layers,dropout_e)
        self.decoder = BahdanauAttnDecoderRNN(2*hidden_size,num_layers,dropout_p)
        self.final_out = nn.Sequential(
            nn.Linear(2*hidden_size,1),
            nn.Tanh()
        )
    def get_go_frame(self,batch_size):
        z = np.zeros((batch_size,1,2*self.hidden_size))
        return torch.Tensor(z).cuda()
    
    def forward(self,x):
        """
        x : B,1,T
        inputs : B,T,nefilters+1
        """
        inputs = self.UBlock(x)
        
        B = inputs.shape[0]
        T = inputs.shape[1]
        
        encoder_outputs,hidden = self.encoder(inputs) ##(B,T,2H), (2N,B,H)
#         print("encoder hidden:",hidden.shape)
        hidden = hidden.view(self.num_layers,B,-1) ## (N,B,2H)
        decoder_outputs = []
        decoder_input = self.get_go_frame(B)
        for i in range(T):
            decoder_input, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            decoder_outputs.append(decoder_input)
        decoder_outputs = torch.stack(decoder_outputs) #T,B,H
        decoder_outputs = decoder_outputs.transpose(0,1)
        # Many-to-Many
        decoder_outputs = self.final_out(decoder_outputs) # B,T,H -> B,T,1
        return decoder_outputs

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features
    
class _DenseBlock(nn.Module):
    __constants__ = ['layers']

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        self.layers = nn.ModuleDict()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.layers['denselayer%d' % (i + 1)] = layer

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.layers.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class _MDenseNet(nn.Module):
    def __init__(self,scale):
        super(MDenseNet,self).__init__()
        
    def forward(self,input):
        return input
    
class MMDenseNet(nn.Module):
    def __init__(self):
        super(MMDenseNet,self).__init__()
        
        self.lowNet = _MDenseNet()
        self.highNet = _MDenseNet()
        self.fullNet = _MDenseNet()
        
        self.out = nn.Sequential(
            _DenseBlock(),
            nn.Conv2d()
        )
        
    def forward(self,input):
        B,F,C = input.shape
        low_input = input[:,:F//2,:]
        high_input = input[:,F//2:,:]
        
        lowNet_output = self.lowNet(low_input)
        highNet_output = self.highNet(high_input)
        full_output = self.fullNet(input)
        
        output = torch.cat([lowNet_output,highNet_output],1) ##Frequency 방향
        output = torch.cat([output,full_output],2) ## Channel 방향
        output = self.out(output)
        
        return output