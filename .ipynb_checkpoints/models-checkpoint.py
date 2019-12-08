import torch
# import encoding ## pip install torch-encoding . For synchnonized Batch norm in pytorch 1.0.0
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch import Tensor
from torchvision import models
from collections import namedtuple
from torch.nn.utils import spectral_norm

def conv3x3(in_channel, out_channel, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class CBAMBlock(nn.Module):
    def __init__(self,in_channel):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_channel)
        self.sa = SpatialAttention()
    def forward(self,inp):
        out = inp*self.ca(inp)
        out = out*self.sa(out)
        return out
        
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
        self.num_input_features = num_input_features
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
        
class _MDenseNet_STEM(nn.Module):
    def __init__(self,first_channel=32,first_kernel = (3,3),scale=3,kl = [],drop_rate = 0.1,bn_size=4,attention = None):
        super(_MDenseNet_STEM,self).__init__()
        self.first_channel = 32
        self.first_kernel = first_kernel
        self.scale = scale
        self.kl = kl
        
        self.first_conv = nn.Conv2d(1,first_channel,first_kernel)
        self.downsample_layer = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.upsample_layers = nn.ModuleList()
        self.dense_padding = nn.ModuleList()
        self.dense_layers = nn.ModuleList()
        
        if attention:
            assert attention == 'CBAM'
            self.attention_layers = nn.ModuleList()
        
        self.channels = [self.first_channel]
        ## [_,d1,...,ds,ds+1,u1,...,us]
        for k,l in kl[:scale+1]:
            self.dense_layers.append(_DenseBlock( 
                l, self.channels[-1], bn_size, k, drop_rate))
            self.channels.append(self.channels[-1]+k*l)
        
        for i,(k, l) in enumerate(kl[scale+1:]):
            self.upsample_layers.append(nn.ConvTranspose2d(self.channels[-1],self.channels[-1], kernel_size=2, stride=2))
            self.channels.append(self.channels[-1]+self.channels[scale-i])
            self.dense_layers.append(_DenseBlock(
                l, self.channels[-1], bn_size, k, drop_rate))
            self.channels.append(self.channels[-1]+k*l)
            
            if attention:
                self.attention_layers.append(CBAMBlock(self.channels[-1]))
            
    def _pad(self,x,target):
        if x.shape != target.shape:
            padding_1 = target.shape[2] - x.shape[2]
            padding_2 = target.shape[3] - x.shape[3]
        return F.pad(x,(padding_2//2,padding_2 - padding_2//2,padding_1//2,padding_1-padding_1//2),'constant', 0)
    
    def forward(self,input):
        ## stem
        output = self.first_conv(input)
        dense_outputs = []
        
        ## downsample way
        for i in range(self.scale):
            output = self.dense_layers[i](output)
            dense_outputs.append(output)
            output = self.downsample_layer(output) ## downsample

        ## upsample way
        output = self.dense_layers[self.scale](output)
        for i in range(self.scale):
            output = self.upsample_layers[i](output)
            output = self._pad(output,dense_outputs[-(i+1)])
            output = torch.cat([output,dense_outputs[-(i+1)]],dim = 1)
            output = self.dense_layers[self.scale+1+i](output)
            output = self.attention_layers[i](output)
        output = self._pad(output,input)
        return output
    
class MMDenseNet(nn.Module):
    def __init__(self,drop_rate = 0.1,bn_size=4,k1=10,l1=3,k2=None,l2=None,attention = None):
        super(MMDenseNet,self).__init__()
        
        if k2 is None:
            k2 = k1
        if l2 is None:
            l2 = l1
        
        kl_low = [(k1,l1)]*7 ## (14,4)
        kl_high = [(k1,l1)]*7 ## (10,3)
        kl_full = [(k2,l2)]*7 ## (6,2)
        self.lowNet = _MDenseNet_STEM(first_channel=32,first_kernel = (3,3),scale=3,kl = kl_low,drop_rate = drop_rate,bn_size=bn_size,attention = attention)
        self.highNet = _MDenseNet_STEM(first_channel=32,first_kernel = (3,3),scale=3,kl = kl_high,drop_rate = drop_rate,bn_size=bn_size,attention = attention)
        self.fullNet = _MDenseNet_STEM(first_channel=32,first_kernel = (4,3),scale=3,kl = kl_full,drop_rate = drop_rate,bn_size=bn_size,attention = attention)
        
        last_channel = self.lowNet.channels[-1] + self.fullNet.channels[-1]
        self.out = nn.Sequential(
            _DenseBlock( 
                2, last_channel, bn_size, 4, drop_rate),
            nn.ReLU(),
            nn.Conv2d(last_channel+8,1,1),
#             nn.Tanh()
            nn.Sigmoid()
        )
        
    def forward(self,input):
        B,C,Fre,T = input.shape
        low_input = input[:,:,:Fre//2,:]
        high_input = input[:,:,Fre//2:,:]
#         print("input_shape : ",low_input.shape)
        output = torch.cat([self.lowNet(low_input),self.highNet(high_input)],2) ##Frequency 방향
        full_output = self.fullNet(input)
        output = torch.cat([output,full_output],1) ## Channel 방향
        output = self.out(output)
        
        return output


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
#         print(vgg_pretrained_features)
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return h_relu1_2,h_relu2_2,h_relu3_3,h_relu4_3

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        
#         print(vgg_pretrained_features)
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_4 = h
        h = self.slice4(h)
        h_relu4_4 = h
        return h_relu1_2,h_relu2_2,h_relu3_4,h_relu4_4

class customGAN_discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc,drop_rate = 0, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, spectral = True):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(customGAN_discriminator, self).__init__()
        self.drop_rate = drop_rate
        self.stem = nn.Sequential(nn.Conv2d(input_nc,ndf,kernel_size=4, stride = 2,padding=1) if not spectral else spectral_norm(nn.Conv2d(input_nc,ndf,kernel_size=4, stride = 2,padding=1)),
                                  nn.LeakyReLU(0.2),
                                  nn.Dropout2d(p = self.drop_rate),
                                  nn.Conv2d(ndf,2*ndf,kernel_size=4, stride = 2,padding=1) if not spectral else spectral_norm(nn.Conv2d(ndf,2*ndf,kernel_size=4, stride = 2,padding=1)),
                                  nn.LeakyReLU(0.2),
                                  nn.Dropout2d(p = self.drop_rate), 
                                  nn.Conv2d(2*ndf,4*ndf,kernel_size=4, stride = 2,padding=1) if not spectral else spectral_norm(nn.Conv2d(2*ndf,4*ndf,kernel_size=4, stride = 2,padding=1)))
        self.out = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv2d(4*ndf,1,1,1))
    def forward(self, input):
        """Standard forward."""
        out = self.stem(input)
        out = self.out(out)
        return out
    
class patchGAN_discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(patchGAN_discriminator, self).__init__()
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
        use_bias = True
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    
class pixelGAN_discriminator(torch.nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(pixelGAN_discriminator, self).__init__()
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
        use_bias = True
        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

class InterpolateWrapper(torch.nn.Module):
    def __init__(self, model, step=32):
        super().__init__()
        
        self.model = model
        self.step = step
        
    def forward(self, x):
        initial_size = list(x.size()[-2:])
        interpolated_size = [(d // self.step) * self.step for d in initial_size] 
        
        x = torch.nn.functional.interpolate(x, interpolated_size)
        x = self.model(x)
        x = torch.nn.functional.interpolate(x, initial_size)

        return x