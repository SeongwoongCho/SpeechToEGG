from collections import OrderedDict
from .layers import *
from .efficientnet import EfficientNet
import numpy as np
import torch

__all__ = ['EfficientUnet', 'get_efficientunet_b0', 'get_efficientunet_b1', 'get_efficientunet_b2',
           'get_efficientunet_b3', 'get_efficientunet_b4', 'get_efficientunet_b5', 'get_efficientunet_b6',
           'get_efficientunet_b7']

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

def get_blocks_to_be_concat(model, x):
#     shapes = set()
    shapes = []
    blocks = OrderedDict()
    hooks = []
    count = 0
    
    def register_hook(module):
        def hook(module, input, output):
            try:
                nonlocal count
                if module.name == f'blocks_{count}_output_batch_norm':
                    count += 1
                    shape = output.size()[-2:]
                    shape_list = list(shape)
                    if shape_list not in shapes:
                        shapes.append(shape_list)
                        blocks[module.name] = output

                elif module.name == 'head_swish':
                    # when module.name == 'head_swish', it means the program has already got all necessary blocks for
                    # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
                    # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
                    # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
                    # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
                    # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
                    blocks.popitem()
                    blocks[module.name] = output

            except AttributeError:
                pass

        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model)):
            hooks.append(module.register_forward_hook(hook))

    # register hook
    model.apply(register_hook)

    # make a forward pass to trigger the hooks
    model(x)
    
    # remove these hooks
    for h in hooks:
        h.remove()

    return blocks


class EfficientUnet(nn.Module):
    def __init__(self, encoder, out_channels=2, concat_input=True, bn = BatchNorm2d, mode ='mask'):
        super().__init__()

        self.encoder = encoder
        self.out_channels = out_channels
        self.concat_input = concat_input
        self.mode = mode
        
        self.up_conv1 = up_conv(self.n_channels, 512)
        self.double_conv1 = double_conv(self.size[0], 512)
        
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = double_conv(self.size[1], 256)
        
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(self.size[2], 128)
        
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = double_conv(self.size[3], 64)

        if self.concat_input:
            self.up_conv_input = up_conv(64, 32)
            self.double_conv_input = double_conv(self.size[4], 32)

        self.final_conv = nn.Conv2d(self.size[5], out_channels, kernel_size=1)

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 34, 32], 'efficientnet-b1': [592, 296, 152, 80, 34, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 34, 32], 'efficientnet-b3': [608, 304, 160, 88, 34, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 34, 32], 'efficientnet-b5': [640, 320, 168, 88, 34, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 34, 32], 'efficientnet-b7': [672, 336, 176, 96, 34, 32]}
        return size_dict[self.encoder.name]
    
    def _pad(self,x,target):
        if x.shape != target.shape:
            padding_1 = target.shape[2] - x.shape[2]
            padding_2 = target.shape[3] - x.shape[3]
        return F.pad(x,(padding_2//2,padding_2 - padding_2//2,padding_1//2,padding_1-padding_1//2),'constant', 0)
    
    def forward(self, x):
        input_ = x

        blocks = get_blocks_to_be_concat(self.encoder, x)
        _, x = blocks.popitem()
        target = blocks.popitem()[1]
        x = self.up_conv1(x)
        x = self._pad(x,target)
        x = torch.cat([x, target], dim=1)
        x = self.double_conv1(x)

        target = blocks.popitem()[1]
        x = self.up_conv2(x)
        x = self._pad(x,target)
        x = torch.cat([x, target], dim=1)
        x = self.double_conv2(x)
        
        target = blocks.popitem()[1]
        x = self.up_conv3(x)
        x = self._pad(x,target)
        x = torch.cat([x, target], dim=1)
        x = self.double_conv3(x)
        
        target = blocks.popitem()[1]
        x = self.up_conv4(x)
        x = self._pad(x,target)
        x = torch.cat([x, target], dim=1)
        x = self.double_conv4(x)
        
        if self.concat_input:
            x = self.up_conv_input(x)
            x = self._pad(x, input_)
            x = torch.cat([x, input_.type(x.dtype)], dim=1)
            x = self.double_conv_input(x)

        x = self.final_conv(x)
        
        if self.out_channels==3:
            x[:,0,:,:] = torch.clamp(x[:,0,:,:].clone(),-12,7) ## clamp magnitude channel
            x[:,1,:,:] = torch.clamp(x[:,1,:,:].clone(),-np.pi,np.pi) ## clamp phase channel
            x[:,2,:,:] = torch.clamp(x[:,2,:,:].clone(),-10,10) ## clamp mask channel
        if self.out_channels == 2:
            x[:,0,:,:] = torch.clamp(x[:,0,:,:].clone(),-12,7) ## clamp magnitude channel
            x[:,1,:,:] = torch.clamp(x[:,1,:,:].clone(),-np.pi,np.pi) ## clamp phase channel
            
        if self.out_channels==1 and self.mode == 'mask':
            x = torch.clamp(x.clone(),-10,10)
        if self.out_channels==1 and self.mode == 'mag':
            x = torch.clamp(x.clone(),-12,7)
        if self.out_channels==1 and self.mode == 'phase':
            x = torch.clamp(x.clone(),-np.pi,np.pi)
        return x


def get_efficientunet_b0(out_channels=2, concat_input=True, pretrained=True,mode = 'mask', bn = BatchNorm2d):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input, mode = mode,bn = bn)
    return model


def get_efficientunet_b1(out_channels=2, concat_input=True, pretrained=True,mode = 'mask',bn = BatchNorm2d):
    encoder = EfficientNet.encoder('efficientnet-b1', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input, mode = mode,bn = bn)
    return model


def get_efficientunet_b2(out_channels=2, concat_input=True, pretrained=True,mode = 'mask',bn = BatchNorm2d):
    encoder = EfficientNet.encoder('efficientnet-b2', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input,mode = mode, bn = bn)
    return model


def get_efficientunet_b3(out_channels=2, concat_input=True, pretrained=True,mode = 'mask',bn = BatchNorm2d):
    encoder = EfficientNet.encoder('efficientnet-b3', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input, mode = mode,bn = bn)
    return model


def get_efficientunet_b4(out_channels=2, concat_input=True, pretrained=True,mode = 'mask',bn = BatchNorm2d):
    encoder = EfficientNet.encoder('efficientnet-b4', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input, mode = mode,bn = bn)
    return model


def get_efficientunet_b5(out_channels=2, concat_input=True, pretrained=True,mode = 'mask',bn = BatchNorm2d):
    encoder = EfficientNet.encoder('efficientnet-b5', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input, mode = mode,bn = bn)
    return model


def get_efficientunet_b6(out_channels=2, concat_input=True, pretrained=True,mode = 'mask',bn = BatchNorm2d):
    encoder = EfficientNet.encoder('efficientnet-b6', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input, mode = mode,bn = bn)
    return model


def get_efficientunet_b7(out_channels=2, concat_input=True, pretrained=True,mode = 'mask',bn = BatchNorm2d):
    encoder = EfficientNet.encoder('efficientnet-b7', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input, mode = mode,bn = bn)
    return model