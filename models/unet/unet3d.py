import torch.nn as nn
import torch
from functools import partial
# from models.unet3d.buildingblocks import create_conv
import torch.nn.functional as F
from models.unet.se import ChannelSpatialSELayer3D

class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, relu=True, num_groups=8,
                 padding=1):
        super(ConvLayer, self).__init__()
        self.add_module('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=in_channels))
        self.add_module('conv',nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False))
        if relu:
            self.add_module('ReLU', nn.ReLU(inplace=True))


class DiffusionConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_groups=8, temb_dim=128):
        super(DiffusionConvBlock, self).__init__()

        if in_channels != out_channels:
            self.conv1 = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.conv1 = nn.Identity()

        # residual block
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size=kernel_size, relu=True, num_groups=num_groups,
                                )
        self.conv3 = ConvLayer(out_channels, out_channels, kernel_size=kernel_size, relu=False,
                                num_groups=num_groups)
        self.temb_proj = torch.nn.Linear(temb_dim, out_channels)

        self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x, temb):
        residual = self.conv1(x)
        out = self.conv2(residual)
        out = out + self.temb_proj(temb*torch.sigmoid(temb))[:, :, None, None, None]
        out = self.conv3(out)
        out += residual
        out = self.non_linearity(out)
        return out

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_groups=8):
        super(ResNetBlock, self).__init__()

        if in_channels != out_channels:
            self.conv1 = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.conv1 = nn.Identity()

        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size=kernel_size, relu=True, num_groups=num_groups,
                                )
        self.conv3 = ConvLayer(out_channels, out_channels, kernel_size=kernel_size, relu=False,
                                num_groups=num_groups)

        self.non_linearity = nn.ReLU(inplace=True)
        self.se_module = ChannelSpatialSELayer3D(num_channels=out_channels, reduction_ratio=1)

    def forward(self, x, temb=None):
        residual = self.conv1(x)
        out = self.conv2(residual)
        out = self.conv3(out)
        out += residual
        out = self.non_linearity(out)
        out = self.se_module(out)
        return out

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2):
        super().__init__()
        self.conv_transposed = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size,
                                                 stride=scale_factor, padding=1, bias=False)

    def forward(self, encoder_features, x):
        output_size = encoder_features.size()[2:]
        x = self.conv_transposed(x)
        return F.interpolate(x, size=output_size)
    
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=2,
                 num_groups=8, tune_channel=False, scale_factor=None,conv_block='residual', temb_dim=128):
        super(Block, self).__init__()
        self.upsample = False
        if scale_factor is not None:
            self.upsample = True
            self.upsampling = Upsample(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=conv_kernel_size, scale_factor=scale_factor,)
        if apply_pooling:
            self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None
        if tune_channel:
            in_channels = out_channels
        if conv_block == 'residual':
            self.basic_module = ResNetBlock(in_channels, out_channels,
                                            kernel_size=conv_kernel_size,
                                            num_groups=num_groups,
                                            )
        elif conv_block=='diffusion':
            self.basic_module = DiffusionConvBlock(in_channels, out_channels,
                                            kernel_size=conv_kernel_size,
                                            num_groups=num_groups,temb_dim=temb_dim,)
        else:
            raise NotImplementedError('Unknown block type: {}'.format(conv_block))

    def forward(self, x, encoded_features=None, temb=None):
        if self.pooling is not None:
            x = self.pooling(x)
        if encoded_features is not None and self.upsample:
            x = self.upsampling(encoder_features=encoded_features, x=x)
            x = encoded_features + x
        x = self.basic_module(x, temb=temb)
        return x

def create_encoders(in_channels, f_maps, conv_kernel_size,
                    num_groups, pool_kernel_size, conv_block, temb_dim=128):
    blocks = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            block = Block(in_channels, out_feature_num,
                              apply_pooling=False, 
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              conv_block=conv_block,
                              temb_dim=temb_dim
                              )
        else:
            block = Block(f_maps[i - 1], out_feature_num,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              pool_kernel_size=pool_kernel_size,
                              conv_block=conv_block,
                              temb_dim=temb_dim
                              )

        blocks.append(block)

    return nn.ModuleList(blocks)

def create_decoders(f_maps, conv_kernel_size,  
                    num_groups, conv_block, temb_dim=128):
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        in_feature_num = reversed_f_maps[i]

        out_feature_num = reversed_f_maps[i + 1]

        decoder = Block(in_feature_num, out_feature_num,
                          conv_kernel_size=conv_kernel_size,
                          num_groups=num_groups,
                          conv_block=conv_block,
                          scale_factor=2,
                          tune_channel=True,
                          apply_pooling=False,
                          temb_dim=temb_dim
                          )
        decoders.append(decoder)
    return nn.ModuleList(decoders)


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, f_maps=128, conv_block='residual',
                 num_groups=8, num_levels=4, conv_kernel_size=3, pool_kernel_size=2,):
        super(UNet3D, self).__init__()
                    
        f_maps = [f_maps * 2 ** k for k in range(num_levels)]

        self.encoders = create_encoders(in_channels, f_maps, conv_kernel_size,
                                        num_groups, pool_kernel_size, conv_block)

        self.decoders = create_decoders(f_maps, conv_kernel_size,
                                        num_groups, conv_block
                                        )
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)


    def forward(self, x):
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)

        encoders_features = encoders_features[1:]

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(x,encoder_features)

        x = self.final_conv(x)

        return x

class DiffusionUNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, f_maps=64, conv_block='diffusion',
                 num_groups=8, num_levels=4, conv_kernel_size=3, pool_kernel_size=2, temb_dim=128):
        
        super(DiffusionUNet3D, self).__init__()
                    
        f_maps = [f_maps * 2 ** k for k in range(num_levels)]

        self.encoders = create_encoders(in_channels, f_maps, conv_kernel_size,
                                        num_groups, pool_kernel_size, conv_block, temb_dim)

        self.decoders = create_decoders(f_maps, conv_kernel_size,
                                        num_groups, conv_block, temb_dim
                                        )
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)


    def forward(self, x, temb):
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x, temb=temb)
            encoders_features.insert(0, x)

        encoders_features = encoders_features[1:]

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(x,encoded_features = encoder_features, temb=temb)

        x = self.final_conv(x)

        return x