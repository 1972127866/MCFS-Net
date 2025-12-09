import math

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from models.stylegan2.model import EqualLinear, PixelNorm

#额外使用专门的mapper来学toRGB层潜码的映射
class Mapper(Module):

    def __init__(self, in_channel=512, out_channel=512, norm=True, num_layers=4):
        super(Mapper, self).__init__()

        layers = [PixelNorm()] if norm else []
        
        layers.append(EqualLinear(in_channel, out_channel, lr_mul=0.01, activation='fused_lrelu'))
        for _ in range(num_layers-1):
            layers.append(EqualLinear(out_channel, out_channel, lr_mul=0.01, activation='fused_lrelu'))
        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mapping(x)
        return x

class DeltaMapper(Module):

    def __init__(self):
        super(DeltaMapper, self).__init__()

        #Style Module(sm)
        # self.sm_coarse = Mapper(512,  512)
        self.sm_medium = Mapper(512,  512)
        self.sm_fine   = Mapper(1344, 1344)
        self.sm_rgb = Mapper(1984,1984)

        #Condition Module(cm)
        # self.cm_coarse = Mapper(1024, 512)
        self.cm_medium = Mapper(1024, 512)
        self.cm_fine   = Mapper(1024, 1344)
        self.cm_rgb    = Mapper(1024, 1984)

        #Fusion Module(fm)
        # self.fm_coarse = Mapper(512*2,  512,  norm=False)
        self.fm_medium = Mapper(512*2,  512,  norm=False)
        self.fm_fine   = Mapper(1344*2, 1344, norm=False)
        self.fm_rgb   = Mapper(1984*2, 1984, norm=False)
        
    def forward(self, sspace_feat, clip_feat):

        s_coarse = sspace_feat[:, :4*512].view(-1,4*512)#对应0~3号通道
        # s_medium = sspace_feat[:, 4*512:11*512].view(-1,7,512)
        # s_fine   = sspace_feat[:, 11*512:] #channels:2464  #channels:1344 #channels:1792

        s_medium = torch.cat([sspace_feat[:, 5*512:7*512],
                              sspace_feat[:, 8*512:10*512],],dim=1).view(-1,4,512)

        s_fine   = torch.cat([sspace_feat[:, 11*512:12*512+256],
                              sspace_feat[:, 12*512+256*2 : 12*512 + 256*3 + 128],
                              sspace_feat[:, 12*512 + 256*3 + 128*2 : 12*512 + 256*3+ 128*3 + 64],],dim=1)#[batchSize][512+256,256+128,128+64]
        
        s_rgb   = torch.cat([sspace_feat[:, 4*512:5*512],
                             sspace_feat[:, 7*512:8*512],
                             sspace_feat[:, 10*512:11*512],
                             sspace_feat[:, 12*512+256:12*512+256*2],
                             sspace_feat[:, 12*512 + 256*3 + 128:12*512 + 256*3 + 128*2],
                             sspace_feat[:, 12*512 + 256*3 + 128*3 + 64:],],dim=1)#[batchSize][512,512,512,256,128,64]
        

        # s_coarse = self.sm_coarse(s_coarse)
        s_medium = self.sm_medium(s_medium)
        s_fine   = self.sm_fine(s_fine)
        s_rgb   = self.sm_rgb(s_rgb)

        # c_coarse = self.cm_coarse(clip_feat)
        c_medium = self.cm_medium(clip_feat)
        c_fine   = self.cm_fine(clip_feat)
        c_rgb   = self.cm_rgb(clip_feat)

        # x_coarse = torch.cat([s_coarse, torch.stack([c_coarse]*4, dim=1)], dim=2) #[b,3,1024]
        x_medium = torch.cat([s_medium, torch.stack([c_medium]*4, dim=1)], dim=2) #[b,4,1024]
        x_fine   = torch.cat([s_fine, c_fine], dim=1) #[b,2464*2]
        x_rgb   = torch.cat([s_rgb, c_rgb], dim=1)

        # x_coarse = self.fm_coarse(x_coarse)
        # x_coarse = x_coarse.view(-1,4*512)

        x_medium = self.fm_medium(x_medium)
        x_medium = x_medium.view(-1,4*512)

        x_fine   = self.fm_fine(x_fine)
        x_rgb   = self.fm_rgb(x_rgb)

        #重建s
        x_coarse=torch.zeros_like(s_coarse)
        x_mapped = torch.cat([x_rgb[:,:512],
                              x_medium[:,:2*512],x_rgb[:,512:2*512],
                              x_medium[:,2*512:4*512],x_rgb[:,2*512:3*512],
                              x_fine[:,:512+256],x_rgb[:,3*512:3*512+256],
                              x_fine[:,512+256:512+256+256+128],x_rgb[:,3*512+256:3*512+256+128],
                              x_fine[:,512+256+256+128:],x_rgb[:,3*512+256+128:],],dim=1)

        out = torch.cat([x_coarse, x_mapped], dim=1)
        return out