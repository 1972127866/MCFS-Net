import argparse
from matplotlib import pyplot as plt
import numpy as np
import PIL.Image
import copy
import torch
import dnnlib
import legacy
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
from tqdm import tqdm

import os
import numpy as np
from PIL import Image

device = torch.device('cuda')
with dnnlib.util.open_url("cloth-v2-620t-s-test.pkl") as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
        avg_latent = G.make_mean_latent(4086)

layer=6
channel=475

### from latent_s
# name = '1115.npy'
# data = np.load(os.path.join('../../hdd/wanqing/editGAN/latent_S', name), allow_pickle=True)
# ss = []
# ss_pos=[]
# ss_neg=[]
# for i in range(20):
#     ss.append(data[i].cuda())
#     ss_pos.append(ss[i].clone())
#     ss_neg.append(ss[i].clone())
###

### from sample 
z = torch.from_numpy(np.random.RandomState(2027).randn(1, G.z_dim)).to(device)
w = G.mapping(z.to(device), None)
w1=0.2*avg_latent+0.8*w
_,_,ss = G.synthesis(w1, noise_mode='const',return_style=True)
ss_pos=[]
ss_neg=[]
for i in range(20):
    ss_pos.append(ss[i].clone())
    ss_neg.append(ss[i].clone())
###


print('ss',ss[layer][0][channel])
ss_pos[layer][0][channel]+=40.
print("ss_pos",ss_pos[layer][0][channel])
ss_neg[layer][0][channel]-=80.
#ss_neg[6][0][108]-=70.
print("ss_neg",ss_neg[layer][0][channel])
print("ss",ss[layer][0][channel])

# s_cpu=[]
# for i in range(20):
#     s_cpu.append(ss_neg[i].cpu())
# np.save('../../hdd/wanqing/editGAN/p-sam2027_sleeve-7080.npy',s_cpu)

output_orig,_,_=G.synthesis(None,return_style=True,use_styles=True,input_styles=ss,noise_mode='const')
img1 = (output_orig.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
PIL.Image.fromarray(img1[0].cpu().numpy(), 'RGB').save('../../hdd/wanqing/editGAN/p-sam2027'+"orig.jpg")

output_3,_,_=G.synthesis(None,return_style=True,use_styles=True,input_styles=ss_pos,noise_mode='const')
img2 = (output_3.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
PIL.Image.fromarray(img2[0].cpu().numpy(), 'RGB').save('../../hdd/wanqing/editGAN/p-sam2027l'+str(layer)+"c"+str(channel)+"_pos.jpg")

output_7,_,_=G.synthesis(None,return_style=True,use_styles=True,input_styles=ss_neg,noise_mode='const')
img3 = (output_7.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
PIL.Image.fromarray(img3[0].cpu().numpy(), 'RGB').save('../../hdd/wanqing/editGAN/p-sam2027l'+str(layer)+"c"+str(channel)+"_neg.jpg")
print("done")