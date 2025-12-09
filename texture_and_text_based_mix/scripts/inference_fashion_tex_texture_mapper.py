import os
import sys
sys.path.append(".")
sys.path.append("..")

import copy
import clip
import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader

import torch.nn.functional as F

from datasets.test_dataset_fashion_tex_mapper import TestLatentsDataset

from models.stylegan2.model import Generator
from fashion_tex_latent_mapper import Mapper

from options.test_options_white import TestOptions

from utils import stylespace_util

def main(opts):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Initialize test dataset
    # test_dataset = TestLatentsDataset(opts.test_latent_dir,opts.classname,opts.sample)
    test_dataset = TestLatentsDataset(opts)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=opts.batch_size,
                                 shuffle=False,
                                 num_workers=int(opts.workers),
                                 drop_last=True)

    #Initialize generator
    print('Loading stylegan weights from pretrained!')
    g_ema = Generator(size=opts.stylegan_size, style_dim=512, n_mlp=8)
    g_ema_ckpt = torch.load(opts.stylegan_weights)
    g_ema.load_state_dict(g_ema_ckpt['g_ema'], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    net = Mapper(opts)
    net_ckpt = torch.load(opts.checkpoint_path)
    net.load_state_dict(net_ckpt)
    net = net.to(device)

    os.makedirs(os.path.join(opts.save_dir,"fashion_tex_texture_mapper",opts.images_classname,opts.texture_classname), exist_ok=True)

    for bid, batch in enumerate(test_dataloader):
        if bid == opts.num_all:
            break
        latent_w1, latent_c= batch
        latent_w1 = latent_w1.to(device)
        latent_c = latent_c.to(device)
        
        with torch.no_grad():

            fake_w = latent_w1 + 0.1*net(latent_w1, latent_c)
            style_space_fake, noise = stylespace_util.encoder_latent(g_ema, fake_w)
            img_fake = stylespace_util.decoder(g_ema, style_space_fake,fake_w, noise)
            torchvision.utils.save_image(img_fake, os.path.join(opts.save_dir,"fashion_tex_texture_mapper",opts.images_classname,opts.texture_classname, "%04d.jpg" %(bid+1)), normalize=True, range=(-1, 1))

    print(f'completed👍! Please check results in {opts.save_dir}')

if __name__ == "__main__":
    opts = TestOptions().parse()
    main(opts)