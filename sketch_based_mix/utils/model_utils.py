# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import numpy as np
import torch
torch.manual_seed(0)
import os
device_ids = [0]
import dnnlib
import legacy
from PIL import Image
import torch.nn as nn
import re
import base64
import torch.optim as optim
import  math
import cv2
import gc
import torch.nn.functional as F
from tqdm import tqdm
import copy
from models.stylegan2_pytorch.stylegan2_pytorch import Generator as Stylegan2Generator
from models.stylegan1_pytorch.stylegan1 import G_mapping, Truncation, G_synthesis

from models.DatasetGAN.classifer import pixel_classifier
from collections import OrderedDict

#from embed_project_restyle_wo_s import restyle_run_on_batch
#不知道为啥报错我给注释了，但是就能用了-lyx

import warnings
warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def latent_to_image(g_all, upsamplers, latents, return_upsampled_layers=False, use_style_latents=False,
                    process_out=True, return_stylegan_latent=False, dim=512,
                    return_only_im=False, noise=None,return_style=False,use_style=False,input_styles=None):
    '''Given a input latent code, generate corresponding image and concatenated feature maps'''
    # assert (len(latents) == 1)  # for GPU memory constraints   
    if not use_style_latents:
        # generate style_latents from latents
        style_latents = g_all.module.truncation(g_all.module.mapping(latents,None))
        style_latents = style_latents.clone()  # make different layers non-alias

    else:
        #print(" not if not use_style_latents:")
        style_latents = latents

    if return_stylegan_latent:
        return  style_latents
    #noise_mode='random'
    noise_mode='const'
    if not use_style:
        if style_latents.ndim>3:
            style_latents_modi=style_latents.squeeze(0)
        elif style_latents.ndim<3:
            style_latents_modi=style_latents.unsqueeze(0)
        else:
            style_latents_modi=style_latents
        img_list, affine_layers, styles = g_all.module.synthesis(style_latents_modi, noise_mode=noise_mode, return_style=True)#"style_latents"sometimes need to be squeeze(0)
    else:
        img_list, affine_layers, styles = g_all.module.synthesis(None,noise_mode=noise_mode,return_style=True,use_styles=True,input_styles=input_styles)
    #img_list, affine_layers = g_all.module.synthesis(style_latents, noise_mode=noise_mode)
    #print("img_list, affine_layers = g_all.module.g_synthesis(style_latents, noise=noise)")
    #print("len(affine_layers)",len(affine_layers),"affine_layers[0].shape",affine_layers[0].shape)

    if return_only_im:
        if process_out:
            if img_list.shape[-2] > 512:
                img_list = upsamplers[-1](img_list)
            img_list = img_list.cpu().detach().numpy()
            img_list = process_image(img_list)
            img_list = np.transpose(img_list, (0, 2, 3, 1)).astype(np.uint8)
        if return_style:
            return img_list, style_latents,styles
        return img_list, style_latents

    number_feautre = 0

    for item in affine_layers:
        number_feautre += item.shape[1]
        #print("item (in affine_layers).shape[1]",item.shape[1])
    #print("number_feautre",number_feautre)

    if return_upsampled_layers:
        affine_layers_upsamples = torch.FloatTensor(1, number_feautre, dim, dim).cuda()
        #print("affine_layers_upsamples.shape",affine_layers_upsamples.shape)
        start_channel_index = 0
        for i in range(len(affine_layers)):
            len_channel = affine_layers[i].shape[1]
            #print("affine_layers[i].shape",affine_layers[i].shape,"upsamplers[i](affine_layers[i]).shape",upsamplers[i](affine_layers[i]).shape)
            affine_layers_upsamples[:, start_channel_index:start_channel_index + len_channel] = upsamplers[i](
                affine_layers[i])
            start_channel_index += len_channel
    else:
        affine_layers_upsamples = affine_layers
    
    #print("img_list.shape[-2]",img_list.shape[-2])
    if img_list.shape[-2] != 256:
        img_list = upsamplers[-1](img_list)

    if process_out:
        img_list = img_list.cpu().detach().numpy()
        img_list = process_image(img_list)
        img_list = np.transpose(img_list, (0, 2, 3, 1)).astype(np.uint8)
        #print("img_list.shape",img_list.shape)

    if return_style:
        return img_list, affine_layers_upsamples,styles
    return img_list, affine_layers_upsamples


def process_image(images):
    drange = [-1, 1]
    scale = 255 / (drange[1] - drange[0])
    images = images * scale + (0.5 - drange[0] * scale)

    images = images.astype(int)
    images[images > 255] = 255
    images[images < 0] = 0

    return images.astype(int)


def reverse_process_image(images):
    drange = [-1, 1]
    scale = 255 / (drange[1] - drange[0])
    images = images - (0.5 - drange[0] * scale)
    images = images / scale

    return images

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    # t > .75
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    # t < 0.05
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


class Interpolate(nn.Module):
    def __init__(self, size, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if self.align_corners:
            x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        else:
            x = self.interp(x, size=self.size, mode=self.mode)
        return x




def run_embedding_optimization(args, g_all, upsamplers, inter, percept, img_tensor, latent_in,   #inter = Interpolate(args['im_size'][1], 'bilinear')
                               steps=1000, stylegan_encoder=None, regular_by_org_latent=False, early_stop=True,
                               encoder_loss_weight=1, use_noise=False, noise_loss_weight=100,use_restyle=False,
                               restyle_opts=None,avg_img=None):

    gc.collect()
    torch.cuda.empty_cache()
    latent_in = latent_in.detach()
    img_tensor = (img_tensor + 1.0) / 2.0

    org_latnet_in = copy.deepcopy(latent_in.detach())
    with torch.no_grad():
        im_out_wo_opti, _ = latent_to_image(g_all, upsamplers, org_latnet_in, process_out=False,
                                            dim=args['im_size'][1],
                                            use_style_latents=True, return_only_im=True)

        im_out_wo_opti = inter(im_out_wo_opti)
        im_out_wo_opti = (im_out_wo_opti + 1.0) / 2.0
        p_loss = percept(im_out_wo_opti, img_tensor).mean()
        mse_loss = F.mse_loss(im_out_wo_opti, img_tensor)
        #print("before_w_optied,p_loss:",p_loss,",mse_loss:",mse_loss)

    best_loss = args['loss_dict']['p_loss'] * p_loss + \
                args['loss_dict']['mse_loss'] * mse_loss

    if args['truncation']:
        latent_in = g_all.module.truncation(latent_in)

    latent_in.requires_grad = True

    if use_noise:
        noises = g_all.module.make_noise()

        for noise in noises:
            noise.requires_grad = True
    else:
        noises = None
    if not use_noise:
        optimizer = optim.Adam([latent_in], lr=3e-5)
    else:

        optimizer = optim.Adam([latent_in] + noises, lr=3e-5)
        optimized_noise = noises

    count = 0
    optimized_latent = latent_in

    loss_cache = [best_loss.item()]
    #print("steps",steps)
    for _ in tqdm(range(1, steps),position=0):
        #print(_)
        t = _ / steps
        lr = get_lr(t, 0.1)
        #print("t:",t,"lr:",lr)
        optimizer.param_groups[0]['lr'] = lr

        img_out, _ = latent_to_image(g_all, upsamplers, latent_in, process_out=False,
                                     dim=args['im_size'][1],
                                     use_style_latents=True, return_only_im=True, noise=noises)

        img_out = inter(img_out)

        img_out = (img_out + 1.0) / 2.0



        p_loss = percept(img_out, img_tensor).mean()

        mse_loss = F.mse_loss(img_out, img_tensor)
        if regular_by_org_latent or use_restyle:#regular_by_org_latent决定是跟优化后的w算loss还是encoder的输出算loss
            encoder_loss = F.mse_loss(latent_in, org_latnet_in)
        else:
            if use_restyle:#这里使用encoder的效益较低，可以不使用
                with torch.no_grad():
                    input_cuda = img_out.float()
                    _, restyle_latents = restyle_run_on_batch(input_cuda, stylegan_encoder, restyle_opts, avg_img)
                restyle_latent=torch.from_numpy(restyle_latents[0][4]).unsqueeze(0).cuda()
                encoder_loss = F.mse_loss(latent_in,restyle_latent.detach())
            else:
                encoder_loss = F.mse_loss(latent_in, stylegan_encoder(img_out).detach())

        reconstruction_loss = args['loss_dict']['p_loss'] * p_loss + \
                              args['loss_dict']['mse_loss'] * mse_loss

        loss = reconstruction_loss + encoder_loss_weight * encoder_loss
       
        if use_noise:
            n_loss = noise_regularize(noises)

            loss += noise_loss_weight * n_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_cache.append(reconstruction_loss.item())
        if reconstruction_loss.item() < best_loss:
            best_loss = reconstruction_loss.item()
            count = 0
            optimized_latent = latent_in.detach()
            if use_noise:
                optimized_noise = [noise.detach().cpu().numpy() for noise in noises]
        else:
            count += 1
        if early_stop and count > 100:
            break

    #print("loss:",loss,"  reconstruction_loss:" ,reconstruction_loss,"  encoder_loss:" ,encoder_loss)
    gc.collect()
    torch.cuda.empty_cache()

    if use_noise:
        return optimized_latent, optimized_noise, loss_cache
    else:
        return optimized_latent, None, loss_cache

def run_embedding_optimization_s_space(args, g_all, upsamplers, inter, percept, img_tensor, latent_in,   #inter = Interpolate(args['im_size'][1], 'bilinear')
                               steps=1000, stylegan_encoder=None, regular_by_org_latent=False, early_stop=True,
                               encoder_loss_weight=1, use_noise=False, noise_loss_weight=100):

    gc.collect()
    torch.cuda.empty_cache()
    latent_in = latent_in.detach()
    img_tensor = (img_tensor + 1.0) / 2.0

    org_latnet_in = copy.deepcopy(latent_in.detach())
    with torch.no_grad():
        im_out_wo_opti, _,styles = latent_to_image(g_all, upsamplers, org_latnet_in, process_out=False,
                                            dim=args['im_size'][1],
                                            use_style_latents=True, return_only_im=True,return_style=True)#use_style_latents只决定是否由随机数生成latent code，和用s还是w没有关系

        im_out_wo_opti = inter(im_out_wo_opti)
        im_out_wo_opti = (im_out_wo_opti + 1.0) / 2.0
        p_loss = percept(im_out_wo_opti, img_tensor).mean()
        
        mse_loss = F.mse_loss(im_out_wo_opti, img_tensor) 
#         mse_loss = F.mse_loss(im_out_wo_opti[:,:,0:128,0:128], img_tensor[:,:,0:128,0:128]) +F.mse_loss(im_out_wo_opti[:,:,0:128,128:256], img_tensor[:,:,0:128,128:256])+ F.mse_loss(im_out_wo_opti[:,:,128:256,0:128], img_tensor[:,:,128:256,0:128])+ F.mse_loss(im_out_wo_opti[:,:,128:256,128:256], img_tensor[:,:,128:256,128:256])
        
        reconstruction_loss = args['loss_dict']['p_loss'] * p_loss + \
                              args['loss_dict']['mse_loss'] * mse_loss
        
        # im_out_wo_opti_s, _s = latent_to_image(g_all, upsamplers, org_latnet_in, process_out=False,
        #                                             dim=args['im_size'][1],
        #                                             use_style_latents=True, return_only_im=True,use_style=True,input_styles=styles)

#         im_out_wo_opti_s = inter(im_out_wo_opti_s)
#         im_out_wo_opti_s = (im_out_wo_opti_s + 1.0) / 2.0
#         p_loss_s = percept(im_out_wo_opti_s, img_tensor).mean()
#         mse_loss_s = F.mse_loss(im_out_wo_opti_s, img_tensor)
        
#         print("before_s_optied,p_loss:",p_loss,",mse_loss:",mse_loss,"reconstruction_loss:",reconstruction_loss)
        #print("before_s_optied,use styles to make image,p_loss_s:",p_loss_s,",mse_loss_s:",mse_loss_s)
    
    org_styles = []
    for i in range(len(styles)):
        org_styles.append(copy.deepcopy(styles[i].detach()))
    #mask=torch.ones_like(org_styles)


    best_loss = args['loss_dict']['p_loss'] * p_loss + \
                args['loss_dict']['mse_loss'] * mse_loss
    
#     print("best_loss:",best_loss)

    if args['truncation']:
        latent_in = g_all.module.truncation(latent_in)

    #latent_in.requires_grad = True
    for i in range(len(styles)):
        if i%3==1:
             continue
        styles[i].requires_grad = True

    if use_noise:
        noises = g_all.module.make_noise()

        for noise in noises:
            noise.requires_grad = True
    else:
        noises = None
    if not use_noise:
        #optimizer = optim.Adam([latent_in], lr=3e-5)
         optimizer = optim.Adam([styles[0],styles[2],styles[3],styles[5],styles[6],styles[8],styles[9],styles[11],styles[12],styles[14],styles[15],styles[17],styles[18]], lr=3e-5)
#         optimizer = optim.Adam([styles[0],styles[1],styles[2],styles[3],styles[4],styles[5],styles[6],styles[7],styles[8],styles[9],styles[10],styles[11],styles[12],styles[13],styles[14],styles[15],styles[16],styles[17],styles[18],styles[19]], lr=3e-5)
#         optimizer = optim.Adam([styles[0],styles[2],styles[3],styles[5],styles[6],styles[8],styles[9],styles[11],styles[12],styles[14],styles[15],styles[17],styles[18]], lr=3e-5)
#         optimizer = optim.Adam([styles[8],styles[9],styles[11],styles[12],styles[14],styles[15],styles[17],styles[18]], lr=3e-1)
    else:

        optimizer = optim.Adam([latent_in] + noises, lr=3e-5)
        optimized_noise = noises

    count = 0
    #optimized_latent = latent_in
    optimized_styles = styles

    loss_cache = [best_loss.item()]
    for _ in tqdm(range(1, steps),position=0):
        t = _ / steps
        #lr = get_lr(t, 0.1)
        # if t<0.2:
        #     lr=0.1
        # elif t<0.5:
        #     lr=0.05
        # else:
        #     lr=0.01
        
        lr_ramp = min(1.0, (1.0 - t) /0.25)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / 0.05)
        lr = 0.01 * lr_ramp
        #lr=0.001
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        #optimizer.param_groups[0]['lr'] = lr

        img_out, _ = latent_to_image(g_all, upsamplers, latent_in, process_out=False,
                                     dim=args['im_size'][1],
                                     use_style_latents=True, return_only_im=True, noise=noises,use_style=True,input_styles=styles)

        img_out = inter(img_out)

        img_out = (img_out + 1.0) / 2.0



        p_loss = percept(img_out, img_tensor).mean()

        mse_loss = F.mse_loss(img_out, img_tensor)
#         mse_loss = F.mse_loss(im_out_wo_opti[:,:,0:128,0:128], img_tensor[:,:,0:128,0:128]) +F.mse_loss(im_out_wo_opti[:,:,0:128,128:256], img_tensor[:,:,0:128,128:256])+ F.mse_loss(im_out_wo_opti[:,:,128:256,0:128], img_tensor[:,:,128:256,0:128])+ F.mse_loss(im_out_wo_opti[:,:,128:256,128:256], img_tensor[:,:,128:256,128:256])
        
        #encoder loss
        # if regular_by_org_latent:#regular_by_org_latent决定是跟优化后的w算loss还是encoder的输出算loss
        #     encoder_loss = F.mse_loss(styles, org_styles)
        # else:#默认走这里，但因为encoder输出为w，这里优化的是s，不方便比，所以还是直接跟由优化后的w得来的org_s比
        #     encoder_loss = F.mse_loss(styles[0], org_styles[0]) 
        #     for j in range(19):
        #         if (j+1)%3==1:#torgb层的不用比
        #             continue
        #         encoder_loss += F.mse_loss(styles[j+1], org_styles[j+1])


        reconstruction_loss = args['loss_dict']['p_loss'] * p_loss + \
                              args['loss_dict']['mse_loss'] * mse_loss

        #loss = reconstruction_loss + encoder_loss_weight * encoder_loss
        #取消了encodr loss
        loss = reconstruction_loss
        #print("t:",t,"lr:",lr,"loss:",loss)

        if use_noise:
            n_loss = noise_regularize(noises)

            loss += noise_loss_weight * n_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_cache.append(reconstruction_loss.item())
        if reconstruction_loss.item() < best_loss:
            best_loss = reconstruction_loss.item()
            count = 0
            #optimized_styles = styles.detach()
            optimized_styles = []
            for i in range(len(styles)):
                optimized_styles.append(styles[i].detach())
            if use_noise:
                optimized_noise = [noise.detach().cpu().numpy() for noise in noises]
        else:
            count += 1
        if early_stop and count > 100:
            break
#     print("-------------------------------------------------------")
#     for i in range(len(org_styles)):
#         x=org_styles[i]-optimized_styles[i]
#         print(i,x)
    #print("loss:",loss,"  reconstruction_loss:" ,reconstruction_loss,"  encoder_loss:" ,encoder_loss)
    gc.collect()
    torch.cuda.empty_cache()

    if use_noise:
        return optimized_styles, optimized_noise, loss_cache
    else:
        return optimized_styles, None, loss_cache


def prepare_model(args, classfier_checkpoint_path="", classifier_iter=10000, num_class=34, num_classifier=10):

    if args['category'] == 'face' or args['category'] == 'flickr_car':
        res = 1024
        out_res = 512
    elif args['category'] == 'face_256':
        res = 256
        out_res = 256

    else:
        res = 256
        out_res = 256

    if args['stylegan_ver'] == "1":

        if args['category'] == "car":
            max_layer = 8
        elif args['category'] == "face":
            max_layer = 8
        elif args['category'] == "bedroom":
            max_layer = 7
        elif args['category'] == "cat":
            max_layer = 7
        else:
            assert "Not implementated!"


        avg_latent = np.load(args['average_latent'])
        avg_latent = torch.from_numpy(avg_latent).type(torch.FloatTensor).cuda()

        g_all = nn.Sequential(OrderedDict([
            ('g_mapping', G_mapping()),
            ('truncation', Truncation(avg_latent,max_layer=max_layer, device=device, threshold=0.7)),
            ('g_synthesis', G_synthesis( resolution=res))
        ]))

        g_all.load_state_dict(torch.load(args['stylegan_checkpoint'], map_location=device))
        g_all.eval()
        g_all = nn.DataParallel(g_all, device_ids=device_ids).cuda()

    elif args['stylegan_ver'] == "2":
        g_all = Stylegan2Generator(res, 512, 8, channel_multiplier=2, randomize_noise=False)
        checkpoint = torch.load(args['stylegan_checkpoint'])

        print("Load stylegan from, " , args['stylegan_checkpoint'], " at res, ", str(res))
        g_all.load_state_dict(checkpoint["g_ema"], strict=True)
        avg_latent = g_all.make_mean_latent(4086)

    elif args['stylegan_ver'] == "3":
        # g_all = Stylegan2Generator(res, 512, 8, channel_multiplier=2, randomize_noise=False)
        # checkpoint = torch.load(args['stylegan_checkpoint'])
        #
        # print("Load stylegan from, " , args['stylegan_checkpoint'], " at res, ", str(res))
        # g_all.load_state_dict(checkpoint["g_ema"], strict=True)
        # avg_latent = g_all.make_mean_latent(4086)
        print('Loading networks from "%s"...' % args['stylegan_checkpoint'])
        device = torch.device('cuda')
        with dnnlib.util.open_url(args['stylegan_checkpoint']) as f:
            g_all = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
        avg_latent = g_all.make_mean_latent(4086)


    g_all.eval()
    g_all = nn.DataParallel(g_all, device_ids=device_ids).cuda()
    mode = "nearest"
    nn_upsamplers = [nn.Upsample(scale_factor=out_res / 4, mode=mode),
                  nn.Upsample(scale_factor=out_res / 4, mode=mode),
                  nn.Upsample(scale_factor=out_res / 8, mode=mode),
                  nn.Upsample(scale_factor=out_res / 8, mode=mode),
                  nn.Upsample(scale_factor=out_res / 16, mode=mode),
                  nn.Upsample(scale_factor=out_res / 16, mode=mode),
                  nn.Upsample(scale_factor=out_res / 32, mode=mode),
                  nn.Upsample(scale_factor=out_res / 32, mode=mode),
                  nn.Upsample(scale_factor=out_res / 64, mode=mode),
                  nn.Upsample(scale_factor=out_res / 64, mode=mode),
                  nn.Upsample(scale_factor=out_res / 128, mode=mode),
                  nn.Upsample(scale_factor=out_res / 128, mode=mode),
                  nn.Upsample(scale_factor=out_res / 256, mode=mode),
                  nn.Upsample(scale_factor=out_res / 256, mode=mode),
                  nn.Upsample(scale_factor=out_res / 512, mode=mode),
                  nn.Upsample(scale_factor=out_res / 512, mode=mode)]

    if res > 512:
        nn_upsamplers.append(Interpolate(512, mode, align_corners=None))
        nn_upsamplers.append(Interpolate(512, mode, align_corners=None))


    mode = 'bilinear'
    bi_upsamplers = [nn.Upsample(scale_factor=out_res / 4, align_corners=False,mode=mode),
                  nn.Upsample(scale_factor=out_res / 4, align_corners=False,mode=mode),
                  nn.Upsample(scale_factor=out_res / 8, align_corners=False,mode=mode),
                  nn.Upsample(scale_factor=out_res / 8, align_corners=False,mode=mode),
                  nn.Upsample(scale_factor=out_res / 16, align_corners=False,mode=mode),
                  nn.Upsample(scale_factor=out_res / 16, align_corners=False,mode=mode),
                  nn.Upsample(scale_factor=out_res / 32, align_corners=False,mode=mode),
                  nn.Upsample(scale_factor=out_res / 32, align_corners=False,mode=mode),
                  nn.Upsample(scale_factor=out_res / 64, align_corners=False,mode=mode),
                  nn.Upsample(scale_factor=out_res / 64, align_corners=False,mode=mode),
                  nn.Upsample(scale_factor=out_res / 128, align_corners=False,mode=mode),
                  nn.Upsample(scale_factor=out_res / 128, align_corners=False,mode=mode),
                  nn.Upsample(scale_factor=out_res / 256, align_corners=False,mode=mode),
                  nn.Upsample(scale_factor=out_res / 256, align_corners=False,mode=mode),
                  nn.Upsample(scale_factor=out_res / 512, align_corners=False,mode=mode),
                  nn.Upsample(scale_factor=out_res / 512, align_corners=False,mode=mode)]

    if res > 512:
        bi_upsamplers.append(Interpolate(512, mode))
        bi_upsamplers.append(Interpolate(512, mode))


    if classfier_checkpoint_path != "":
        print("Load Classifier path, ", classfier_checkpoint_path)
        classifier_list = []
        for MODEL_NUMBER in range(num_classifier):
            classifier = pixel_classifier(num_class, dim=args['dim'])
            classifier = nn.DataParallel(classifier, device_ids=device_ids).cuda()
            if classifier_iter > 0:
                checkpoint = torch.load(os.path.join(classfier_checkpoint_path,
                                                     'model_iter' + str(classifier_iter) + '_number_' + str(
                                                         MODEL_NUMBER) + '.pth'))
            else:
                checkpoint = torch.load(os.path.join(classfier_checkpoint_path,
                                                     'model_' + str(MODEL_NUMBER) + '.pth'))

            classifier.load_state_dict(checkpoint['model_state_dict'], strict=True)
            classifier.eval()
            classifier_list.append(classifier)

        for c in classifier_list:
            for i in c.parameters():
                i.requires_grad = False
    else:
        classifier_list = []

    for i in g_all.parameters():
        i.requires_grad = False



    return g_all, nn_upsamplers, bi_upsamplers, classifier_list, avg_latent
