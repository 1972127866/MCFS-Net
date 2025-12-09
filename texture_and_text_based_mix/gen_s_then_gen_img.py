import os
import argparse

from tqdm import tqdm
import clip

import random
import numpy as np
import torch
from torchvision import utils
from utils import stylespace_util
from utils import stylespace_util_all_s
from models.stylegan2.model import Generator
import torchvision

def save_image_pytorch(img, name):
    """Helper function to save torch tensor into an image file."""
    utils.save_image(
        img,
        name,
        nrow=1,
        padding=0,
        normalize=True,
        range=(-1, 1),
    )

def listdir(path, list_name):
    for file in os.listdir(path):
        if file[-3:]=="npy":
            file_path = os.path.join(path, file)
            list_name.append(file_path)


def gen_single_codes_from_path_wo_clip(latent_path,upsample,avg_pool,model):
    latent_code=np.load(latent_path, allow_pickle=True)
    w_latents = torch.from_numpy(latent_code).to(device).unsqueeze(0)

    #使用w生成s
    style_space, noise = stylespace_util.encoder_latent(netG, w_latents)
    s_latents = torch.cat(style_space, dim=1)

    return w_latents,s_latents

def gen_single_codes_from_path_wo_clip_all_s(latent_path,upsample,avg_pool,model):
    latent_code=np.load(latent_path, allow_pickle=True)
    w_latents = torch.from_numpy(latent_code).to(device).unsqueeze(0)

    #使用w生成s（包含toRGB层）
    style_space, noise = stylespace_util_all_s.encoder_latent(netG, w_latents)
    s_latents = torch.cat(style_space, dim=1)

    return w_latents,s_latents

def generate_from_npy(args, netG, device, mean_latent):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    # avg_pool = torch.nn.AvgPool2d(kernel_size=1024 // 32)
    avg_pool = torch.nn.AvgPool2d(kernel_size=args.size // 32)
    upsample = torch.nn.Upsample(scale_factor=7)

    path_list = []
    listdir(args.latent_dir, path_list)

    ind = 0
    with torch.no_grad():
        netG.eval()


        for i in tqdm(range(2,3)):
            # latent_path=args.latent_dir+"/"+str(i)+"_embed_restyle_wo_s_latent_w.npy"
            latent_path=args.latent_dir+"/"+str(i)+".npy"
            w_latents,s_latents=gen_single_codes_from_path_wo_clip(latent_path,upsample,avg_pool,model)
            _,s_latents_all_s=gen_single_codes_from_path_wo_clip_all_s(latent_path,upsample,avg_pool,model)

        with torch.no_grad():
            img_ori = stylespace_util.decoder_validate(netG, s_latents, w_latents)
            img_all_s=stylespace_util_all_s.decoder_validate(netG, s_latents_all_s, w_latents)
            img_list = [img_ori]
            img_list.append(img_all_s)
            img_gen_all = torch.cat(img_list, dim=3)
            torchvision.utils.save_image(img_gen_all, os.path.join("all_s_test_gen","s.jpg"), normalize=True, range=(-1, 1))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

   
    parser.add_argument('--save_dir', type=str, default='./latent_code', help="place to save the output")
    parser.add_argument('--latent_dir', type=str, default='./latent_code', help="place to the embedding results of e4e")
    parser.add_argument('--ckpt', type=str, default='./models/pretrained_models', help="checkpoint file for the generator")
    parser.add_argument('--size', type=int, default=1024, help="output size of the generator")
    parser.add_argument('--fixed_z', type=str, default=None, help="expect a .pth file. If given, will use this file as the input noise for the output")
    parser.add_argument('--w_shift', type=str, default=None, help="expect a .pth file. Apply a w-latent shift to the generator")
    parser.add_argument('--batch_size', type=int, default=10, help="batch size used to generate outputs")
    parser.add_argument('--samples', type=int, default=200000, help="200000 number of samples to generate, will be overridden if --fixed_z is given")
    parser.add_argument('--truncation', type=float, default=1, help="strength of truncation:0.5ori")
    parser.add_argument('--truncation_mean', type=int, default=4096, help="number of samples to calculate the mean latent for truncation")
    parser.add_argument('--seed', type=int, default=None, help="if specified, use a fixed random seed")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_npy', type=bool, default=False)

    args = parser.parse_args()

    device = args.device
    # use a fixed seed if given
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # netG = Generator(args.size, 512, 8).to(device)
    netG = Generator(args.size, 512, 2).to(device)
    ckpt_path = os.path.join(args.ckpt)
    print(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    netG.load_state_dict(checkpoint['g_ema'])

    # get mean latent if truncation is applied
    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = netG.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    if args.use_npy:
        generate_from_npy(args, netG, device, mean_latent)
