import os
import argparse

from tqdm import tqdm
import clip

import random
import numpy as np
import torch
from torchvision import utils
from utils import stylespace_util_all_s
from models.stylegan2.model import Generator
from datasets.latents_dataset import LatentsDataset
from torch.utils.data import DataLoader

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

#由latents.pt文件生成训练数据，一般不用这个函数
def generate_from_pt(args, netG, device, mean_latent):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    # avg_pool = torch.nn.AvgPool2d(kernel_size=1024 // 32)
    avg_pool = torch.nn.AvgPool2d(kernel_size=args.size // 32)
    upsample = torch.nn.Upsample(scale_factor=7)

    #读取embed得到的文件，获得w
    latents_file_path = os.path.join(args.latent_dir, 'latents.pt')
    latent_codes = torch.load(latents_file_path)
    dataset_latent = LatentsDataset(latents=latent_codes.cpu())
    dataset_latent_dataloader = DataLoader(dataset_latent,
									batch_size=args.batch_size,
									shuffle=True,
									drop_last=True)
    ind = 0
    with torch.no_grad():
        netG.eval()

        # Generate image by sampling input noises
        w_latents_list = []
        s_latents_list = []
        c_latents_list = []
        for batch_idx, batch in tqdm(enumerate(dataset_latent_dataloader)):
            w_latents = batch
            w_latents=w_latents.to(device)

            #使用w生成s
            style_space, noise = stylespace_util_all_s.encoder_latent(netG, w_latents)
            s_latents = torch.cat(style_space, dim=1)

            tmp_imgs = stylespace_util_all_s.decoder(netG, style_space, w_latents, noise)
            # print("tmp_imgs.shape",tmp_imgs.shape)#[10, 3, 256, 256]

            img_gen_for_clip = upsample(tmp_imgs)
            img_gen_for_clip = avg_pool(img_gen_for_clip)
            # print("img_gen_for_clip.shape",img_gen_for_clip.shape)#[10, 3, 256, 256]

            c_latents = model.encode_image(img_gen_for_clip)

            w_latents_list.append(w_latents)
            s_latents_list.append(s_latents)
            c_latents_list.append(c_latents)
        w_all_latents = torch.cat(w_latents_list, dim=0)
        s_all_latents = torch.cat(s_latents_list, dim=0)
        c_all_latents = torch.cat(c_latents_list, dim=0)

        print(w_all_latents.size())
        print(s_all_latents.size())
        print(c_all_latents.size())

        w_all_latents = w_all_latents.cpu().numpy()
        s_all_latents = s_all_latents.cpu().numpy()
        c_all_latents = c_all_latents.cpu().numpy()

        os.makedirs(os.path.join(args.save_dir, args.classname), exist_ok=True)
        np.save(f"{args.save_dir}/{args.classname}/wspace_{args.classname}_feat.npy", w_all_latents)
        np.save(f"{args.save_dir}/{args.classname}/sspace_{args.classname}_feat.npy", s_all_latents)
        np.save(f"{args.save_dir}/{args.classname}/cspace_{args.classname}_feat.npy", c_all_latents)

def listdir(path, list_name):
    for file in os.listdir(path):
        if file[-3:]=="npy":
            file_path = os.path.join(path, file)
            list_name.append(file_path)


def gen_single_codes_from_path_wo_clip(latent_path,upsample,avg_pool,model):
    latent_code=np.load(latent_path, allow_pickle=True)
    w_latents = torch.from_numpy(latent_code).to(device).unsqueeze(0)

    #使用w生成s（包含toRGB层）
    style_space, noise = stylespace_util_all_s.encoder_latent(netG, w_latents)
    s_latents = torch.cat(style_space, dim=1)

    return w_latents,s_latents

#由w+空间的潜向量生成训练数据，这个文件里一般是用这个函数
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

        # Generate image by sampling input noises
        w_latents_list = []
        s_latents_list = []
        c_latents_list = []

        for i in tqdm(range(200000)):
            # latent_path=args.latent_dir+"/"+str(i)+"_embed_restyle_wo_s_latent_w.npy"
            latent_path=args.latent_dir+"/"+str(i)+".npy"
            w_latents,s_latents=gen_single_codes_from_path_wo_clip(latent_path,upsample,avg_pool,model)
            w_latents_list.append(w_latents)
            s_latents_list.append(s_latents)

            #因为一下子生成200000个特征会爆显存，所以逐10000个保存一个文件
            if (i+1)%10000==0:
                w_all_latents = torch.cat(w_latents_list, dim=0)
                s_all_latents = torch.cat(s_latents_list, dim=0)

                w_all_latents = w_all_latents.cpu().numpy()
                s_all_latents = s_all_latents.cpu().numpy()

                os.makedirs(os.path.join(args.save_dir, args.classname), exist_ok=True)
                np.save(f"{args.save_dir}/{args.classname}/"+str((i+1)/10000)+"_w.npy", w_all_latents)
                np.save(f"{args.save_dir}/{args.classname}/"+str((i+1)/10000)+"_s.npy", s_all_latents)
                w_latents_list = []
                s_latents_list = []

        w_all_latents = torch.cat(w_latents_list, dim=0)
        s_all_latents = torch.cat(s_latents_list, dim=0)

        print(w_all_latents.size())
        print(s_all_latents.size())

        w_all_latents = w_all_latents.cpu().numpy()
        s_all_latents = s_all_latents.cpu().numpy()

        os.makedirs(os.path.join(args.save_dir, args.classname), exist_ok=True)
        np.save(f"{args.save_dir}/{args.classname}/wspace_{args.classname}_feat.npy", w_all_latents)
        np.save(f"{args.save_dir}/{args.classname}/sspace_{args.classname}_feat.npy", s_all_latents)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--classname', type=str, default='ffhq', help="place to save the output")
    parser.add_argument('--save_dir', type=str, default='./latent_code', help="place to save the output")
    parser.add_argument('--latent_dir', type=str, default='./latent_code', help="path to the folder that store latent codes in W+ space")
    parser.add_argument('--ckpt', type=str, default='./models/pretrained_models', help="checkpoint file for the generator")
    parser.add_argument('--size', type=int, default=256, help="output size of the generator")
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
    if args.classname == 'ffhq':
        ckpt_path = os.path.join(args.ckpt,f'stylegan2-{args.classname}-config-f.pt')
    else:
        # ckpt_path = os.path.join(args.ckpt,f'stylegan2-{args.classname}','netG.pth')
        ckpt_path = os.path.join(args.ckpt)
    print(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    if args.classname == 'ffhq':
        netG.load_state_dict(checkpoint['g_ema'])
    else:
        # netG.load_state_dict(checkpoint)
        netG.load_state_dict(checkpoint['g_ema'])

    # get mean latent if truncation is applied
    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = netG.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    if args.use_npy:
        generate_from_npy(args, netG, device, mean_latent)
    else:
        generate_from_pt(args, netG, device, mean_latent)
