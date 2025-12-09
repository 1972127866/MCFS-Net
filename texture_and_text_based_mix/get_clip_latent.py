import os
import argparse

from tqdm import tqdm
import clip

import random
import numpy as np
import torch
from torchvision import utils
from utils import stylespace_util
from models.stylegan2.model import Generator
from datasets.latents_dataset import LatentsDataset
from torch.utils.data import DataLoader
from PIL import Image

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


def generate_cspace(args, device):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)


    with torch.no_grad():
        c_latents_list = []
        for i in tqdm(range(args.nums)):
            image_path=os.path.join(args.image_dir,str(i)+".jpg")
            # image_path=os.path.join(args.image_dir,str(i+1).zfill(2) +".png")
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            c_latents = model.encode_image(image)

            c_latents_list.append(c_latents)

        c_all_latents = torch.cat(c_latents_list, dim=0)

        print(c_all_latents.size())

        c_all_latents = c_all_latents.cpu().numpy()

        os.makedirs(os.path.join(args.save_dir, args.classname), exist_ok=True)
        np.save(f"{args.save_dir}/{args.classname}/cspace_{args.classname}_feat.npy", c_all_latents)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--classname', type=str, default='ffhq', help="place to save the output")
    parser.add_argument('--save_dir', type=str, default='./latent_code', help="place to save the output")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--nums', type=int, default=None, help="image num")
    parser.add_argument('--image_dir', type=str, default='./latent_code', help="place to the image")

    args = parser.parse_args()

    device = args.device

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    generate_cspace(args,device)

