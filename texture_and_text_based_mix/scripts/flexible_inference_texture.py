import os
import sys
sys.path.append(".")
sys.path.append("..")

import copy
import clip
import numpy as np

import torch
import torchvision
from models.stylegan2.model import Generator
from delta_mapper import DeltaMapper
from utils import stylespace_util

import argparse
import re
from typing import List
from tqdm import tqdm
from PIL import Image
from utils import map_tool

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        list_name.append(file_path)

def getClipFromImage(image_path,ori_texture_c,clip_model,preprocess,device,index,save_path,save_name):
    image = Image.open(image_path).convert('RGB')
    # 中心裁剪为50*50
    # image = image.crop((103, 103, 153, 153))
    image.save(os.path.join(save_path,save_name, str(index)+"_"+"texture.jpg"))
    image = preprocess(image).unsqueeze(0).to(device)
    target_texture_c = clip_model.encode_image(image)
    target_texture_c = target_texture_c / target_texture_c.norm(dim=-1, keepdim=True).float()
    delta_c = target_texture_c[0] - ori_texture_c[0]
    delta_c = delta_c / delta_c.norm(dim=-1, keepdim=True).float().clamp(min=1e-5)
    delta_c = torch.cat([ori_texture_c[0], delta_c], dim=0)
    return delta_c.unsqueeze(0)
    

def main(opts):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Initialize generator
    print('Loading stylegan weights from pretrained!')
    g_ema = Generator(size=256, style_dim=512, n_mlp=2)
    g_ema_ckpt = torch.load(opts.stylegan_weights)
    g_ema.load_state_dict(g_ema_ckpt['g_ema'], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    #Initialze DeltaMapper
    net = DeltaMapper()
    net_ckpt = torch.load(opts.checkpoint_path)
    net.load_state_dict(net_ckpt)
    net = net.to(device)
    
    #Load CLIP model
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    os.makedirs(os.path.join(opts.save_path,opts.save_name), exist_ok=True)

    #Initialize test dataset
    batch_size_latent=torch.zeros([1])
    s_latents_list=[]
    w_latents_list=[]
    delta_c_list=[]
    dt_list = []
    ori_texture_c_list=[]
    # 采样衣服
    print("sampling cloths")
    for num in tqdm(range(len(opts.cloth_ids)),position=0):
        w_path=os.path.join(opts.sample_w_path,str(opts.cloth_ids[num])+".npy")
        latent_code=np.load(w_path, allow_pickle=True)
        w = torch.from_numpy(latent_code).to(device).unsqueeze(0)
        #使用w生成s（包含toRGB层）
        style_space, noise = stylespace_util.encoder_latent(g_ema, w)
        s = torch.cat(style_space, dim=1)
        s_latents_list.append(s)
        w_latents_list.append(w)
        # 获取采样服装的texture
        ori_texture=Image.open(os.path.join(opts.sample_cloth_texture_path,str(opts.cloth_ids[num])+".jpg"))
        ori_texture = preprocess(ori_texture).unsqueeze(0).to(device)
        ori_texture_c = clip_model.encode_image(ori_texture)
        ori_texture_c = ori_texture_c / ori_texture_c.norm(dim=-1, keepdim=True).float()
        ori_texture_c_list.append(ori_texture_c)
    # 将目标纹理转为clip特征
    if opts.texture_ids[0]!=-1:
        print("Loading specified texture") 
        for num in tqdm(range(len(opts.texture_ids)),position=0):
            for cloth_num in range(len(ori_texture_c_list)):
                # image_path=os.path.join(opts.texture_path,str(num).zfill(2) +".png")
                # image_path=os.path.join(opts.texture_path,str(opts.texture_ids[num]) +".png")
                # image_path=os.path.join(opts.texture_path,str(num).zfill(3) +".jpg")
                image_path=os.path.join(opts.texture_path,str(opts.texture_ids[num]) +".jpg")
                delta_c = getClipFromImage(image_path,ori_texture_c_list[cloth_num],clip_model,preprocess,device,num,opts.save_path,opts.save_name)
                delta_c_list.append(delta_c)
    else:
        print("Loading all textures in folder")
        path_list = []
        listdir(opts.texture_path, path_list)
        count=-1
        for image_path in tqdm(path_list):#texture
            count+=1
            if count>=0 and count<100:
            # if count>=100 and count<200:
            # if count>=200 and count<300:
            # if count>=300 and count<400:
            # if count>=400 and count<470:
                for cloth_num in range(len(ori_texture_c_list)):#sample cloth
                    delta_c = getClipFromImage(image_path,ori_texture_c_list[cloth_num],clip_model,preprocess,device,count,opts.save_path,opts.save_name)
                    delta_c_list.append(delta_c)
    print("len(delta_c_list):",len(delta_c_list))
    if opts.target_text is not None:
        neutral=opts.neutral
        target_list = opts.target_text.split(',')
        # target_name=opts.target_text.replace(' ', '-').replace(',', '_')
        for target in target_list:
            classnames=[target,neutral]
            dt = map_tool.GetDt(classnames,clip_model)
            dt = torch.Tensor(dt).to(device)
            dt = dt / dt.norm(dim=-1, keepdim=True).float().clamp(min=1e-5)
            dt_list.append(dt)

    print("generating result")
    for texture_index, delta_c in tqdm(enumerate(delta_c_list)): #t0s0,t0s1,t0s2,t1s0,t1s1,t1s2
        cloth_index=texture_index%len(s_latents_list)
        latent_s=s_latents_list[cloth_index]
        latent_w=w_latents_list[cloth_index]
        # for cloth_index, latent_s in enumerate(s_latents_list):
        with torch.no_grad():
            fake_delta_s = net(latent_s, delta_c)
            improved_fake_delta_s = copy.copy(fake_delta_s[0])
            delta_s = improved_fake_delta_s.unsqueeze(0)
            # delta_s = torch.zeros_like(fake_delta_s)
            if opts.target_text is not None:
                for text_index, dt in enumerate(dt_list):
                    delta_c_text=torch.zeros_like(delta_c)
                    delta_c_text[0, :512]=delta_c[0, :512]
                    delta_c_text[0, 512:] = dt
                    fake_delta_s_text = net(latent_s, delta_c_text)
                    #coarse
                    fake_delta_s_text[:, :4*512]=0
                    #medium
                    fake_delta_s_text[:, 5*512:7*512]=0
                    fake_delta_s_text[:, 8*512:10*512]=0
                    #fine
                    fake_delta_s_text[:, 11*512:12*512+256]=0
                    fake_delta_s_text[:, 12*512+256*2 : 12*512 + 256*3 + 128]=0
                    fake_delta_s_text[:, 12*512 + 256*3 + 128*2 : 12*512 + 256*3+ 128*3 + 64]=0
                    # rgb
                    # fake_delta_s_text[:, 4*512:5*512]=0
                    # fake_delta_s_text[:, 7*512:8*512]=0
                    # fake_delta_s_text[:, 10*512:11*512]=0
                    # fake_delta_s_text[:, 12*512+256:12*512+256*2]=0
                    # fake_delta_s_text[:, 12*512 + 256*3 + 128:12*512 + 256*3 + 128*2]=0
                    # fake_delta_s_text[:, 12*512 + 256*3 + 128*3 + 64:]=0
                    delta_s[:, 4*512:5*512]=0
                    delta_s[:, 7*512:8*512]=0
                    delta_s[:, 10*512:11*512]=0
                    delta_s[:, 12*512+256:12*512+256*2]=0
                    delta_s[:, 12*512 + 256*3 + 128:12*512 + 256*3 + 128*2]=0
                    delta_s[:, 12*512 + 256*3 + 128*3 + 64:]=0

                    img_gen = stylespace_util_all_s.decoder_validate(g_ema, latent_s + delta_s + fake_delta_s_text, batch_size_latent)
                    torchvision.utils.save_image(img_gen, os.path.join(opts.save_path,opts.save_name, str(texture_index//len(opts.cloth_ids))+"_"+str(cloth_index)+"_"+target_list[text_index]+".jpg"), normalize=True, range=(-1, 1))
                    s_cpu_0=[]
                    s_out_0=stylespace_util_all_s.split_stylespace_256(latent_s)
                    for i in range(20):
                        s_cpu_0.append(s_out_0[i].cpu())
                    np.save(os.path.join(opts.save_path,opts.save_name, str(cloth_index)+"_orig.npy"),s_cpu_0)
                    s_cpu=[]
                    s_out=latent_s + delta_s + fake_delta_s_text
                    s_out=stylespace_util_all_s.split_stylespace_256(s_out)
                    for i in range(20):
                        s_cpu.append(s_out[i].cpu())
                    np.save(os.path.join(opts.save_path,opts.save_name, str(texture_index//len(opts.cloth_ids))+"_"+str(cloth_index)+"_"+target_list[text_index]+".npy"),s_cpu)
            else:
                img_gen = stylespace_util.decoder_validate(g_ema, latent_s + delta_s, latent_w)
                torchvision.utils.save_image(img_gen, os.path.join(opts.save_path,opts.save_name, str(texture_index//len(opts.cloth_ids))+"_"+str(cloth_index)+".jpg"), normalize=True, range=(-1, 1))
    print(f'completed👍! Please check results in {opts.save_path}/{opts.save_name}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='opt')
    parser.add_argument('-sample_w_path', default='../editGANdata/620t_sample_200000/w', type=str,help='path to sampled w')
    parser.add_argument('-texture_path', default='./', type=str,help='path to texture folder')
    parser.add_argument('-texture_ids', type=num_range, help='ids of texture that needed. If not specified, the entire folder will be traversed', default='-1')
    parser.add_argument('-cloth_ids', type=num_range, help='ids of sampled cloths', default='0')
    parser.add_argument('-avg_latent_path', default='./latent_code/620t_avg_latent_all_s/sspace_620t_avg_latent_all_s_feat.npy', type=str,help='path to the avg latent')
    parser.add_argument('-target_text', type=str, default= None, help='Specify the target attributes to be edited')
    parser.add_argument('-stylegan_weights', default="../editGAN/cloth-v2-620t.pt", type=str, help='')
    parser.add_argument('-checkpoint_path', type=str, default='checkpoints/all_s_extra_mapper_img_loss/resume/620t_sample_200000_all_s/texture_cropped_sample_200000_620t/net_640000.pth')
    parser.add_argument('-save_path', type=str, default='output/flexible')
    parser.add_argument('-save_name', type=str, required=True, help='the name of the folder that save outputs')
    parser.add_argument('-avg_texture_latent_path', default='./latent_code/texture_cropped_sample_avg_620t/cspace_texture_cropped_sample_avg_620t_feat.npy', type=str,help='path to the avg texture latent')
    parser.add_argument('-sample_cloth_texture_path', default='../editGANdata/620t_sample_200000/texture_crop', type=str,help='path to the avg texture latent')
    parser.add_argument('-neutral', type=str,  help='neutral word')
    
    opt = parser.parse_args()

    main(opt)