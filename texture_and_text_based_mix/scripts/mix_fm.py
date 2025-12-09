import os
import sys
sys.path.append(".")
sys.path.append("..")

import numpy as np

import torch
import torchvision
from models.stylegan2.model import Generator
from utils import stylespace_util_all_s

import argparse
import re
from typing import List
from tqdm import tqdm
import cv2
import PIL

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

def gen_orig_mask(img,th):
    h, w = img.shape[0:2]  # 获取图像的高和宽
    blured = cv2.blur(img, (1, 1))  # 进行滤波去掉噪声，(1,1)表示取原图
    mask = np.zeros((h + 2, w + 2), np.uint8)  # 掩码长和宽都比输入图像多两个像素点，满水填充不会超出掩码的非零边缘
    cv2.floodFill(blured, mask, (w - 1, h - 1), (255, 255, 255), (1, 1, 1), (1, 1, 1), 8)
    gray = cv2.cvtColor(blured, cv2.COLOR_BGR2GRAY) # 得到灰度图
    _, binary = cv2.threshold(gray, th, 200, cv2.THRESH_BINARY)# 求二值图，大于阈值的变255(白)，其余的变黑（白底黑衣）
    orig_mask=cv2.bitwise_not(src=binary)#取反（黑底白衣）。上一句的阈值越高，这里的白衣部分越大
    return orig_mask

def gen_sketch_mask(img):
    h, w = img.shape[:2]  # 获取图像的高和宽
    mask = np.zeros((h+2, w+2), np.uint8)  # 掩码长和宽都比输入图像多两个像素点，满水填充不会超出掩码的非零边缘
    mask_fill = 255
    flags = 4|(mask_fill<<8)|cv2.FLOODFILL_FIXED_RANGE
    # 进行泛洪填充img中以中点为起点，附近所有跟中点差值绝对值小于5的变白色
    cv2.floodFill(img, mask, (128, 128), (255, 255, 255), (5, 5, 5), (5, 5, 5), flags)#描边的内侧变白，描边及描边外侧变黑（黑底白衣）
    sketch_mask=mask[1:257,1:257]
    return sketch_mask

def main(opts):

    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"

        #加载生成器
        print('Loading stylegan weights from pretrained!')
        g_ema = Generator(size=256, style_dim=512, n_mlp=2)
        g_ema_ckpt = torch.load(opts.stylegan_weights)
        g_ema.load_state_dict(g_ema_ckpt['g_ema'], strict=False)
        g_ema.eval()
        g_ema = g_ema.to(device)

        # 加载服装向量
        for source_num in tqdm(range(len(opt.source_pic_ids)),position=0):
            batch_size_latent=torch.zeros([1])  
            if opts.sample_pic:
                w_path=os.path.join(opts.sample_w_path,str(opt.source_pic_ids[source_num])+".npy")
                latent_code=np.load(w_path, allow_pickle=True)
                w = torch.from_numpy(latent_code).to(device).unsqueeze(0)
                style_space, noise = stylespace_util_all_s.encoder_latent(g_ema, w)
                s = torch.cat(style_space, dim=1)
                s = stylespace_util_all_s.split_stylespace_256(s)
            else:
                s_path=os.path.join(opts.s_path,str(opt.source_pic_ids[source_num])+".npy")
                latent_code=np.load(s_path, allow_pickle=True)
                s = []
                for i in range(20):
                    s.append(latent_code[i].cuda())
            img1= stylespace_util_all_s.decoder_validate_splitted_s(g_ema, s, batch_size_latent)
            img1 = (img1.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            # 加载草图向量
            for num in tqdm(range(len(opt.targets)),position=0):
                data = np.load(os.path.join(opt.sketch_latents_path, str(opt.targets[num])+'.npy'), allow_pickle=True)
                ss = []
                for i in range(20):
                    ss.append(data[i].cuda())
                # 将草图向量跟服装向量融合
                for i in range(len(opt.styles)):
                    ss[opt.styles[i]]=s[opt.styles[i]]

                # 生成掩码
                orig_mask=gen_orig_mask(img1[0].cpu().numpy(),opt.th)
                #这里打开的是草图本图，不是latent
                sketch=PIL.Image.open(os.path.join(opt.sketch_path,str(opt.targets[num])+'.png'))
                sketch_mask=gen_sketch_mask(np.array(sketch))
                mask_pic=cv2.bitwise_and(src1=orig_mask,src2=sketch_mask)
                mask_pic=cv2.bitwise_not(src=mask_pic)
                mask_pic=torch.from_numpy(np.array(mask_pic, dtype='float32')).unsqueeze(0).repeat(3,1,1)
                torchvision.utils.save_image(mask_pic, os.path.join("scripts/mask.jpg"), normalize=True, range=(-1, 1))
                mask_pic[mask_pic>0]=1.0
                mask_pic = mask_pic.unsqueeze(0)
                img_gen = stylespace_util_all_s.decoder_fm_mix(g_ema, s ,ss, batch_size_latent,layer_id=14,mask=mask_pic)
                torchvision.utils.save_image(img_gen, os.path.join(opts.save_path,str(opt.source_pic_ids[source_num])+"_mix_"+str(opt.targets[num])+"_th_"+str(opts.th)+".jpg"), normalize=True, range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='opt')
    parser.add_argument('-sample_w_path', default='../editGANdata/620t_sample_200000/w', type=str,help='path to sampled w')
    parser.add_argument('-stylegan_weights', default="../editGAN/cloth-v2-620t.pt", type=str, help='')
    parser.add_argument('-save_path', type=str, default='output/fm_mix')
    parser.add_argument('-sketch_latents_path', default='/home/scut/hdd/wanqing/editGAN/latent_S', type=str,
                        help='path to the folder that save the latent code of  sketches')
    parser.add_argument('-styles', type=num_range, help='Style layer range', default='7-19')
    parser.add_argument('-targets', type=num_range, help='target pic id, usually is the id of the sketch', default='1007')
    parser.add_argument('-sample_pic', default=False, type=bool, help='sample pic or use existing pics')
    parser.add_argument('-source_pic_ids', default='1', type=num_range, help='the ids of latent codes or the seeds for noise sampling')
    parser.add_argument('-s_path', type=str, default='/home/scut/hdd/wanqing/editGAN/latent_S',help='The folder where the latent codes are stored')
    parser.add_argument('-sketch_path', default='/home/scut/hdd/wanqing/editGAN/sketch', type=str,help='path to sketch image')
    parser.add_argument('-th', default=200, type=int,help='Controls the threshold for mask generation')
    
    opt = parser.parse_args()

    main(opt)