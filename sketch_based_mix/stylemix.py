from typing import List
import torch
import legacy
import dnnlib
from tqdm import tqdm 
import numpy as np
import os
import PIL.Image
import argparse
import re
import warnings
warnings.filterwarnings("ignore")

# 这个文件应该是只融合潜向量不融合特征图，懒得加注释了，用fm_mix.py吧
def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def creatPics(opt):
    device = torch.device('cuda')
    with dnnlib.util.open_url(opt.stylegan_checkpoint) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
        avg_latent = G.make_mean_latent(4086)

    save_path=opt.output_path
    os.makedirs(save_path, exist_ok=True)
    w_mix=avg_latent

    if opt.sample_pic:
        z = torch.from_numpy(np.random.RandomState(int(opt.source_pic_id)).randn(1, G.z_dim)).to(device)
        w = G.mapping(z.to(device), None)
        w1=0.2*avg_latent+0.8*w
        w_mix=w1
        _,_,s_source = G.synthesis(w1, noise_mode='const',return_style=True)
    else:
        source_data=np.load(os.path.join(opt.latents_path, opt.source_pic_id+'.npy'), allow_pickle=True)
        s_source = []
        for i in range(20):
            s_source.append(source_data[i].cuda())
    source_pic, _, _ = G.synthesis(None, return_style=True, use_styles=True, input_styles=s_source, noise_mode='const')
    img1 = (source_pic.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    PIL.Image.fromarray(img1[0].cpu().numpy(), 'RGB').save(
    os.path.join(save_path, opt.source_pic_id + "_orig.jpg"))

    for num in tqdm(range(len(opt.targets)),position=0):
        if opt.sample_target:
            z = torch.from_numpy(np.random.RandomState(int(opt.targets[num])).randn(1, G.z_dim)).to(device)
            w = G.mapping(z.to(device), None)
            w1=0.2*avg_latent+0.8*w
            # 融合两张图的w，7等价于融合s的0-10层,5等价于融合s的0-7层，3等价于融合s的0-4层
            # for i in range(3):
            #     w_mix[0][i]=w1[0][i]
            _,_,ss = G.synthesis(w1, noise_mode='const',return_style=True)
        else:
            data = np.load(os.path.join(opt.latents_path, str(opt.targets[num])+'.npy'), allow_pickle=True)
            ss = []
            for i in range(20):
                ss.append(data[i].cuda())


        for i in range(len(opt.styles)):
            s_source[opt.styles[i]]=ss[opt.styles[i]]
        mix_pic, _, _ = G.synthesis(None, return_style=True, use_styles=True, input_styles=s_source, noise_mode='const')
        

        img3 = (mix_pic.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img3[0].cpu().numpy(), 'RGB').save(
        os.path.join(save_path,str(opt.targets[num])+"_mix_"+str(opt.source_pic_id) + ".jpg"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='predict pose of object')

    parser.add_argument('-styles', type=num_range, help='Style layer range', default='0-4')
    parser.add_argument('-targets', type=num_range, help='target pic id', default='1-3')
    parser.add_argument('-latents_path', default='../../hdd/wanqing/editGAN/latent_S', type=str,
                        help='path to ')
    parser.add_argument('-source_pic_id', default='1', type=str,
                        help='path to save folder')
    parser.add_argument('-output_path', default='../../hdd/wanqing/editGAN/mix_result', type=str,
                        help='path to ')
    parser.add_argument('-sample_pic', default=False, type=bool, help='sample source?')
    parser.add_argument('-sample_target', default=False, type=bool, help='sample target?')
    parser.add_argument('-stylegan_checkpoint', default="cloth-v2-620t-s-test.pkl", type=str, help='')

    opt = parser.parse_args()

    print("styles",opt.styles)

    creatPics(opt)
            
        
        


