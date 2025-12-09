import torch
import dnnlib
import numpy as np
import legacy
import matplotlib.pyplot as plt
import PIL.Image
import os
from tqdm import tqdm
print('Loading networks...')
device = torch.device('cuda')

# 这是一个由随机采样的噪声生成图片以及对应的潜向量的代码，使用时注意checkpoint路径是否正确

#生成S空间的潜向量
# with dnnlib.util.open_url('cloth-v2-620t-s-test.pkl') as f:
#   g_all = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
#   avg_latent = g_all.make_mean_latent(4086)
# save_path=os.path.join('../../hdd/wanqing/editGAN/sample_pic')
# os.makedirs(save_path, exist_ok=True)
# for seed in tqdm(range(5000),position=0):
#     seed+=5000
#     z = torch.from_numpy(np.random.RandomState(seed).randn(1, g_all.z_dim)).to(device)
#     w = g_all.mapping(z.to(device), None)
#     w1=0.2*avg_latent+0.8*w
#     _,_,s = g_all.synthesis(w1, noise_mode='const',return_style=True)
#     image,_,_=g_all.synthesis(None,return_style=True,use_styles=True,input_styles=s,noise_mode='const')
#     image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
#     PIL.Image.fromarray(image, 'RGB').save(os.path.join(save_path,str(seed)+'.jpg'))


#生成W+空间的潜向量
with dnnlib.util.open_url("../../hdd/wanqing/editGAN/pretrained_styleGAN/collar_cloth_8385_300t.pkl") as f:
  g_all = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
save_path=os.path.join('../../hdd/wanqing/editGAN/pretrained_styleGAN/collar_cloth_8385_300t_sample')
os.makedirs(save_path, exist_ok=True)
for seed in tqdm(range(200),position=0):
    seed+=200
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, g_all.z_dim)).to(device)
    w = g_all.mapping(z.to(device), None)
    image = g_all.synthesis(w, noise_mode='const')
    image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(image, 'RGB').save(os.path.join(save_path,str(seed)+'.jpg'))
    w_cpu=w.squeeze(0).cpu().numpy()
    np.save(f'{save_path}/%s.npy'%str(seed),w_cpu)
