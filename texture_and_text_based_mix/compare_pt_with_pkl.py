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

from utils import stylespace_util_all_s
from delta_mapper_medium import DeltaMapperMedium

import argparse
import dnnlib
import legacy
from utils import stylespace_util


# 对比cloth-v2-620t-s-test.pkl、cloth-v2-620t.pkl和cloth-v2-620t.pt的avg_latent的区别
# 查之前训练用的avg_latent是不是真的avg
#-------------------------------------
# 实验结论
# 在均值向量上：
#     只有cloth-v2-620t-s-test.pkl的avg_latent是正常的,输出shape是[1,14, 512]
#     pt格式的mean_latent的输出shape都是[1, 512]
#     cloth-v2-620t.pkl的make_mean_latent(4086)的输出shape是[4086,1, 512]，如果手动在模型外部跑一次make_mean_latent的代码，则能生成[1,14, 512]的输出(but内容跟cloth-v2-620t-s-test.pkl的avg_latent不同)
#     cloth-v2-620t.pt将mlp层数设为2和8都能跑通，但是输出的结果不一样
# 在随机采样上，对于同一个z：
#     cloth-v2-620t.pkl和cloth-v2-620t.pt输出的w是一样的
#     cloth-v2-620t.pkl和cloth-v2-620t-s-test.pkl输出的w不一样，但是同一个w输出的图片肉眼看不出区别
#     cloth-v2-620t.pt将mlp层数设为2时与前面两个模型相比同一个w输出的图片肉眼看不出区别，设为8则差别很大
# 在随机采样上，对于同一个w：
#     所有模型的输出在肉眼上很相似



def main(opts):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    z0 = torch.from_numpy(np.random.RandomState(0).randn(1, 512)).to(device)
    # z1 = torch.from_numpy(np.random.RandomState(0).randn(1, 512)).to(device)
    

    # cloth-v2-620t.pt mlp=2
    with torch.no_grad():
        g_ema_2 = Generator(size=256, style_dim=512, n_mlp=2)#为什么inference代码里n_mlp是8？
        g_ema_ckpt_2 = torch.load(opts.stylegan_weight_path)
        g_ema_2.load_state_dict(g_ema_ckpt_2['g_ema'], strict=False)
        g_ema_2.eval()
        g_ema_2 = g_ema_2.to(device)
        mean_latent_2 = g_ema_2.mean_latent(4086)
        print("------------------------")
        print("cloth-v2-620t.pt 2mlp")
        print("mean_latent_2.shape:",mean_latent_2.shape)
        # style_space, noise = stylespace_util.encoder_latent(g_ema_2, mean_latent_2)
        # tmp_imgs = stylespace_util.decoder(g_ema_2, style_space, mean_latent_2, noise)
        # torchvision.utils.save_image(tmp_imgs, "avg_620t_pt_2mlp.jpg", normalize=True, range=(-1, 1))
        # print("mean_latent_2:",mean_latent_2)
        _, w_1 = g_ema_2([z0.to(torch.float32)],return_latents=True)
        style_space, noise = stylespace_util.encoder_latent(g_ema_2, w_1)
        tmp_imgs = stylespace_util.decoder(g_ema_2, style_space, w_1, noise)
        torchvision.utils.save_image(tmp_imgs, "z0_620t_pt_2mlp.jpg", normalize=True, range=(-1, 1))

    # cloth-v2-620t.pt mlp=8
    with torch.no_grad():
        g_ema_8 = Generator(size=256, style_dim=512, n_mlp=8)#为什么inference代码里n_mlp是8？
        g_ema_ckpt_8 = torch.load(opts.stylegan_weight_path)
        g_ema_8.load_state_dict(g_ema_ckpt_8['g_ema'], strict=False)
        g_ema_8.eval()
        g_ema_8 = g_ema_8.to(device)
        mean_latent_8 = g_ema_8.mean_latent(4086)
        print("------------------------")
        print("cloth-v2-620t.pt 8mlp")
        print("mean_latent_8.shape:",mean_latent_8.shape)
        # style_space, noise = stylespace_util.encoder_latent(g_ema_8, mean_latent_8)
        # tmp_imgs = stylespace_util.decoder(g_ema_8, style_space, mean_latent_8, noise)
        # torchvision.utils.save_image(tmp_imgs, "avg_620t_pt_8mlp.jpg", normalize=True, range=(-1, 1))
        # print("mean_latent_8:",mean_latent_8)
        _, w_2 = g_ema_8([z0.to(torch.float32)],return_latents=True)
        style_space, noise = stylespace_util.encoder_latent(g_ema_8, w_2)
        tmp_imgs = stylespace_util.decoder(g_ema_8, style_space, w_2, noise)
        torchvision.utils.save_image(tmp_imgs, "z0_620t_pt_8mlp.jpg", normalize=True, range=(-1, 1))
        style_space, noise = stylespace_util.encoder_latent(g_ema_8, w_1)
        tmp_imgs = stylespace_util.decoder(g_ema_8, style_space, w_1, noise)
        torchvision.utils.save_image(tmp_imgs, "z0_620t_pt_8mlp_using_2mlp_w.jpg", normalize=True, range=(-1, 1))

    # print("mean_latent_2.equal(mean_latent_8)：",mean_latent_2.equal(mean_latent_8))
    # print(" w_2.equal( w_1)：",w_2.equal(w_1))
        
    # latent_in = torch.randn(
    #         3, 512, device=torch.device('cuda')
    #     )

    # cloth-v2-620t-s-test.pkl
    with dnnlib.util.open_url(opts.stylegan_path_s) as f:
        G_s = legacy.load_network_pkl(f)['G_ema'].to(device)
        avg_latent_pkl_s = G_s.make_mean_latent(4086)
        print("------------------------")
        print("cloth-v2-620t-s-test.pkl")
        print("avg_latent_pkl_s.shape:",avg_latent_pkl_s.shape)
        image,_ = G_s.synthesis(avg_latent_pkl_s, noise_mode='const')
        torchvision.utils.save_image(image[0], "avg_620t_s_test_pkl.jpg", normalize=True, range=(-1, 1))
        w_3 = G_s.mapping(z0.to(device), None)
        img_3=G_s(z0.to(device), None)
        torchvision.utils.save_image(img_3, "z0_620t_s_test_pkl.jpg", normalize=True, range=(-1, 1))
        # print("avg_latent_pkl_s:",avg_latent_pkl_s)
        # latent_3= G_s.mapping(latent_in ,None)
        # print("latent_3.shape",latent_3.shape)
        # latent_3 =latent_3.mean(0, keepdim=True)
        # print("latent_3.shape",latent_3.shape)

    # cloth-v2-620t.pkl
    with dnnlib.util.open_url(opts.stylegan_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
        avg_latent_pkl = G.make_mean_latent(4086)
        print("------------------------")
        print("cloth-v2-620t.pkl")
        print("avg_latent_pkl.shape:",avg_latent_pkl.shape)
        image,_ = G.synthesis(avg_latent_pkl_s, noise_mode='const')
        torchvision.utils.save_image(image[0], "620t_pkl_use_s_test_pkl_avg.jpg", normalize=True, range=(-1, 1))
        w_4 = G.mapping(z0.to(device), None)
        # print("avg_latent_pkl:",avg_latent_pkl)
        # latent_4= G.mapping(latent_in ,None)
        # print("latent_4.shape",latent_4.shape)
        # latent_4 =latent_4.mean(0, keepdim=True)
        # print("latent_4.shape",latent_4.shape)
        # image,_ = G.synthesis(latent_4, noise_mode='const')
        # torchvision.utils.save_image(image[0], "avg_620t_pkl.jpg", normalize=True, range=(-1, 1))
        img_4=G(z0.to(device), None)
        torchvision.utils.save_image(img_4, "z0_620t_pkl.jpg", normalize=True, range=(-1, 1))

    # print("latent_4.equal(latent_3):",latent_4.equal(latent_3))


    # print(" w_4.equal( w_3)：",w_4.equal(w_3))
    # os.makedirs(os.path.join(opts.save_dir,opts.texture_classname,"use_avg_all_s_extra_mapper",opts.images_classname), exist_ok=True)

    # def make_mean_latent(self, n_latent):
    #     latent_in = torch.randn(
    #         n_latent, self.w_dim, device=torch.device('cuda')
    #     )
    #     latent= self.mapping(latent_in ,None)
    #     latent =latent.mean(0, keepdim=True)
    #     self.mean_latent = latent.cuda()
    #     print(" make_mean_latent ")
    #     return self.mean_latent


    # print(" w_1.equal( w_3)：",w_1.equal(w_3))
    # print(" w_1.equal( w_4)：",w_1.equal(w_4))
    # print(" w_2.equal( w_3)：",w_2.equal(w_3))
    # print(" w_2.equal( w_4)：",w_2.equal(w_4))

    # with torch.no_grad():
    #     img_ori = stylespace_util_all_s.decoder_validate(g_ema, latent_s, latent_w)

    #     img_list = [img_ori]
    #     for delta_s in delta_s_list:
    #         img_gen = stylespace_util_all_s.decoder_validate(g_ema, latent_s + delta_s, latent_w)
    #         img_list.append(img_gen)
    #     img_gen_all = torch.cat(img_list, dim=3)
    #     torchvision.utils.save_image(img_gen_all, os.path.join(opts.save_dir,opts.texture_classname,"use_avg_all_s_extra_mapper",opts.images_classname, "%04d.jpg" %(bid+1)), normalize=True, range=(-1, 1))
    # print(f'completed👍! Please check results in {opts.save_dir}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--classname', type=str, default='ffhq', help="place to save the output")
    parser.add_argument('--save_dir', type=str, default='./latent_code', help="place to save the output")
    parser.add_argument('--stylegan_path_s', type=str, default='../editGAN/cloth-v2-620t-s-test.pkl', help="place to save the output")
    parser.add_argument('--stylegan_weight_path', type=str, default='../editGAN/cloth-v2-620t.pt', help="place to save the output")
    parser.add_argument('--stylegan_path', type=str, default='../editGAN/cloth-v2-620t.pkl', help="place to save the output")

    opts = parser.parse_args()
    main(opts)