import argparse
from matplotlib import pyplot as plt
import numpy as np
import PIL.Image
import copy
import torch
import dnnlib
import legacy
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
from tqdm import tqdm
import seaborn as sns


def calDiverse(orig_color,edit_color):
    #计算两个点的三通道值之差的绝对值之和
    diverse = np.sum(np.abs(orig_color-edit_color))
    return diverse

def maxminnorm(array):
    maxcols=array.max(axis=0)
    mincols=array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t=np.empty((data_rows,data_cols))
    for i in range(data_cols):
        t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i]+0.001)
    return t


def creatPics(opt,lc_list):
    device = torch.device('cuda')
    with dnnlib.util.open_url("cloth-v2-620t-s-test.pkl") as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
        avg_latent = G.make_mean_latent(4086)

    # sample_pic_seed=[10,167,288,448,468,526,628,650,671,759,974,993,1063,1082,1088,
    #     1090,1163,1276,1278,1323,1399,1403,1444,1535,2035,2184,2634,
    #     2784,2832,2962,3173,3226,3270,3436,3483,3715,3850,3927,4233,
    #     4280,4350,4404,4423,4660,4684,4725,4982,4996,4999,2529,2721,3974,4725]
    sample_pic_seed=[2962]

    for n in tqdm(range(len(sample_pic_seed)),position=0):
        z = torch.from_numpy(np.random.RandomState(sample_pic_seed[n]).randn(1, G.z_dim)).to(device)
        w = G.mapping(z.to(device), None)
        w1=0.2*avg_latent+0.8*w
        _,_,ss = G.synthesis(w1, noise_mode='const',return_style=True)
        ss_pos = []
        ss_neg = []
        for i in range(20):
            ss_pos.append(ss[i].clone())
            ss_neg.append(ss[i].clone())

        save_path=os.path.join(opt.output_path, "class_"+opt.sema_inx)
        os.makedirs(save_path, exist_ok=True)

        output_orig, _, _ = G.synthesis(None, return_style=True, use_styles=True, input_styles=ss, noise_mode='const')
        img1 = (output_orig.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img1[0].cpu().numpy(), 'RGB').save(
            os.path.join(opt.output_path, "class_"+opt.sema_inx, "S"+str(sample_pic_seed[n]) + "_orig.jpg"))

        for j in range(len(lc_list)):
            layer,channel=lc_list[j]
            #print('ss', ss[layer][0][channel])
            if ss[layer][0][channel]>5 or ss[layer][0][channel]<-5:
                bias=100.
            else:
                bias=70.
            ss_pos[layer][0][channel] += bias
            #print("ss_pos", ss_pos[layer][0][channel])
            ss_neg[layer][0][channel] -= bias
            #print("ss_neg", ss_neg[layer][0][channel])
            #print("ss", ss[layer][0][channel])

            save_path=os.path.join(opt.output_path, "class_"+opt.sema_inx,str(layer)+"_"+str(channel))
            os.makedirs(save_path, exist_ok=True)

            output_pos, _, _ = G.synthesis(None, return_style=True, use_styles=True, input_styles=ss_pos, noise_mode='const')
            img2 = (output_pos.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img2[0].cpu().numpy(), 'RGB').save(
                os.path.join(save_path,"S"+str(sample_pic_seed[n])+"_pos"+str(round(ss[layer][0][channel].item()))+".jpg"))

            output_neg, _, _ = G.synthesis(None, return_style=True, use_styles=True, input_styles=ss_neg, noise_mode='const')
            img3 = (output_neg.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img3[0].cpu().numpy(), 'RGB').save(
                os.path.join(save_path,"S"+str(sample_pic_seed[n])+"_neg"+str(round(ss[layer][0][channel].item()))+".jpg"))

            ### heatmap
            #将图片还原成[channel,w,h]的形式，转为float32防止计算差值时溢出
            # img1_cpu=np.transpose(img1[0].cpu().numpy(), [2, 0, 1]).astype(np.float32)
            # img2_cpu = np.transpose(img2[0].cpu().numpy(), [2, 0, 1]).astype(np.float32)
            # img3_cpu = np.transpose(img3[0].cpu().numpy(), [2, 0, 1]).astype(np.float32)

            # diverse_pos = np.zeros((256, 256))
            # diverse_neg = np.zeros((256, 256))
            # for i in range(256):
            #     for j in range(256):
            #         diverse_pos[i][j] = calDiverse(img1_cpu[:,i,j], img2_cpu[:,i,j])
            #         diverse_neg[i][j] = calDiverse(img1_cpu[:, i, j], img3_cpu[:, i, j])
            # norm_diverse_pos = maxminnorm(diverse_pos)
            # norm_diverse_neg = maxminnorm(diverse_neg)

            # heatmap_pos = sns.heatmap(norm_diverse_pos, cmap="YlGnBu_r", xticklabels=False, yticklabels=False)
            # heatmap_pos.get_figure().savefig(os.path.join(save_path,str(picid+200)+"_pos_heatmap"+".png"))
            # plt.close()
            # heatmap_neg = sns.heatmap(norm_diverse_neg, cmap="YlGnBu_r", xticklabels=False, yticklabels=False)
            # heatmap_neg.get_figure().savefig(os.path.join(save_path,str(picid+200)+"_neg_heatmap"+".png"))
            # plt.close()
            ss_pos[layer][0][channel] -= bias
            # print("ss_pos", ss_pos[layer][0][channel])
            ss_neg[layer][0][channel] += bias



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='predict pose of object')

    parser.add_argument('-semantic_scores_path', default='../../hdd/wanqing/editGAN/scores/semantic_top_32.npy', type=str,
                        help='path to ')
    parser.add_argument('-latents_path', default='../../hdd/wanqing/editGAN/latent_S', type=str,
                        help='path to ')
    parser.add_argument('-pic_num', default='1', type=str,
                        help='path to save folder')
    parser.add_argument('-output_path', default='../../hdd/wanqing/editGAN/sample_pic_modi', type=str,
                        help='path to ')
    parser.add_argument('-sema_inx', default='2', type=str, help='the target semantic class')
   
    opt = parser.parse_args()
    lc_list=[(5,193)]
    creatPics(opt,lc_list)

