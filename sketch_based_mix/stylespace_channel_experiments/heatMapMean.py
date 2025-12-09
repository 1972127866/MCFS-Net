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

search_array=[]
count_s=0
for l_index in range(20):
    if l_index%3==1:
        search_array.append(8000)
        continue
    if l_index<12:
        channel_num=512
    elif l_index<15:
        channel_num = 256
    elif l_index<18:
        channel_num = 128
    else:
        channel_num = 64
    search_array.append(count_s)
    count_s+=channel_num


def findChannel(opt,sema_inx):
    scores=np.load(opt.semantic_scores_path,allow_pickle=True)
    scores_c=np.concatenate(scores)
    #print(scores_c.shape)#(4928,12)
    target_index=(sema_inx,)
    top_sum=scores_c[:,target_index].sum(axis=1)#选中的类别们的分数加和，当只选一个类别时，等价于直接取该类别的分数

    tmp=list(np.arange(12))
    for i in target_index:
        tmp.remove(i)
    tmp=scores_c[:,tmp]
    second_max=tmp.max(axis=1)#second_max是非选中类别中得分最高的类别的分数
    select1=top_sum>0.5#select1和2都是01数组
    #print(select1.sum))
    select2=top_sum-second_max>0.25
    #print(select2.sum())

    select=np.logical_and(select1,select2)#select是一个bool数组
    findex=np.arange(len(select))[select]
    print(findex)
    return findex

def getLC(index):
    for i in range(20):
        if index>=search_array[19-i]:
            layer=19-i
            channel=index- search_array[19-i]
            return layer,channel

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


def creatPics(opt,l_c_list=None):
    device = torch.device('cuda')
    with dnnlib.util.open_url("cloth-v2-620t-s-test.pkl") as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    if not opt.special_channel:
        findex=findChannel(opt,int(opt.sema_inx))
        l_c_list = []
        print("calculating the layers index and channels index that effect the semantic most")
        for i in range(len(findex)):
            l_c_list.append(getLC(findex[i]))
            print("l",l_c_list[i][0],"c",l_c_list[i][1])


    # diverse_pos_mean = np.zeros((256, 256))
    # diverse_neg_mean = np.zeros((256, 256))
    # diverse_mean = np.zeros((256, 256))
    # for picid in tqdm(range(int(opt.pic_num)),position=0):
    #     data = np.load(os.path.join(opt.latents_path, str(picid)+'.npy'), allow_pickle=True)
    #     ss = []
    #     ss_pos = []
    #     ss_neg = []
    #     for i in range(20):
    #         ss.append(data[i].cuda())
    #         ss_pos.append(ss[i].clone())
    #         ss_neg.append(ss[i].clone())

    #     save_path=os.path.join(opt.output_path, "class_"+opt.sema_inx)
    #     os.makedirs(save_path, exist_ok=True)

    #     output_orig, _, _ = G.synthesis(None, return_style=True, use_styles=True, input_styles=ss, noise_mode='const')
    #     img1 = (output_orig.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    #     #PIL.Image.fromarray(img1[0].cpu().numpy(), 'RGB').save(
    #         #os.path.join(opt.output_path, "class_"+opt.sema_inx, str(picid+200) + "_orig.jpg"))

    for j in range(len(l_c_list)):
        layer,channel=l_c_list[j]
        diverse_pos_mean = np.zeros((256, 256))
        diverse_neg_mean = np.zeros((256, 256))
        diverse_mean = np.zeros((256, 256))
        for picid in tqdm(range(int(opt.pic_num)),position=0):
            data = np.load(os.path.join(opt.latents_path, str(picid)+'.npy'), allow_pickle=True)
            ss = []
            ss_pos = []
            ss_neg = []
            for i in range(20):
                ss.append(data[i].cuda())
                ss_pos.append(ss[i].clone())
                ss_neg.append(ss[i].clone())

            save_path=os.path.join(opt.output_path, "class_"+opt.sema_inx)
            os.makedirs(save_path, exist_ok=True)

            output_orig, _, _ = G.synthesis(None, return_style=True, use_styles=True, input_styles=ss, noise_mode='const')
            img1 = (output_orig.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            #PIL.Image.fromarray(img1[0].cpu().numpy(), 'RGB').save(
                #os.path.join(opt.output_path, "class_"+opt.sema_inx, str(picid+200) + "_orig.jpg"))


        # for j in range(len(l_c_list)):
        #     layer,channel=l_c_list[j]
            #print('ss', ss[layer][0][channel])
            if ss[layer][0][channel]>5 or ss[layer][0][channel]<-5:
                bias=50.
            else:
                bias=40.
            ss_pos[layer][0][channel] += bias
            #print("ss_pos", ss_pos[layer][0][channel])
            ss_neg[layer][0][channel] -= bias
            #print("ss_neg", ss_neg[layer][0][channel])
            #print("ss", ss[layer][0][channel])

            save_path=os.path.join(opt.output_path, "class_"+opt.sema_inx,str(layer)+"_"+str(channel))
            os.makedirs(save_path, exist_ok=True)

            output_pos, _, _ = G.synthesis(None, return_style=True, use_styles=True, input_styles=ss_pos, noise_mode='const')
            img2 = (output_pos.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            # PIL.Image.fromarray(img2[0].cpu().numpy(), 'RGB').save(
            #     os.path.join(save_path,str(picid+200)+"_pos"+str(round(ss[layer][0][channel].item()))+".jpg"))

            output_neg, _, _ = G.synthesis(None, return_style=True, use_styles=True, input_styles=ss_neg, noise_mode='const')
            img3 = (output_neg.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            # PIL.Image.fromarray(img3[0].cpu().numpy(), 'RGB').save(
            #     os.path.join(save_path,str(picid+200)+"_neg"+str(round(ss[layer][0][channel].item()))+".jpg"))

            ### heatmap
            #将图片还原成[channel,w,h]的形式，转为float32防止计算差值时溢出
            img1_cpu=np.transpose(img1[0].cpu().numpy(), [2, 0, 1]).astype(np.float32)
            img2_cpu = np.transpose(img2[0].cpu().numpy(), [2, 0, 1]).astype(np.float32)
            img3_cpu = np.transpose(img3[0].cpu().numpy(), [2, 0, 1]).astype(np.float32)

            diverse_pos = np.zeros((256, 256))
            diverse_neg = np.zeros((256, 256))
            for i in range(256):
                for j in range(256):
                    diverse_pos[i][j] = calDiverse(img1_cpu[:,i,j], img2_cpu[:,i,j])
                    diverse_neg[i][j] = calDiverse(img1_cpu[:, i, j], img3_cpu[:, i, j])
            norm_diverse_pos = maxminnorm(diverse_pos)
            norm_diverse_neg = maxminnorm(diverse_neg)
            diverse_pos_mean+=norm_diverse_pos
            diverse_neg_mean+=norm_diverse_neg
            diverse_mean=diverse_mean+diverse_neg_mean+diverse_pos_mean
            ss_pos[layer][0][channel] -= bias
            # print("ss_pos", ss_pos[layer][0][channel])
            ss_neg[layer][0][channel] += bias


        heatmap_pos = sns.heatmap(diverse_pos_mean, cmap="YlGnBu_r", xticklabels=False, yticklabels=False)
        heatmap_pos.get_figure().savefig(os.path.join(save_path,opt.pic_num+"mean_pos_heatmap"+".png"))
        plt.close()
        heatmap_neg = sns.heatmap(diverse_neg_mean, cmap="YlGnBu_r", xticklabels=False, yticklabels=False)
        heatmap_neg.get_figure().savefig(os.path.join(save_path,opt.pic_num+"mean_neg_heatmap"+".png"))
        plt.close()
        heatmap_all = sns.heatmap(diverse_mean, cmap="YlGnBu_r", xticklabels=False, yticklabels=False)
        heatmap_neg.get_figure().savefig(os.path.join(save_path,opt.pic_num+"mean_heatmap"+".png"))
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='predict pose of object')

    parser.add_argument('-semantic_scores_path', default='../../hdd/wanqing/editGAN/scores/semantic_top_32.npy', type=str,
                        help='path to ')
    parser.add_argument('-latents_path', default='../../hdd/wanqing/editGAN/latent_S', type=str,
                        help='path to ')
    parser.add_argument('-pic_num', default='1', type=str,
                        help='path to save folder')
    parser.add_argument('-output_path', default='../../hdd/wanqing/editGAN/heatmap_mean', type=str,
                        help='path to ')
    parser.add_argument('-sema_inx', default='5', type=str, help='the target semantic class')
    parser.add_argument('-special_channel', default=False, type=bool, help='choss specal channel or not')
    opt = parser.parse_args()
    # l_c_list=[(6,126),(8,279),(3,25),(5,82)]
    l_c_list=[(6,175)]
    if opt.special_channel:
        creatPics(opt,l_c_list=l_c_list)
    else:
        creatPics(opt)

