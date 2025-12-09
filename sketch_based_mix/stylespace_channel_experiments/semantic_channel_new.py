#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 18:51:36 2020|

@author: wuzongze
"""

import pickle 
import numpy as np
import pandas as pd
import argparse
import os 

def LoadAMask(opt):
    all_var_grad=[]
    for i in range(20):
        if i%3==1:
            continue
        tmp = os.path.join(opt.align_folder, 'L'+str(i)+'.npy')
        var_grad=np.load(tmp)
        all_var_grad.append(var_grad)
    print('len(all_var_grad)',len(all_var_grad))
    print('all_var_grad[0].shape',all_var_grad[0].shape)


    # for i in range(0,1000,int(opt.num_per)):
    #     try:
    #         tmp=os.path.join(opt.align_folder,str(i))
    #         with open(tmp, 'rb') as handle:
    #             var_grad = pickle.load(handle)#[20,4,channel_num,12,3]
    #
    #         if not 'all_var_grad' in locals():
    #             num_layer=len(var_grad)
    #             all_var_grad=[[] for i in range(num_layer)]
    #
    #         for k in range(num_layer):
    #             all_var_grad[k].append(var_grad[k])
    #     except FileNotFoundError:
    #         print(i)
    #         continue
    #
    # for i in range(num_layer):
    #     all_var_grad[i]=np.concatenate(all_var_grad[i])#结果应该是list(len=20),每一层是一个[pic_num,channel_num,12,3]
    # print('num of sample:',all_var_grad[0].shape[0])
    # return all_var_grad#结果应该是list(len=20),每一层是一个[pic_num,channel_num,12,3]
    

def TopRate(all_var_grad):
    num_layer=len(all_var_grad)
    num_semantic=all_var_grad[0].shape[2]
    discount_factor=2 #large number means pay higher weight precision (prefer small area) 
    all_count_top=[]
    for lindex in range(num_layer):
        layer_g=all_var_grad[lindex]#shape is [pic_num,channel_num,12,3]
        num_channel=layer_g.shape[1]
        count_top=np.zeros([num_channel,num_semantic])#对于每一个通道，统计900个s的最高分语意类别出现的次数
        for cindex in range(num_channel):
            semantic_in=layer_g[:,cindex,:,0]/(layer_g[:,cindex,:,2]**discount_factor)#计算900个s在指定通道上的各个语意类别的分数
            semantic_top=np.nanargmax(np.abs(semantic_in),axis=1)#找到各个图片的针对指定通道的语义分数最高的类别编号
            
            semantic_top=pd.Series(semantic_top)
            tmp=semantic_top.value_counts()#针对cindex这个通道，计算900张图片的语义类别编号出现的个数
            count_top[cindex,tmp.index]=tmp.values#count_top[12,3]=42 表示在lindex层的第12通道，有42张图片的语义分数最高的语意类别是第3类
        all_count_top.append(count_top)
    
    tmp=all_var_grad[0][:,0,:,2]#shape=[900,12],如果梯度图和语义区域没有重叠部分，则该位置值为nan
    mask_counts2=~np.isnan(tmp)#shape=[900,12],如果值为nan则为false，否则为true
    mask_counts3=mask_counts2.sum(axis=0)#shape=[12,],计算900张梯度图中与各语义区域有重叠部分的各有几张
    mask_counts3[mask_counts3==0]=1 # ignore 0 ,消除数组中的0
    
    all_count_top2=[]
    for lindex in range(len(all_count_top)):#len(all_count_top)=20-trgb层数，all_count_top[0].shape=[channel_num,12]
        all_count_top2.append(all_count_top[lindex]/mask_counts3)#除数是为了降低像背景这样大面积的语义区域的权重
    return all_count_top2

def PadTRGB(opt,all_count_top2):
    # with open(opt.s_path, "rb") as fp:   #Pickling
    #     s_names,all_s=pickle.load( fp)
    #
    tmp_index=0
    all_count_top3=[[] for i in range(20)]
    num_sa=all_count_top2[0].shape[1]#12
    #
    #
    # for i in range(len(s_names)):#补齐20层(trgb层补0矩阵)
    #     s_name=s_names[i]
    #     if 'ToRGB' in s_name:
    #         tmp=np.zeros([all_s[i].shape[1],num_sa])#[channel_num,12]
    #     else:
    #         tmp=all_count_top2[tmp_index]
    #         tmp_index+=1
    #     all_count_top3[i]=tmp

    
    all_count_top2=all_count_top3
    return all_count_top2

#%%
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='predict pose of object')
    
    parser.add_argument('-align_folder',default='../autodl-tmp/editGAN/s_with_mask/scores',type=str,help='path to align_mask_32 folder')
    parser.add_argument('-s_path',default='./npy/ffhq/S',type=str,help='path to ') 
    parser.add_argument('-save_folder',default='../autodl-tmp/editGAN/s_with_mask/scores',type=str,help='path to save folder')
    
    parser.add_argument('-num_per',default='4',type=str,help='path to model file') 
    parser.add_argument('-include_trgb', action='store_true')
    
    opt = parser.parse_args()
    
    #%%
    all_var_grad=LoadAMask(opt)
    all_count_top2=TopRate(all_var_grad)
    # if not opt.include_trgb:
    #     all_count_top2=PadTRGB(opt,all_count_top2)
    #%%
    tmp=os.path.join(opt.save_folder,'semantic_top_32.npy')
    np.save(tmp,all_count_top2)
    # with open(tmp, 'wb') as handle:
    #     pickle.dump(all_count_top2, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
    #%%
    
    
    
    
    
    
    
    
    
    
    
    
    
    