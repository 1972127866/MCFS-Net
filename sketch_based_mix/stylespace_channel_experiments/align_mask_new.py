#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:51:06 2020

@author: wuzongze
"""
import os
import pickle 
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import time
import argparse

def ExpendSMask(semantic_masks,num_semantic):#semantic_masks中顺序存放着num_per张32×32的语义分割图

    semantic_masks2=[]
    for i in range(num_semantic):#假设有七个语义类别，則此循环的作用是将单张七值的语义分割图转变成七张二值的语义分割图
        tmp=semantic_masks==i
        semantic_masks2.append(tmp)
    semantic_masks2=np.array(semantic_masks2)#此时的shape为[7,num_per,32,32]
    semantic_masks2=np.transpose(semantic_masks2, [1,0,2,3])#此时的shape为[num_per,7,32,32]
    return semantic_masks2
    
def OverlapScore(mask2,tmp_mask):
    o=tmp_mask.sum() #size of semantic mask
    if o==0:
        return np.nan,np.nan,np.nan
    
    p=o/(mask2.shape[0]*mask2.shape[1])
    
    threshold=np.percentile(mask2.reshape(-1),(1-p)*100)
    gmask=mask2>threshold
    
    n=np.sum(np.logical_and(gmask,tmp_mask))
    u=np.sum(np.logical_or(gmask,tmp_mask))
    
    return n,u,o
    
    
def GetScore(mask2,semantic_mask2):#这里只是计算梯度图和语义图的重叠情况，semantic_channel.py里才是计算相关度
#    scores=np.zeros(len(semantic_mask2))
    scores=[]
    for i in range(len(semantic_mask2)):
        tmp_mask=semantic_mask2[i]
        n,u,o=OverlapScore(mask2,tmp_mask)
        scores.append([n,u,o])
    scores=np.array(scores)
    return scores




#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='predict pose of object')
    
    parser.add_argument('-gradient_folder',default='../autodl-tmp/editGAN/s_with_mask/grad_map',type=str,help='path to gradient_mask_32')
    parser.add_argument('-semantic_path',default='../autodl-tmp/editGAN/s_with_mask/mask/mask_all.npy',type=str,help='path to semantic_mask')
    parser.add_argument('-save_folder',default='../autodl-tmp/editGAN/s_with_mask/scores',type=str,help='path to save folder') 
    
    parser.add_argument('-img_sindex',default='0',type=str,help='path to model file') 
    parser.add_argument('-num_per',default='4',type=str,help='path to model file')
    parser.add_argument('-all_pic_num', default='900', type=str, help='path to model file')
    parser.add_argument('-num_semantic', default='12', type=str, help='path to model file')
    parser.add_argument('-num_layer', default='2', type=str, help='path to model file')

    opt = parser.parse_args()
    
    #%%
    out_size=32
    
    # tmp=os.path.join(opt.gradient_folder,opt.img_sindex)
    # with open(tmp, 'rb') as handle:
    #     var_grad = pickle.load(handle)#五维，[layer,pic_num,该层的channel数,32,32]
    
    semantic_masks_orig=np.load(opt.semantic_path)
    
    #num_semantic=int(semantic_masks.max()+1)#语义类别数量
    num_semantic=int(opt.num_semantic)#语义类别数量
    all_scores = []
    semantic_masks_all=[]
    ###将[pic_num,256,256]的num_semantic值语义分割矩阵分解为[pic_num,num_semantic,32,32]的二值语义分割矩阵
    for cur_index in range(0,int(opt.all_pic_num),int(opt.num_per)):
        semantic_masks=semantic_masks_orig[cur_index:cur_index+int(opt.num_per)]#只取四张图？
        semantic_masks2=ExpendSMask(semantic_masks,num_semantic)#返回值的shape为[num_per,num_semantic,256,256]

        mask_size = semantic_masks2.shape[-1]
        step = int(mask_size / out_size)

        semantic_masks2 = semantic_masks2.reshape(int(opt.num_per), num_semantic, out_size, step, out_size, step)

        semantic_masks2 = np.sum(semantic_masks2, axis=(3, 5))  # 看不懂后面三步
        semantic_masks2_single = np.argmax(semantic_masks2, axis=1)
        semantic_masks2=ExpendSMask(semantic_masks2_single,num_semantic)
        if cur_index==0:
            semantic_masks_all=semantic_masks2
        else:
            semantic_masks_all = np.concatenate([semantic_masks_all, semantic_masks2], axis=0)
        #mask_size=semantic_masks2.shape[-1]
        # step=int(mask_size/out_size)
        #
        # semantic_masks2=semantic_masks2.reshape(int(opt.num_per),num_semantic,out_size,step,out_size,step)
        #
        # semantic_masks2=np.sum(semantic_masks2,axis=(3,5))#看不懂后面三步
        # semantic_masks2_single=np.argmax(semantic_masks2,axis=1)
        #
        # semantic_masks2=ExpendSMask(semantic_masks2_single,num_semantic)

        #%%
        #all_scores=[]
    ###逐层、逐图片、逐s通道地计算语义分割和梯度图的相关度
    for linex in range(opt.num_layer):#逐层
        print('layer index: ',linex)
        #layer_g=var_grad[linex]#layer_g的shape为[pic_num,该层的channel数,32,32]
        layer_g=np.load(os.path.join(opt.gradient_folder,"Layer"+str(linex)+"_gm.npy"))
        num_img,num_channel,_=layer_g.shape

        scores2=np.zeros((num_img,num_channel,num_semantic,3))#3对应于nuo
        for img_index in range(num_img):#逐图片
            semantic_mask=semantic_masks_all[img_index]#取一张图的语义分割图，shape为[num_semantic,32,32]
            for cindex in range(num_channel):#逐通道
                mask=layer_g[img_index,cindex].reshape((out_size,out_size))
                #mask2=np.abs(mask).mean(axis=0)  #need code

                scores=GetScore(mask,semantic_mask)
                scores2[img_index,cindex,:,:]=scores
        #all_scores.append(scores2)
        os.makedirs(opt.save_folder,exist_ok=True)
        tmp=os.path.join(opt.save_folder,'L'+str(linex)+'.npy')
        np.save(tmp,scores2)

    #%%
#     os.makedirs(opt.save_folder,exist_ok=True)
    
#     tmp=os.path.join(opt.save_folder,opt.img_sindex)
#     with open(tmp, 'wb') as handle:
#         pickle.dump(all_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    #%%

    
    