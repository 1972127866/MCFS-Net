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
import cv2

### feature map mix with mask

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def gen_orig_mask(img,th):
    h, w = img.shape[:2]  # 获取图像的高和宽
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

def creatPics(opt):
    device = torch.device('cuda')
    with dnnlib.util.open_url(opt.model_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
        avg_latent = G.make_mean_latent(4086)
    
    save_path=opt.output_path
    os.makedirs(save_path, exist_ok=True)
    #获取原图的s
    for source_num in tqdm(range(len(opt.source_pic_ids)),position=0):
        if opt.sample_pic:
            # 由随机采样的噪声生成s，种子是int(opt.source_pic_ids[source_num])
            z = torch.from_numpy(np.random.RandomState(int(opt.source_pic_ids[source_num])).randn(1, G.z_dim)).to(device)
            w = G.mapping(z.to(device), None)
            w1=0.2*avg_latent+0.8*w
            _,_,s_source = G.synthesis(w1, noise_mode='const',return_style=True)
            sample_pic_flag="s"
        else:
            # 从opt.latents_path文件夹中读取名字为str(opt.source_pic_ids[source_num])+'.npy'的潜向量
            source_data=np.load(os.path.join(opt.latents_path, str(opt.source_pic_ids[source_num])+'.npy'), allow_pickle=True)
            s_source = []
            for i in range(20):
                s_source.append(source_data[i].cuda())
            sample_pic_flag=""
        #生成原图并保存
        source_pic, _= G.synthesis(None, return_style=False, use_styles=True, input_styles=s_source, noise_mode='const')
        img1 = (source_pic.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img1[0].cpu().numpy(), 'RGB').save(
        os.path.join(save_path, sample_pic_flag+str(opt.source_pic_ids[source_num]) + "_orig.jpg"))

        # 获取草图的s
        for num in tqdm(range(len(opt.targets)),position=0):
            if opt.sample_target:
                # 由随机采样的噪声生成s，种子是opt.targets[num]
                z = torch.from_numpy(np.random.RandomState(opt.targets[num]).randn(1, G.z_dim)).to(device)
                w = G.mapping(z.to(device), None)
                w1=0.2*avg_latent+0.8*w
                _,_,ss = G.synthesis(w1, noise_mode='const',return_style=True)
                sample_target_flag="s"
            else:
                # 从opt.latents_path文件夹中读取名字为opt.targets[num]+'.npy'的潜向量
                data = np.load(os.path.join(opt.latents_path, str(opt.targets[num])+'.npy'), allow_pickle=True)
                ss = []
                for i in range(20):
                    ss.append(data[i].cuda())
                sample_target_flag=""
            #生成草图并保存
            target_pic, _ = G.synthesis(None, return_style=False, use_styles=True, input_styles=ss, noise_mode='const')
            img2 = (target_pic.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img2[0].cpu().numpy(), 'RGB').save(
            os.path.join(save_path,sample_target_flag+ str(opt.targets[num]) + "_orig.jpg"))
            
            if opt.fm_mix:
                # 根据服装图像生成掩码
                orig_mask=gen_orig_mask(img1[0].cpu().numpy(),opt.th)
                # 根据草图生成掩码
                if opt.use_sketch:
                    #这里打开的是草图本图，不是潜向量
                    sketch=PIL.Image.open(os.path.join(opt.sketch_path,str(opt.targets[num])+'.png'))
                    sketch_mask=gen_sketch_mask(np.array(sketch))
                    mask_pic=cv2.bitwise_and(src1=orig_mask,src2=sketch_mask)
                else:
                    # 跟上面的区别在于这里用的是逆映射再生成的草图，上面用的是手绘的原图
                    target_mask=gen_orig_mask(img2[0].cpu().numpy())
                    mask_pic=cv2.bitwise_and(src1=orig_mask,src2=target_mask)
                # 求两个掩码的交集作为最终的掩码
                mask_pic=cv2.bitwise_not(src=mask_pic)
                mask_pic=torch.from_numpy(np.array(mask_pic, dtype='float32')).unsqueeze(0).repeat(3,1,1)
                mask_pic[mask_pic>0]=1.0
                mask_pic = mask_pic.unsqueeze(0)
                # 草图的潜向量和服装图像的潜向量融合
                for i in range(len(opt.styles)):
                    ss[opt.styles[i]]=s_source[opt.styles[i]]
                # 生成结果 #the shape_style controls the texture while the input_styles controls the shape
                mix_pic, _,  = G.synthesis(None, return_style=False, use_styles=True, input_styles=ss, fm_mix=True,res_mix=opt.res_mix,shape_styles=s_source,mask_pic=mask_pic,noise_mode='const')

                print("mix_pic.shape",mix_pic.shape)
                mix_pic=mix_pic* 127.5 + 128

                # 提取掩码图的轮廓
                mask_pic_dim3=mask_pic.squeeze(0).to(torch.uint8)*255
                tmp=cv2.bitwise_not(src=mask_pic_dim3[0].cpu().numpy())
                contours, _ = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(tmp,contours,-1,(10,0,0),5) #(0,0,255)为轮廓颜色，5为轮廓线粗细 
                tmp=cv2.GaussianBlur(tmp, (15, 15), 0, 0)
                tmp=torch.from_numpy(tmp).to(torch.float32).cuda()
                tmp=tmp/255
                mix_realflag=""

                # 根据掩码图的轮廓将真实服装图像和生成的服装图像进行拼接
                if opt.mix_real:
                    source_pic_1=np.array(PIL.Image.open(os.path.join(opt.sketch_path,str(opt.source_pic_ids[source_num]).zfill(5)+'.jpg')).convert("RGB"))
                    source_pic=torch.from_numpy(source_pic_1).to(torch.float32).permute(2,0,1).unsqueeze(0).cuda()
                    mix_pic= source_pic * tmp + mix_pic* (1 - tmp)
                    mix_realflag="_mix_real"
            else:
                # 不融合特征图只融合潜向量
                for i in range(len(opt.styles)):
                    s_source[opt.styles[i]]=ss[opt.styles[i]]
                mix_pic, _, _ = G.synthesis(None, return_style=True, use_styles=True, input_styles=s_source, noise_mode='const')
                mix_pic=mix_pic* 127.5 + 128


            img3 = (mix_pic.permute(0, 2, 3, 1)).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img3[0].cpu().numpy(), 'RGB').save(
            os.path.join(save_path, sample_pic_flag+str(opt.source_pic_ids[source_num])+"_mix_"+sample_target_flag+str(opt.targets[num]) + ".jpg"))
       


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='predict pose of object')

    # num_range是一种自定义的数据结构，可以用a-c或a,b,c的形式表示目标序号
    parser.add_argument('-styles', type=num_range, help='Layers of latent code fusion', default='7-19')
    parser.add_argument('-targets', type=num_range, help='target sketch ids', default='1-3')
    parser.add_argument('-source_pic_ids', default='1', type=num_range,help='the ids of latent codes or the seeds for noise sampling')
    parser.add_argument('-latents_path', default='/home/scut/hdd/wanqing/editGAN/latent_S', type=str,
                        help='The folder where the latent codes are stored')
    parser.add_argument('-output_path', default='../../hdd/wanqing/editGAN/fm_mix', type=str,
                        help='path to save results')
    parser.add_argument('-sample_pic', default=False, type=bool, help='sample pic or use existing pics')
    parser.add_argument('-res_mix', default=64, type=int, help='Controls when feature maps are fused. Can be 32, 64, 128, 256')
    parser.add_argument('-fm_mix', default=True, type=bool, help='Whether to fuse feature maps')
    parser.add_argument('-use_sketch', default=False, type=bool, help='Whether to use sketch')
    parser.add_argument('-sample_target', default=False, type=bool, help='just keep it false')
    parser.add_argument('-sketch_path', default='/home/scut/hdd/wanqing/editGAN/sketch', type=str,help='path to sketch image')
    parser.add_argument('-model_path', default='cloth-v2-620t-s-test-fm.pkl', type=str,help='path to stylegan checkpoint')
    parser.add_argument('-mix_real', default=False, type=bool,help='whether to mix generated image with original image')
    parser.add_argument('-th', default=200, type=int,help='Controls the threshold for mask generation')


    opt = parser.parse_args()

    print("styles",opt.styles)

    creatPics(opt)
            
        
        


