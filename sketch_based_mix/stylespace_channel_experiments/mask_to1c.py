import numpy as np
import os
from PIL import Image
from tqdm import tqdm

def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        list_name.append(file_path)

def check_color(x,count,i,j):
    # cloth_12_palette = [
    #     144, 201, 231,  # 浅蓝 衣服主体1
    #     33, 158, 188,  # 湖蓝 袖子2
    #     254, 183, 5,  # 深黄 领子3
    #     2, 48, 74,  # 深蓝 纽扣4
    #     19, 103, 131,  # 深湖蓝 领口5
    #     239, 65, 67,  # 红 印花6
    #     42, 157, 140,  # 青 条纹7
    #     138, 176, 125,  # 绿 格纹8
    #     251, 132, 2,  # 橙 吊带9
    #     131, 64, 38,  # 棕 吊带10
    #     0, 0, 0,  # 黑 领带11
    #     255, 255, 255,  # 白 background 0
    # ]
    if type(x==[144,201,231]) == bool:
        print(count,i,j,"have wrong color")
        return 20
    if (x==[144,201,231]).all():
        return 1
    if (x==[33, 158, 188]).all():
        return 2
    if (x==[254, 183, 5]).all():
        return 3
    if (x==[2, 48, 74]).all():
        return 4
    if (x==[19, 103, 131]).all():
        return 5
    if (x==[239, 65, 67]).all():
        return 6
    if (x==[42, 157, 140]).all():
        return 7
    if (x==[138, 176, 125]).all():
        return 8
    if (x==[251, 132, 2]).all():
        return 9
    if (x==[131, 64, 38]).all():
        return 10
    if (x==[0, 0, 0]).all():
        return 11

    return 0


img_dir_path = 'D:/desktop/arvr/cloth-dataset/s_with_mask/mask'
path_list=[]
listdir(img_dir_path,path_list)
#print(path_list)
single_im = np.zeros((901,256, 256))
for image_p in tqdm(path_list,total=901):
   # print(image_p[len(img_dir_path)+1:-4])
    count=int(image_p[len(img_dir_path)+1:-4])
    image=Image.open(image_p).convert('RGB')
    image=np.array(image)
    image=np.transpose(image, [2, 0, 1])
    for i in range(256):
        for j in range(256):
            single_im[count][i][j]=check_color(image[:,i,j],count,i,j)
    count+=1
np.save(os.path.join(img_dir_path,"mask_all.npy"),single_im)
