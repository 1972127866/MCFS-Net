import seaborn as sns
from PIL import Image, ImageFilter
import numpy as np
from tqdm import tqdm
import os
from matplotlib import pyplot as plt

def calDiverse(orig_color,edit_color):
    #计算两个点的三通道值之差的绝对值之和
    diverse = np.sum(np.abs(orig_color-edit_color))
    return diverse


target_dir="/home/scut/workspace/wanqing/editGANdata/paper_compare_images/collar_orig_images"
gen_dir="/home/scut/workspace/wanqing/editGANdata/paper_compare_images/ACU_collar_result"
save_dir="/home/scut/workspace/wanqing/editGANdata/paper_compare_images/ACU_collar_heatmap"
# gen_dir="/home/scut/workspace/wanqing/editGANdata/paper_compare_images/ours_collar_result"
# save_dir="/home/scut/workspace/wanqing/editGANdata/paper_compare_images/ours_collar_heatmap"
for pic_id in tqdm(range(6)):
    tar_image=Image.open(os.path.join(target_dir,str(pic_id)+".jpg")).convert ('RGB')
    tar_image = tar_image.filter(ImageFilter.BLUR)
    tar_image=np.transpose(np.array(tar_image), [2, 0, 1]).astype(np.float32)
    gen_image=Image.open(os.path.join(gen_dir,str(pic_id)+".jpg")).convert ('RGB')
    gen_image = gen_image.filter(ImageFilter.BLUR)
    gen_image=np.transpose(np.array(gen_image), [2, 0, 1]).astype(np.float32)
    
    print(tar_image.shape)
    diverse = np.zeros((256, 256))
    for i in range(256):
        for j in range(256):
            diverse[i][j] = calDiverse(tar_image[:,i,j], gen_image[:,i,j])
    # heatmap_pos = sns.heatmap(diverse, cmap="YlGnBu_r", xticklabels=False, yticklabels=False,cbar=False)
    heatmap_pos = sns.heatmap(diverse, cmap="jet", xticklabels=False, yticklabels=False,cbar=False,bbox_inches='tight',pad_inches=0.0)
    heatmap_pos.get_figure().savefig(os.path.join(save_dir,str(pic_id)+".png"))
    plt.close()
    