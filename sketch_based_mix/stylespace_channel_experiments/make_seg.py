
import imageio


device_ids = [0]

from utils.data_utils import *
from utils.model_utils import *
import numpy as np
import argparse
import PIL.Image

from models.EditGAN.EditGAN_tool import Tool

np.random.seed(6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# latent_all = []
# for i in range(15):
#     name = 'latents_image_%0d.npy' % i
#     im_frame = np.load(os.path.join("../../hdd/wanqing/editGAN/collar_cloth_exam/collar_cloth_8385_300t_sample_15/collar_cloth_8385_300t_sample_15latent", name))
#     latent_all.append(im_frame)
# latent_all = np.array(latent_all)

# latent_all = torch.from_numpy(latent_all).cuda()

latent_all = []
with dnnlib.util.open_url('viton_360t_s_fm.pkl') as f:
    g_all = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
    avg_latent = g_all.make_mean_latent(4086)
for i in range(50):
    z = torch.from_numpy(np.random.RandomState(i).randn(1, g_all.z_dim)).to(device)
    w = g_all.mapping(z.to(device), None)
    w1=0.2*avg_latent+0.8*w
    image,_,s = g_all.synthesis(w1, noise_mode='const',return_style=True)
    image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(image, 'RGB').save(os.path.join("../editGANresult/model_interpreter/cloth_360t_6-class_v2/test_result","sample_"+str(i)+'.jpg'))
    latent_all.append(s)


tool = Tool()

for i in range(len(latent_all)):
    gc.collect()  # 清理内存
    # latent_input = latent_all[i].float()
    # img_out, img_seg_final=tool.run_seg(latent_input.unsqueeze(0))#img_seg_final是一个二维数组，背景区域为0，衣服区域为1，领子区域为2
    latent_input = latent_all[i]
    img_out, img_seg_final=tool.run_seg(None,use_style=True,input_styles=latent_input)
    os.makedirs("../editGANresult/model_interpreter/cloth_360t_6-class_v2/test_result",exist_ok=True)
    # imageio.imsave(os.path.join("../../hdd/wanqing/editGAN/collar_cloth_exam/model_interpreter/traindata_seg", str(i) + '.jpg'),
    #                img_out.astype(np.uint8))
    seg_vis = colorize_mask(img_seg_final, cloth_6_palette)
    imageio.imsave(os.path.join("../editGANresult/model_interpreter/cloth_360t_6-class_v2/test_result", "sample_"+str(i) + '.png'),
                   seg_vis)


