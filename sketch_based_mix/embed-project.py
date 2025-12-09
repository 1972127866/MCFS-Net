import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import numpy as np
import torch
import random
import torch.nn as nn
import pickle
torch.manual_seed(0)
import scipy.misc
import json
import torch.nn.functional as F
import os
import imageio

device_ids = [0]
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from models.encoder.encoder import FPNEncoder

from utils.data_utils import *
from utils.model_utils import *

import torch.optim as optim
import argparse
import glob
import lpips
from time import perf_counter
import PIL.Image
import dnnlib

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 下面这个函数首先通过encoder将图片映射到W+空间得到w+，然后迭代优化w+。输出里的optimized_latent_np就是优化好的w+
def embed_one_example(args, path, stylegan_encoder, g_all, upsamplers,
                      inter, percept, steps, sv_dir,
                      skip_exist=False):

    if os.path.exists(sv_dir):
        if skip_exist:
            return 0,0,[], []
        else:
            pass
    else:
        os.system('mkdir -p %s' % (sv_dir))
    print('SV folder at: %s' % (sv_dir))
    image_path = path
    label_im_tensor, im_id = load_one_image_for_embedding(image_path, args['im_size'])

    print("****** Run optimization for ", path, " ******")


    label_im_tensor = label_im_tensor.to(device)
    label_im_tensor = label_im_tensor * 2.0 - 1.0
    label_im_tensor = label_im_tensor.unsqueeze(0)
    latent_in = stylegan_encoder(label_im_tensor)
    print("latent_in.shape",latent_in.shape)
    im_out_wo_encoder, _ = latent_to_image(g_all, upsamplers, latent_in,
                                           process_out=True, use_style_latents=True,
                                           return_only_im=True)

    out = run_embedding_optimization(args, g_all,
                                     upsamplers, inter, percept,
                                     label_im_tensor, latent_in, steps=steps,
                                     stylegan_encoder=stylegan_encoder,
                                     use_noise=args['use_noise'],
                                     noise_loss_weight=args['noise_loss_weight']
                                     )

    if args['use_noise']:
        optimized_latent, optimized_noise, loss_cache = out
        optimized_noise = [torch.from_numpy(noise).cuda() for noise in optimized_noise]
    else:
        optimized_latent, optimized_noise, loss_cache = out
        optimized_noise = None
    print("Curr loss, ", loss_cache[0], loss_cache[-1] )

    optimized_latent_np = optimized_latent.detach().cpu().numpy()[0]
    if args['use_noise']:
        loss_cache_np = [noise.detach().cpu().numpy() for noise in optimized_noise]
    else:
        loss_cache_np = []
    return loss_cache[0], loss_cache[-1], optimized_latent_np, loss_cache_np

# 下面这个函数的作用就是在S空间中优化s，参数里的target是目标图像，opti_s是待优化的s，函数的输出是优化好的s
def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 1000,
    w_avg_samples              = 100,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    use_opti                   =False,
    opti_s                     =None,
    device: torch.device
):
    assert target.shape == (G.module.img_channels, G.module.img_resolution, G.module.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)  # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.module.z_dim)
    w_samples = G.module.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    img, styles_feature, s_samples = G.module.synthesis(w_samples, noise_mode='const', return_style=True)
    s_sample = []
    s_avg = []
    s_std = []
    for i in range(len(s_samples)):
        s_sample.append(s_samples[i].cpu().numpy().astype(np.float32))
        s_avg.append(np.mean(s_sample[i], axis=0, keepdims=True))  # [1, 1, C]
        s_std.append((np.sum((s_sample[i] - s_avg[i]) ** 2) / w_avg_samples) ** 0.5)

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.module.synthesis.named_buffers() if 'noise_const' in name}

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    s_out = []
    s_opt=[]
    if use_opti:
        for i in range(len(s_avg)):
            s_opt.append(opti_s[i].detach())
            s_opt[i].requires_grad=True # pylint: disable=not-callable
            s_out.append(torch.zeros([num_steps] + list(s_opt[i].shape[1:]), dtype=torch.float32, device=device))

    else:
        s_opt = []
        for i in range(len(s_avg)):
            s_opt.append(torch.tensor(s_avg[i], dtype=torch.float32, device=device,
                                      requires_grad=True))  # pylint: disable=not-callable
            s_out.append(torch.zeros([num_steps] + list(s_opt[i].shape[1:]), dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(s_opt + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        # w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        s_noise_scale = []
        for i in range(len(s_std)):
            s_noise_scale.append(s_std[i] * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2)

        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        s_noise = []
        s = []
        for i in range(len(s_std)):
            s_noise.append(torch.randn_like(s_opt[i]) * s_noise_scale[i])
            s.append((s_opt[i] + s_noise[i]))
            # print("s[",i,"].shape",s[i].shape)
        synth_images, _ = G.module.synthesis(None, noise_mode='const', return_style=False, use_styles=True, input_styles=s)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255 / 2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        for i in range(len(s_opt)):
            s_out[i][step] = s_opt[i].detach()

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    logprint(f'dist {dist:<4.2f} loss {float(loss):<5.2f}')
    return s_out

# 下面这段代码就是图像逆映射模块
def test_stylegan_proj(args, resume, steps1,  latent_sv_folder='', skip_exist=False,num_steps=250,save_video=False):

    # 加载用到的模型
    g_all, _, upsamplers, _, avg_latent = prepare_model(args)
    inter = Interpolate(args['im_size'][1], 'bilinear')
    percept = lpips.PerceptualLoss(
        model='net-lin', net='vgg', use_gpu=device.startswith('cuda'), normalize=args['normalize']
    ).to(device)
    stylegan_encoder = FPNEncoder(3, n_latent=args['n_latent'], only_last_layer=args['use_w'])
    stylegan_encoder = stylegan_encoder.to(device)
    stylegan_encoder.load_state_dict(torch.load(resume, map_location=device)['model_state_dict'], strict=False)

    #加载要被逆映射的图片的路径 
    assert latent_sv_folder != ""
    all_images = []
    all_id = []
    curr_images_all = glob.glob(args['testing_data_path'] +  "*/*")
    curr_images_all = [data for data in curr_images_all if ('jpg' in data or 'webp' in data or 'png' in data  or 'jpeg' in data or 'JPG' in data) and not os.path.isdir(data)  and not 'npy' in data ]

    for i, image in enumerate(curr_images_all):
        all_id.append(image.split("/")[-1].split(".")[0])
        all_images.append(image)

    print("All files, " , len(all_images))

    # 开始逐图片进行逆映射
    all_loss_before_opti, all_loss_after_opti = [], []
    for i, id in enumerate(tqdm(all_id)):
#         if i>1:
#              break
        print("Curr dir,", id)
        sv_folder = os.path.join(latent_sv_folder,'w-step'+str(steps1)+ '_s-step' + str(num_steps))

        label_im_tensor, im_id = load_one_image_for_embedding(all_images[i], args['im_size'])

        print("****** Run optimization for ", path, " ******")

        label_im_tensor = label_im_tensor.to(device)
        label_im_tensor = label_im_tensor * 2.0 - 1.0
        label_im_tensor = label_im_tensor.unsqueeze(0)
        latent_in = stylegan_encoder(label_im_tensor)
        #img, styles_feature, all_styles = g_all.module.synthesis(latent_in, return_style=True)

        # embed_one_example里包括图片输入encoder得到w+，对w+优化
        loss_before_opti, loss_after_opti , all_final_latent, all_final_noise = embed_one_example(args, all_images[i],
                                                                                                  stylegan_encoder, g_all,
                                                                                                  upsamplers, inter, percept, steps1,
                                                                                                  sv_folder, skip_exist=skip_exist)

        all_final_latent_copy=torch.from_numpy(all_final_latent).cuda().unsqueeze(0)
        # 将W+空间的w+映射到S空间，得到s
        img, styles_feature, all_styles = g_all.module.synthesis(all_final_latent_copy, return_style=True)
        # Load target image.
        target_pil = PIL.Image.open(all_images[i]).convert('RGB')
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((g_all.module.img_resolution, g_all.module.img_resolution), PIL.Image.LANCZOS)
        target_uint8 = np.array(target_pil, dtype=np.uint8)

        # Optimize projection. 优化s
        start_time = perf_counter()
        projected_w_steps = project(
            g_all,
            target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device),  # pylint: disable=not-callable
            num_steps=num_steps,
            device=device,
            verbose=True,
            use_opti=True,
            opti_s=all_styles
        )
        print(f'Elapsed: {(perf_counter() - start_time):.1f} s')

        # Render debug output: optional video and projected image and W vector.
        os.makedirs(sv_folder, exist_ok=True)
        if save_video:
            video = imageio.get_writer(f'{sv_folder}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
            print(f'Saving optimization progress video "{sv_folder}/proj.mp4"')
            # for projected_w in projected_w_steps:
            #     synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
            #     synth_image = (synth_image + 1) * (255/2)
            #     synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            #     video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
            for j in range(num_steps):
                style = []
                for i in range(20):
                    style.append(projected_w_steps[i][j].unsqueeze(0))
                    print("in step ", j, ",style[", i, "].shape", projected_w_steps[i][j].shape)
                synth_image, _ = g_all.module.synthesis(None, noise_mode='const', use_styles=True, input_styles=style)
                synth_image = (synth_image + 1) * (255 / 2)
                synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                video.append_data(np.concatenate([target_uint8, synth_image], axis=1))

            video.close()

        projected_s = []
        for i in range(20):
            projected_s.append(projected_w_steps[i][-1].unsqueeze(0))
        synth_image, _ = g_all.module.synthesis(None, noise_mode='const', use_styles=True, input_styles=projected_s)
        synth_image = (synth_image + 1) * (255 / 2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        PIL.Image.fromarray(synth_image, 'RGB').save(f'{sv_folder}/%s_proj.png'%str(id))
        s_cpu=[]
        for i in range(20):
            s_cpu.append(projected_s[i].cpu())
        np.save(f'{sv_folder}/%s.npy'%str(id),s_cpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str, help='I forgot what this parameter does, it may not be useful')
    parser.add_argument('--resume', type=str, default="", help='path to checkpoint')

    parser.add_argument('--testing_path', type=str, default='',help='path to the folder where the image to be embedded is located')
    parser.add_argument('--latent_sv_folder', type=str, default='',help='path to the result folder')
    parser.add_argument('--skip_exist', type=bool, default=False,help='Whether to end optimization early based on the gradient situation')
    parser.add_argument('--steps', type=int, default=400, help='Number of optimization iterations in W+ space')

    parser.add_argument('--use_noise', type=bool, default=False)
    parser.add_argument('--noise_loss_weight', type=float, default=100)
    parser.add_argument('--num_steps', type=int, default=250, help='Number of optimization iterations in S space')

    args = parser.parse_args()
    opts = json.load(open(args.exp, 'r'))


    path =opts['exp_dir']
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))

    os.system('cp %s %s' % (args.exp, opts['exp_dir']))
    opts['use_noise'] = args.use_noise
    opts['noise_loss_weight'] = args.noise_loss_weight

    print("Opt", opts)

    if args.testing_path != "":
        opts['testing_data_path'] = args.testing_path

    #将testing_path文件夹中的图片全部映射到S空间，结果保存在latent_sv_folder
    test_stylegan_proj(opts, args.resume, args.steps,num_steps=args.num_steps,
            latent_sv_folder=args.latent_sv_folder, skip_exist=args.skip_exist)