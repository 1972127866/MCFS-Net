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

from models.psp import pSp
from argparse import Namespace
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def embed_one_example(args, path, stylegan_encoder, g_all, upsamplers,
                      inter, percept, steps, sv_dir,seed,
                      skip_exist=False,encoded_latent=None):

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

    #latent_in = stylegan_encoder(label_im_tensor)
    latent_in=torch.from_numpy(encoded_latent).unsqueeze(0).cuda()
    
    #不用encoder，随机初始嵌入
    # latent_z = torch.from_numpy(np.random.RandomState(int(seed)).randn(1, 512)).to(device)
    # latent_in= g_all.module.mapping(latent_z.to(device), None)
    # im_out_wo_encoder, _ = latent_to_image(g_all, upsamplers, latent_in,
    #                                        process_out=True, use_style_latents=True,
    #                                        return_only_im=True)
    # PIL.Image.fromarray(im_out_wo_encoder[0], 'RGB').save('random.png')

    out = run_embedding_optimization(args, g_all,
                                     upsamplers, inter, percept,
                                     label_im_tensor, latent_in, steps=steps,
                                     stylegan_encoder=stylegan_encoder,
                                     use_noise=args['use_noise'],
                                     noise_loss_weight=args['noise_loss_weight'],
                                     use_restyle=True
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

    logprint(f'Computing S midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.module.z_dim)
    w_samples = G.module.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    _, _, s_samples = G.module.synthesis(w_samples, noise_mode='const', return_style=True)
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

    # w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    # w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)  #w_out.shape=torch.Size([500, 1, 512]),list(w_opt.shape[1:])=[1, 512]
    # optimizer = torch.optim.Adam(s_opt + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
    s_out = []
    s_opt=[]
    if use_opti:#使用在w+空间优化后的潜码所生成的s作为起始点
        for i in range(len(s_avg)):
            s_opt.append(opti_s[i].detach())
            s_opt[i].requires_grad=True # pylint: disable=not-callable
            # print("s_out[",i,"].shape is",([num_steps] + list(s_opt[i].shape[1:])))
            s_out.append(torch.zeros([num_steps] + list(s_opt[i].shape[1:]), dtype=torch.float32, device=device))
            #print("s_out[", i, "][0].shape", s_out[i][0].shape)

    else:#使用均值潜码所生成的s作为起始点
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

        # Synth images from opt_w.
        # w_noise = torch.randn_like(w_opt) * w_noise_scale
        # ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        # synth_images = G.synthesis(ws, noise_mode='const')
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
#         if step%10==0:
#             logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        # w_out[step] = w_opt.detach()[0]
        for i in range(len(s_opt)):
            # print("s_opt[",i,"].shape",s_opt[i].shape)
            # print("s_out[",i,"][",step,"].shape",s_out[i][step].shape)
            s_out[i][step] = s_opt[i].detach()

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()
    # w_out.shape=torch.Size([500, 1, 512])
    # w_out.repeat([1, G.mapping.num_ws, 1]).shape=torch.Size([500, 14, 512])
    # return w_out.repeat([1, G.mapping.num_ws, 1])
    logprint(f'dist {dist:<4.2f} loss {float(loss):<5.2f}')
    return s_out

def restyle_run_on_batch(inputs, net, encoder_opts, avg_image):
    print("inputs.shape",inputs.shape)
    y_hat, latent = None, None
    results_batch = {idx: [] for idx in range(inputs.shape[0])}
    results_latent = {idx: [] for idx in range(inputs.shape[0])}
    for iter in range(encoder_opts.n_iters_per_batch):
        if iter == 0:
            avg_image_for_batch = avg_image.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
            x_input = torch.cat([inputs, avg_image_for_batch], dim=1)
        else:
            x_input = torch.cat([inputs, y_hat], dim=1)

        y_hat, latent = net.forward(x_input,
                                    latent=latent,
                                    randomize_noise=False,
                                    return_latents=True,
                                    resize=encoder_opts.resize_outputs)

        # store intermediate outputs
        for idx in range(inputs.shape[0]):
            results_batch[idx].append(y_hat[idx])
            results_latent[idx].append(latent[idx].cpu().numpy())

        # resize input to 256 before feeding into next iteration
        y_hat = net.face_pool(y_hat)

    return results_batch, results_latent


def tensor2im(var):
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))

def test_stylegan_proj(args, resume, steps1,  latent_sv_folder='', skip_exist=False,num_steps=250,save_video=False,parser_args=None):

    g_all, _, upsamplers, _, avg_latent = prepare_model(args)
    inter = Interpolate(args['im_size'][1], 'bilinear')
    
    percept = lpips.PerceptualLoss(
        model='net-lin', net='vgg', use_gpu=device.startswith('cuda'), normalize=args['normalize']
    ).to(device)


    #配置restyle的encoder
    encoder_ckpt = torch.load(resume, map_location='cpu')
    encoder_opts = encoder_ckpt['opts']
    encoder_opts.update(vars(parser_args))
    encoder_opts = Namespace(**encoder_opts)
    restyle_encoder = pSp(encoder_opts)

    restyle_encoder.eval()
    restyle_encoder.cuda()
    
    #计算avg_image
    restyle_encoder.latent_avg = restyle_encoder.decoder.make_mean_latent(int(1e5))[0].detach()
    avg_image = restyle_encoder(restyle_encoder.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    avg_image = avg_image.to('cuda').float().detach()

    #加载待嵌入的图片
    assert latent_sv_folder != ""
    all_images = []
    all_id = []
    curr_images_all = glob.glob(args['testing_data_path'] +  "*/*")
    curr_images_all = [data for data in curr_images_all if ('jpg' in data or 'webp' in data or 'png' in data  or 'jpeg' in data or 'JPG' in data) and not os.path.isdir(data)  and not 'npy' in data ]
    for i, image in enumerate(curr_images_all):
        all_id.append(image.split("/")[-1].split(".")[0])
        all_images.append(image)
    print("All files, " , len(all_images))


    all_loss_before_opti, all_loss_after_opti = [], []
    for i, id in enumerate(tqdm(all_id)):
        print("Curr dir,", id)
        sv_folder = os.path.join(latent_sv_folder,'w-step'+str(steps1)+ '_s-step' + str(num_steps))

        label_im_tensor, im_id = load_one_image_for_embedding(all_images[i], args['im_size'])

        print("****** Run optimization for ", path, " ******")

        label_im_tensor = label_im_tensor.to(device)
        label_im_tensor = label_im_tensor * 2.0 - 1.0
        label_im_tensor = label_im_tensor.unsqueeze(0)

        os.makedirs(sv_folder, exist_ok=True)

        #使用encoder获得初试嵌入
        global_time = []
        all_latents = {}
        with torch.no_grad():
            input_cuda = label_im_tensor.float()
            tic = time.time()
            result_batch, result_latents = restyle_run_on_batch(input_cuda, restyle_encoder, encoder_opts, avg_image)
            toc = time.time()
            global_time.append(toc - tic)

        #保存各迭代回合中生成的图片
        # results = [tensor2im(result_batch[0][iter_idx]) for iter_idx in range(encoder_opts.n_iters_per_batch)]
        # for idx, result in enumerate(results):
        #         result.save(os.path.join(sv_folder, str(id)+"_embed_only_restyle_"+str(idx)+".png"))


        #img, styles_feature, all_styles = g_all.module.synthesis(latent_in, return_style=True)

        loss_before_opti, loss_after_opti , all_final_latent, all_final_noise = embed_one_example(args, all_images[i],
                                                                                                  None, g_all,
                                                                                                  upsamplers, inter, percept, steps1,
                                                                                                  sv_folder,id, skip_exist=skip_exist,encoded_latent=result_latents[0][4])
        

        all_final_latent_copy=torch.from_numpy(all_final_latent).cuda().unsqueeze(0)
        #synth_image, styles_feature, all_styles = g_all.module.synthesis(all_final_latent_copy, return_style=True)
        img, styles_feature, all_styles = g_all.module.synthesis(all_final_latent_copy, return_style=True)
        
        # Load target image.
        target_pil = PIL.Image.open(all_images[i]).convert('RGB')
        target_pil = target_pil.resize((g_all.module.img_resolution, g_all.module.img_resolution), PIL.Image.LANCZOS)
        target_uint8 = np.array(target_pil, dtype=np.uint8)

        # Optimize projection.
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
        projected_s = []
        for i in range(20):
            projected_s.append(projected_w_steps[i][-1].unsqueeze(0))
        synth_image, _ = g_all.module.synthesis(None, noise_mode='const', use_styles=True, input_styles=projected_s)
        synth_image = (synth_image + 1) * (255 / 2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        PIL.Image.fromarray(synth_image, 'RGB').save(f'{sv_folder}/%s_embed_restyle.png'%str(id))
        s_cpu=[]
        for i in range(20):
            s_cpu.append(projected_s[i].cpu())
        np.save(f'{sv_folder}/%s_embed_restyle.npy'%str(id),s_cpu)
        # projected_s = []
        # for i in range(20):
        #     projected_s.append(projected_w_steps[i][-1].unsqueeze(0))
        # synth_image, _ = g_all.module.synthesis(None, noise_mode='const', use_styles=True, input_styles=projected_s)
        # synth_image = (synth_image + 1) * (255 / 2)
        # synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        # PIL.Image.fromarray(synth_image, 'RGB').save(f'{sv_folder}/%s_embed_restyle_wo_s.png'%str(id))
        # s_cpu=[]
        # for i in range(20):
        #     s_cpu.append(all_styles[i].cpu())
        # np.save(f'{sv_folder}/%s_embed_restyle_wo_s.npy'%str(id),s_cpu)
        




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)
    parser.add_argument('--resume', type=str, default="../editGANdata/restyle_encoder_psp_fashionTop_110000/best_model.pt")
    parser.add_argument('--test', type=bool, default=False)

    parser.add_argument('--testing_path', type=str, default='')
    parser.add_argument('--latent_sv_folder', type=str, default='')
    parser.add_argument('--skip_exist', type=bool, default=False)
    parser.add_argument('--steps', type=int, default=400)

    parser.add_argument('--use_noise', type=bool, default=False)
    parser.add_argument('--noise_loss_weight', type=float, default=100)
    parser.add_argument('--num_steps', type=int, default=250)

    parser.add_argument('--exp_dir', type=str,
                                help='Path to experiment output directory')
    parser.add_argument('--checkpoint_path', default="../editGANdata/restyle_encoder_psp_fashionTop_110000/best_model.pt", type=str,
                                help='Path to ReStyle model checkpoint')
    parser.add_argument('--data_path', type=str, default='gt_images',
                                help='Path to directory of images to evaluate')
    parser.add_argument('--resize_outputs', action='store_true',
                                help='Whether to resize outputs to 256x256 or keep at original output resolution')
    parser.add_argument('--test_batch_size', default=1, type=int,
                                help='Batch size for testing and inference')
    parser.add_argument('--test_workers', default=1, type=int,
                                help='Number of test/inference dataloader workers')
    parser.add_argument('--n_images', type=int, default=None,
                                help='Number of images to output. If None, run on all data')

    # arguments for iterative inference
    parser.add_argument('--n_iters_per_batch', default=5, type=int,
                                help='Number of forward passes per batch during training.')


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

    test_stylegan_proj(opts, args.resume, args.steps,num_steps=args.num_steps,
            latent_sv_folder=args.latent_sv_folder, skip_exist=args.skip_exist,parser_args=args)