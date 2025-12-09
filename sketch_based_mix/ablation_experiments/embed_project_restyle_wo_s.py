import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import numpy as np
import torch
torch.manual_seed(0)
import json
import torch.nn.functional as F
import os

device_ids = [0]
from PIL import Image
from tqdm import tqdm

from utils.data_utils import *
from utils.model_utils import *

import argparse
import glob
import lpips
import PIL.Image

from models.psp import pSp
from argparse import Namespace
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def embed_one_example(args, path, stylegan_encoder, g_all, upsamplers,
                      inter, percept, steps, sv_dir,seed,
                      skip_exist=False,encoded_latent=None,restyle_opts=None,avg_img=None):

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
    

    # latent_z = torch.from_numpy(np.random.RandomState(int(seed)).randn(1, 512)).to(device)
    # latent_in= g_all.module.mapping(latent_z.to(device), None)
    
    print("latent_in.shape",latent_in.shape)
    im_out_wo_encoder, _ = latent_to_image(g_all, upsamplers, latent_in,
                                           process_out=True, use_style_latents=True,
                                           return_only_im=True)

    #PIL.Image.fromarray(im_out_wo_encoder[0], 'RGB').save(os.path.join('../../workspace/wanqing/editGANresult/localedit-color_embed',seed+'_embed_only_encoder.png'))
        

    out = run_embedding_optimization(args, g_all,
                                     upsamplers, inter, percept,
                                     label_im_tensor, latent_in, steps=steps,
                                     stylegan_encoder=stylegan_encoder,
                                     use_noise=args['use_noise'],
                                     noise_loss_weight=args['noise_loss_weight'],
                                     use_restyle=True,
                                     restyle_opts=restyle_opts,avg_img=avg_img
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

def restyle_run_on_batch(inputs, net, encoder_opts, avg_image):
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
    
    # get the image corresponding to the latent average
    # avg_image = get_average_image(net, encoder_opts)
    restyle_encoder.latent_avg = restyle_encoder.decoder.make_mean_latent(int(1e5))[0].detach()
    avg_image = restyle_encoder(restyle_encoder.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    avg_image = avg_image.to('cuda').float().detach()


    assert latent_sv_folder != ""
    all_images = []
    all_id = []
    curr_images_all = glob.glob(args['testing_data_path'] +  "*/*")
    curr_images_all = [data for data in curr_images_all if ('jpg' in data or 'webp' in data or 'png' in data  or 'jpeg' in data or 'JPG' in data) and not os.path.isdir(data)  and not 'npy' in data ]
    for i, image in enumerate(curr_images_all):
        all_id.append(image.split("/")[-1].split(".")[0])
        all_images.append(image)
    print("All files, " , len(all_images))

    for i, id in enumerate(tqdm(all_id)):
        print("Curr dir,", id)
        sv_folder = os.path.join(latent_sv_folder,'w-step'+str(steps1)+ '_s-step' + str(num_steps))

        label_im_tensor, im_id = load_one_image_for_embedding(all_images[i], args['im_size'])

        label_im_tensor = label_im_tensor.to(device)
        label_im_tensor = label_im_tensor * 2.0 - 1.0
        label_im_tensor = label_im_tensor.unsqueeze(0)
        os.makedirs(sv_folder, exist_ok=True)
        # latent_in = stylegan_encoder(label_im_tensor)

        global_time = []
        with torch.no_grad():
            input_cuda = label_im_tensor.float()
            tic = time.time()
            result_batch, result_latents = restyle_run_on_batch(input_cuda, restyle_encoder, encoder_opts, avg_image)
            toc = time.time()
            global_time.append(toc - tic)
        # results = [tensor2im(result_batch[0][iter_idx]) for iter_idx in range(encoder_opts.n_iters_per_batch)]
        # for idx, result in enumerate(results):
        #         result.save(os.path.join(sv_folder, str(id)+"_embed_only_restyle_"+str(idx)+".png"))

        loss_before_opti, loss_after_opti , all_final_latent, all_final_noise = embed_one_example(args, all_images[i],
                                                                                                  restyle_encoder, g_all,
                                                                                                  upsamplers, inter, percept, steps1,
                                                                                                  sv_folder,id, skip_exist=skip_exist,
                                                                                                  encoded_latent=result_latents[0][4],
                                                                                                  restyle_opts=encoder_opts,avg_img=avg_image)

        all_final_latent_copy=torch.from_numpy(all_final_latent).cuda().unsqueeze(0)
        synth_image, styles_feature, all_styles = g_all.module.synthesis(all_final_latent_copy, return_style=True)
        
        w_cpu=all_final_latent_copy.squeeze(0).cpu().numpy()
        np.save(f'{sv_folder}/%s_embed_restyle_wo_s_latent_w.npy'%str(id),w_cpu)

        synth_image = (synth_image + 1) * (255 / 2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        PIL.Image.fromarray(synth_image, 'RGB').save(f'{sv_folder}/%s_embed_restyle_wo_s.png'%str(id))
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
    parser.add_argument('--checkpoint_path', default="../../workspace/wanqing/editGANdata/restyle_encoder_psp_fashionTop_110000/best_model.pt", type=str,
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