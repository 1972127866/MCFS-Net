import os
import sys

import torch
from torch.utils.data import DataLoader


import clip



sys.path.append(".")
sys.path.append("..")

from datasets.train_dataset_texture_white import TrainLatentsDataset
from options.train_options_white import TrainOptions
from delta_mapper import DeltaMapper
from utils import stylespace_util
from models.stylegan2.model import Generator

def gen_c_latents_from_edited_with_mask(netG,style_space, w_latents,upsample,avg_pool,model):
    # latent_code=np.load(latent_path, allow_pickle=True)
    # w_latents = torch.from_numpy(latent_code).to(device).unsqueeze(0)

    # #使用w生成s
    # style_space, noise = stylespace_util.encoder_latent(netG, w_latents)
    # s_latents = torch.cat(style_space, dim=1)
    # _, noise = stylespace_util.encoder_latent(netG, w_latents)
    tmp_imgs = stylespace_util.decoder_validate(netG, style_space, w_latents)

    #只取中心50*50区域
    mask=torch.zeros_like(tmp_imgs)
    mask[:,:,103:153,103:153]=1
    tmp_imgs=mask*tmp_imgs

    
    img_gen_for_clip = upsample(tmp_imgs)
    img_gen_for_clip = avg_pool(img_gen_for_clip)

    c_latents = model.encode_image(img_gen_for_clip)

    return c_latents


def main(opts):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = TrainLatentsDataset(opts)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=opts.batch_size,
                                  shuffle=True,
                                  num_workers=int(opts.workers),
                                  drop_last=True)

    #Initialze DeltaMapper
    net = DeltaMapper().to(device)

    #Initialize optimizer
    optimizer = torch.optim.Adam(list(net.parameters()), lr=opts.learning_rate)

    #Initialize loss
    l2_loss = torch.nn.MSELoss().to(device)
    cosine_loss = torch.nn.CosineSimilarity(dim=-1).to(device)

    #save dir
    os.makedirs(os.path.join(opts.checkpoint_path, opts.images_classname,opts.texture_classname), exist_ok=True)

    netG = Generator(256, 512, 2).to(device)
    netG.eval()
    checkpoint = torch.load(opts.stylegan_weights, map_location='cpu')
    netG.load_state_dict(checkpoint['g_ema'])

    model, preprocess = clip.load("ViT-B/32", device=device)
    # avg_pool = torch.nn.AvgPool2d(kernel_size=1024 // 32)
    avg_pool = torch.nn.AvgPool2d(kernel_size=256 // 32)
    upsample = torch.nn.Upsample(scale_factor=7)
    # with dnnlib.util.open_url('../editGAN/viton_360t_s_fm.pkl') as f:
    #     g_all = legacy.load_network_pkl(f)['G_ema'].to(device)
    
    # for term in range(10):
        # lr = opts.learning_rate * (0.1 ** (term // 3))
        # print("lr:",lr)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
            
    for batch_idx, batch in enumerate(train_dataloader):

        latent_s, delta_c, delta_s,latent_c2,latent_w= batch

        latent_s = latent_s.to(device)
        delta_c = delta_c.to(device)
        delta_s = delta_s.to(device)
        latent_c2 = latent_c2.to(device)
        latent_w=latent_w.to(device)

        fake_delta_s = net(latent_s, delta_c)

        fake_c=gen_c_latents_from_edited_with_mask(netG,fake_delta_s+latent_s,latent_w,upsample,avg_pool,model)

        optimizer.zero_grad()
        loss_l2_s = l2_loss(fake_delta_s, delta_s)
        loss_cos_s = 1 - torch.mean(cosine_loss(fake_delta_s, delta_s))
        loss_cos_c = 1 - torch.mean(cosine_loss(fake_c, latent_c2))

        

        loss = opts.l2_lambda * loss_l2_s+ opts.cos_lambda * loss_cos_s+opts.clip_lambda*loss_cos_c

        loss.backward()
        optimizer.step()

        if batch_idx % opts.print_interval == 0 :
            print(batch_idx, loss.detach().cpu().numpy(), loss_l2_s.detach().cpu().numpy(), loss_cos_s.detach().cpu().numpy(),loss_cos_c.detach().cpu().numpy())

        if batch_idx % opts.save_interval == 0:
            torch.save(net.state_dict(), os.path.join(opts.checkpoint_path, opts.images_classname,opts.texture_classname,"net_%06d.pth" % batch_idx))
            # torch.save(net.state_dict(), os.path.join(opts.checkpoint_path, opts.tex_edge_classname, "net_%06d_%02dit.pth" % (batch_idx,term)))

if __name__ == "__main__":
    opts = TrainOptions().parse()
    main(opts)