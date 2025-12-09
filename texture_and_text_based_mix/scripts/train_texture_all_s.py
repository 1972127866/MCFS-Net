import os
import sys
import PIL.Image

import torch
from torch.utils.data import DataLoader




sys.path.append(".")
sys.path.append("..")

from datasets.train_dataset_texture_all_s import TrainLatentsDataset
from options.train_options_white import TrainOptions
from delta_mapper_all_s import DeltaMapper

from delta_mapper_medium import DeltaMapperMedium
from models.stylegan2.model import Generator
from utils import stylespace_util_all_s


def main(opts):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = TrainLatentsDataset(opts)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=opts.batch_size,
                                  shuffle=True,
                                  num_workers=int(opts.workers),
                                  drop_last=True)

    #Initialze DeltaMapper
    if opts.only_medium:
        print("only medium,这里是错的，还没有写针对完整s的DeltaMapperMedium")
        assert 0==1
        net = DeltaMapperMedium()
    else:
        net = DeltaMapper()
    
    if opts.resume is not None:
        print("loading resume from ",opts.resume)
        net_ckpt = torch.load(opts.resume)
        net.load_state_dict(net_ckpt)

    net = net.to(device)
    
    #Initialize generator
    print('Loading stylegan weights from pretrained!')
    g_ema = Generator(size=256, style_dim=512, n_mlp=2)
    g_ema_ckpt = torch.load(opts.stylegan_weights)
    g_ema.load_state_dict(g_ema_ckpt['g_ema'], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)
    
    #Initialize optimizer
    optimizer = torch.optim.Adam(list(net.parameters()), lr=opts.learning_rate)

    #Initialize loss
    l2_loss = torch.nn.MSELoss().to(device)
    cosine_loss = torch.nn.CosineSimilarity(dim=-1).to(device)
    
    #save dir
    if opts.resume is not None:
        os.makedirs(os.path.join(opts.checkpoint_path, "all_s","resume",opts.images_classname,opts.texture_classname), exist_ok=True)
    else:
        os.makedirs(os.path.join(opts.checkpoint_path, "all_s",opts.images_classname,opts.texture_classname), exist_ok=True)
#     os.makedirs(os.path.join(opts.checkpoint_path,"img_val"), exist_ok=True)


    if opts.modi_loss or opts.only_medium:
        fine_mask=torch.zeros([opts.batch_size,4928]).to(device)
        medium_mask=torch.zeros_like(fine_mask).to(device)  
        fine_mask[:,7*512:]=1
        medium_mask[:, 3*512:7*512]=1

    for batch_idx, batch in enumerate(train_dataloader):

        latent_s, delta_c, delta_s= batch

        latent_s = latent_s.to(device)
        delta_c = delta_c.to(device)
        delta_s = delta_s.to(device)

        fake_delta_s = net(latent_s, delta_c)

        # img_avg = stylespace_util_all_s.decoder_validate(g_ema, latent_s, latent_s)
        img_ori = stylespace_util_all_s.decoder_validate(g_ema, latent_s + delta_s, latent_s)
        img_fake=stylespace_util_all_s.decoder_validate(g_ema, latent_s + fake_delta_s, latent_s)
#         torchvision.utils.save_image(img_fake,os.path.join("./checkpoints", "img_val","fake.jpg"))
#         torchvision.utils.save_image(img_ori,os.path.join("./checkpoints", "img_val","real.jpg"))
        # torchvision.utils.save_image(img_avg,os.path.join(opts.checkpoint_path, "img_val","avg_"+str(batch_idx)+"_avg.jpg"))
        # torchvision.utils.save_image(img_ori,os.path.join(opts.checkpoint_path, "img_val","avg_"+str(batch_idx)+"_ori.jpg"))
        # torchvision.utils.save_image(img_fake,os.path.join(opts.checkpoint_path, "img_val","avg_"+str(batch_idx)+"_fake.jpg"))
        
        optimizer.zero_grad()
        # print("fake_delta_s.shape:",fake_delta_s.shape)
        if opts.modi_loss:
            loss_l2 = l2_loss(fake_delta_s*fine_mask, delta_s*fine_mask)+l2_loss(fake_delta_s*medium_mask, delta_s*medium_mask)
            loss_cos = 1 - torch.mean(cosine_loss(fake_delta_s*fine_mask, delta_s*fine_mask))+1 - torch.mean(cosine_loss(fake_delta_s*medium_mask, delta_s*medium_mask))
        elif opts.only_medium:
            loss_l2 = l2_loss(fake_delta_s*medium_mask, delta_s*medium_mask)
            loss_cos = 1 - torch.mean(cosine_loss(fake_delta_s*medium_mask, delta_s*medium_mask))
        else:
            loss_l2 = l2_loss(fake_delta_s, delta_s)
            loss_cos = 1 - torch.mean(cosine_loss(fake_delta_s, delta_s))


        loss = opts.l2_lambda * loss_l2 + opts.cos_lambda * loss_cos
        loss.backward()
        optimizer.step()

        if batch_idx % opts.print_interval == 0 :
            print(batch_idx, loss.detach().cpu().numpy(), loss_l2.detach().cpu().numpy(), loss_cos.detach().cpu().numpy())

        if batch_idx % opts.save_interval == 0:
            if opts.modi_loss or opts.only_medium:
               print("wrong arg")
               assert 0==1
            else:
                if opts.resume is not None:
                    torch.save(net.state_dict(), os.path.join(opts.checkpoint_path, "all_s","resume",opts.images_classname,opts.texture_classname,"net_%06d.pth" % batch_idx))
                else:
                    torch.save(net.state_dict(), os.path.join(opts.checkpoint_path, "all_s",opts.images_classname,opts.texture_classname,"net_%06d.pth" % batch_idx))
            # torch.save(net.state_dict(), os.path.join(opts.checkpoint_path, opts.tex_edge_classname, "net_%06d_%02dit.pth" % (batch_idx,term)))

if __name__ == "__main__":
    opts = TrainOptions().parse()
    main(opts)