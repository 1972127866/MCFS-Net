import os
import sys

import torch
from torch.utils.data import DataLoader




sys.path.append(".")
sys.path.append("..")

from datasets.train_dataset_texture_white import TrainLatentsDataset
from options.train_options_white import TrainOptions
from delta_mapper import DeltaMapper
from delta_mapper_medium import DeltaMapperMedium


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
        print("only medium")
        net = DeltaMapperMedium().to(device)
    else:
        net = DeltaMapper().to(device)
    

    #Initialize optimizer
    optimizer = torch.optim.Adam(list(net.parameters()), lr=opts.learning_rate)

    #Initialize loss
    l2_loss = torch.nn.MSELoss().to(device)
    cosine_loss = torch.nn.CosineSimilarity(dim=-1).to(device)

    #save dir
    os.makedirs(os.path.join(opts.checkpoint_path, opts.images_classname,opts.texture_classname), exist_ok=True)


    # with dnnlib.util.open_url('../editGAN/viton_360t_s_fm.pkl') as f:
    #     g_all = legacy.load_network_pkl(f)['G_ema'].to(device)
    
    # for term in range(10):
        # lr = opts.learning_rate * (0.1 ** (term // 3))
        # print("lr:",lr)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr  

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

        optimizer.zero_grad()
        # print("fake_delta_s.shape:",fake_delta_s.shape)#[64, 4928]
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
            if opts.modi_loss:
                torch.save(net.state_dict(), os.path.join(opts.checkpoint_path, "modi_loss",opts.images_classname,opts.texture_classname,"net_%06d.pth" % batch_idx))
            elif opts.only_medium:
                torch.save(net.state_dict(), os.path.join(opts.checkpoint_path, "only_medium",opts.images_classname,opts.texture_classname,"net_%06d.pth" % batch_idx))
            else:
                torch.save(net.state_dict(), os.path.join(opts.checkpoint_path, opts.images_classname,opts.texture_classname,"net_%06d.pth" % batch_idx))
            # torch.save(net.state_dict(), os.path.join(opts.checkpoint_path, opts.tex_edge_classname, "net_%06d_%02dit.pth" % (batch_idx,term)))

if __name__ == "__main__":
    opts = TrainOptions().parse()
    main(opts)