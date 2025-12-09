import torch
import numpy as np
avg_style_latents_list=[]
avg_style_latents_list.append(torch.Tensor(np.load(f"./latent_code/620t_sample_200000_all_s/sspace_620t_sample_200000_all_s_feat.npy")))
avg_style_latents = torch.cat(avg_style_latents_list, dim=0)[:1]
print("in.shape:",avg_style_latents.shape)
sspace_feat=avg_style_latents.clone().detach()

s_coarse = sspace_feat[:, :4*512].view(-1,4*512)#对应0~3号通道

s_medium = torch.cat([sspace_feat[:, 5*512:7*512],
                        sspace_feat[:, 8*512:10*512],],dim=1).view(-1,4*512)

s_fine   = torch.cat([sspace_feat[:, 11*512:12*512+256],
                        sspace_feat[:, 12*512+256*2 : 12*512 + 256*3 + 128],
                        sspace_feat[:, 12*512 + 256*3 + 128*2 : 12*512 + 256*3+ 128*3 + 64],],dim=1)#[batchSize][512+256,256+128,128+64]

s_rgb   = torch.cat([sspace_feat[:, 4*512:5*512],
                        sspace_feat[:, 7*512:8*512],
                        sspace_feat[:, 10*512:11*512],
                        sspace_feat[:, 12*512+256:12*512+256*2],
                        sspace_feat[:, 12*512 + 256*3 + 128:12*512 + 256*3 + 128*2],
                        sspace_feat[:, 12*512 + 256*3 + 128*3 + 64:],],dim=1)#[batchSize][512,512,512,256,128,64]

x_mapped = torch.cat([s_rgb[:,:512],
                              s_medium[:,:2*512],s_rgb[:,512:2*512],
                              s_medium[:,2*512:4*512],s_rgb[:,2*512:3*512],
                              s_fine[:,:512+256],s_rgb[:,3*512:3*512+256],
                              s_fine[:,512+256:512+256+256+128],s_rgb[:,3*512+256:3*512+256+128],
                              s_fine[:,512+256+256+128:],s_rgb[:,3*512+256+128:]],dim=1)
out = torch.cat([s_coarse, x_mapped], dim=1)
print("out.shape:",out.shape)
print(out.equal(avg_style_latents))

