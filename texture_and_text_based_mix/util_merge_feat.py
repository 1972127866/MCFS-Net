import torch
import numpy as np
import os
s_latents_list=[]
wplus_latents_list=[]
for i in range(200):
    s_latents_list.append(torch.Tensor(np.load(f"./latent_code/620t_sample_200000_all_s_tmp/"+str(i)+".0_s.npy")))
    print("s_latents_list[i].shape:",s_latents_list[i].shape)
    wplus_latents_list.append(torch.Tensor(np.load(f"./latent_code/620t_sample_200000_all_s_tmp/"+str(i)+".0_w.npy")))
s=torch.cat(s_latents_list, dim=0)
w=torch.cat(wplus_latents_list, dim=0)
os.makedirs("./latent_code/620t_sample_200000_all_s", exist_ok=True)
np.save(f"./latent_code/620t_sample_200000_all_s/wspace_620t_sample_200000_all_s_feat.npy", w)
np.save(f"./latent_code/620t_sample_200000_all_s/sspace_620t_sample_200000_all_s_feat.npy", s)