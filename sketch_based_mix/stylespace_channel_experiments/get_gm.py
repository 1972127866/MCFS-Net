import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import time
import os
import copy
import torch
import dnnlib
import legacy
from torch.autograd.gradcheck import _iter_tensors,_compute_analytical_jacobian_rows,_stack_and_check_tensors
import time
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms



device = torch.device('cuda')
with dnnlib.util.open_url("model/ffhq.pkl") as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

z = torch.from_numpy(np.random.randn(1, G.z_dim)).to('cuda')
ws = G.mapping(z,0,truncation_psi=0.7)
img = G.synthesis(ws, noise_mode='random')
# img = G(z, 2, truncation_psi=0.7, noise_mode='random')
img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
ori_im = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').resize((200,200))

total_grad = []

def backward_hook(module, grad_in, grad_out):
    total_grad.append(grad_out[0][0])


back_handle = []
for block_name in G.synthesis._modules:
    for layer_name in G.synthesis._modules[block_name]._modules:
        cur_handle = G.synthesis._modules[block_name]._modules[layer_name].affine.register_backward_hook(backward_hook)
        back_handle.append(cur_handle)
    #         break
#     break
print(len(back_handle))


for h in back_handle:
    h.remove()


def get_analytical_jacobian(inputs, output):  #res = get_analytical_jacobian(ws,output_resize[:,0,:,:])
    diff_input_list = list(_iter_tensors(inputs, True))

    def vjp_fn(grad_output):
        return torch.autograd.grad(output, diff_input_list, grad_output,
                                   retain_graph=True, allow_unused=True)
    # Compute everything twice to check for nondeterminism (which we call reentrancy)
    vjps1 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())

    output_numel = output.numel()#用于计算数组中满足指定条件的元素个数。若是一幅图像，则numel(A)将给出它的像素数
    jacobians1, _, _ = _stack_and_check_tensors(vjps1, inputs, output_numel)
    return jacobians1

start_time = time.time()

ws.requires_grad_(True)
total_grad = []

avgpool = nn.AvgPool2d((32, 32), stride=(32, 32))
output = G.synthesis(ws, noise_mode='const')
output_resize = avgpool(output)

res = get_analytical_jacobian(ws,output_resize[:,0,:,:])

print("total_cost:",time.time() - start_time)
print("total_output_size:", len(total_grad) / len(back_handle),len(back_handle))
# grad = res[0][0][1]
# grad = grad.detach().cpu().numpy()
# Image.fromarray(np.uint8(np.where(grad.reshape(32,32) > 0,0,255))).resize((200,200))

# grad = np.array(list(reversed(total_grad))) # 26 layers * 1024 position # channel
# 梯度转换
grad = np.array((total_grad)) # 26 layers * 1024 position # channel

% matplotlib
inline
import matplotlib.pyplot as plt
from PIL import Image

plt.figure(figsize=(20, 10))
plt.axis("off")
index = 3
# index = 25 - index
for k in range(20):
    #     grad = res[0][0][i+512 * 3]
    #     grad = grad.detach().cpu().numpy()
    #     img = Image.fromarray(np.uint8(np.where(grad.reshape(32,32) > 0,0,255))).resize((200,200))
    #     gradient_map = np.array([grad[26*i + index][k] for i in range(1024)])
    #gradient_map = np.array([grad[26 * i + 25 - index][k].detach().cpu().numpy() for i in range(1024)])
    gradient_map = np.array([grad[20 * i + 19 - index][k].detach().cpu().numpy() for i in range(1024)])
    gradient_heat_map = np.uint8(
        (gradient_map - np.min(gradient_map)) / (np.max(gradient_map) - np.min(gradient_map)) * 255).reshape(32, 32)
    gradient_binary_map = np.uint8(np.where(gradient_heat_map > 150, gradient_heat_map, 0))

    #     gradient_im = Image.fromarray(gradient_heat_map).resize((200,200))
    gradient_im = Image.fromarray(gradient_binary_map).resize((200, 200))

    gradient_blend = Image.blend(gradient_im.convert("RGB"), ori_im, 0.4)

    #     index = i + 512 * 0
    #     grad = np.array([i[index].detach().cpu().numpy() for i in total_grad])
    #     img = Image.fromarray(np.uint8(np.where(grad.reshape(32,32) > -0.002 ,0,255))).resize((200,200))
    #     res = Image.blend(img.convert("RGB"),ori_im,0.5)

    plt.subplot(4, 5, k + 1)
    plt.axis("off")
    plt.imshow(gradient_blend)
