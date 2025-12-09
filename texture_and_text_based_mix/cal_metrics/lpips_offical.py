import torch
import lpips
import os
from tqdm import tqdm
import gc


def cal_lpips(startid,endid):
    use_gpu = True         # Whether to use GPU
    spatial = False         # Return a spatial map of perceptual distance.
    # Linearly calibrated models (LPIPS)
    loss_fn = lpips.LPIPS(net='vgg', spatial=spatial) # Can also set net = 'squeeze' or 'vgg'
    # loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'
    if(use_gpu):
        loss_fn.cuda()

    # target_texture_path="./output/flexible/deep_fashion_multi_model_texture_400"
    # result_texture_path="./output/flexible/all_s_deep_fashion_multi_model_texture_400"
    # target_texture_path="./output/flexible/collar_all_s_deep_fashion_multi_model_texture_400"
    # result_texture_path="./output/flexible/collar_all_s_deep_fashion_multi_model_texture_400"
    # target_texture_path="./output/flexible/collar_all_s_deep_fashion_multi_model_texture_0_4"
    # result_texture_path="./output/flexible/collar_all_s_deep_fashion_multi_model_texture_0_4"
    # target_texture_path="./output/flexible/collar_all_s_deep_fashion_multi_model_texture_256000pth_cloth0_3"
    # result_texture_path="./output/flexible/collar_all_s_deep_fashion_multi_model_texture_256000pth_cloth0_3"
    # target_texture_path="/home/scut/workspace/wanqing/DeltaEdit-main/output/flexible/dtd_470_620t"
    # target_texture_path="/home/scut/workspace/wanqing/DeltaEdit-main/output/ablation/620t_deepfashion_wo_img_loss"
    # target_texture_path="/home/scut/workspace/wanqing/DeltaEdit-main/output/ablation/620t_deepfashion_wo_extra_mapper"
    # target_texture_path="/home/scut/workspace/wanqing/DeltaEdit-main/output/ablation/620t_deepfashion_orig_model"
    # target_texture_path="/home/scut/workspace/wanqing/DeltaEdit-main/output/ablation/collar_deepfashion_wo_img_loss"
    # target_texture_path="/home/scut/workspace/wanqing/DeltaEdit-main/output/ablation/collar_deepfashion_wo_extra_mapper"
    # target_texture_path="/home/scut/workspace/wanqing/DeltaEdit-main/output/ablation/collar_deepfashion_orig_model"
    # target_texture_path="/home/scut/workspace/wanqing/DeltaEdit-main/output/ablation/all_s_deep_fashion_cids_0_2"
    # target_texture_path="/home/scut/workspace/wanqing/DeltaEdit-main/output/flexible/deep_fashion_cids_0_2"
    target_texture_path="/home/scut/workspace/wanqing/DeltaEdit-main/output/ablation/collar_deepfashion_all_s_0_2"
    result_texture_path=target_texture_path

    my_avg_lpips=0

    count=0
    for cloth_idx in range(3):
        single_avg_lpips=0
        for texture_idx in range(startid,endid):
            texture_name=str(texture_idx)+"_texture.jpg"
            gc.collect()
            image_name=str(texture_idx)+"_"+str(cloth_idx)+".jpg"
            ex_ref = lpips.im2tensor(lpips.load_image(os.path.join(target_texture_path,texture_name)))
            ex_my = lpips.im2tensor(lpips.load_image(os.path.join(result_texture_path,image_name))[103:153,103:153])
            # ex_compare = lpips.im2tensor(lpips.load_image(os.path.join(compare_dir_path,image_name)))
            if(use_gpu):
                ex_ref = ex_ref.cuda()
                ex_my = ex_my.cuda()
            ex_my = loss_fn.forward(ex_ref,ex_my)
            my_avg_lpips+=ex_my
            single_avg_lpips+=ex_my
            count+=1
        print("lpips of cloth ",str(cloth_idx)," :",single_avg_lpips/endid)

    my_avg_lpips=my_avg_lpips/count
    # compare_avg_lpips=compare_avg_lpips/count
    # print("count",count)
    # print('avg Distances: (%.3f, %.3f)'%(my_avg_lpips, compare_avg_lpips))
    # print("my max: ",my_max_lpips,"my min: ",my_min_lpips)
    # print("pti max: ",pti_max_lpips,"pti min: ",pti_min_lpips)
    return my_avg_lpips.detach().cpu()

 
    
