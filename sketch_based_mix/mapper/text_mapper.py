import torch
from torch import nn
from torch.nn import Module
from torch.nn import Linear, LayerNorm, LeakyReLU, Sequential
import clip
import torchvision.transforms as transforms

class TextMapper(Module): 
    def __init__(self, opts):
        super(TextMapper, self).__init__()
        self.opts = opts
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.transform = transforms.Compose([transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])#将图片的值映射到-1到1
        self.face_pool = torch.nn.AdaptiveAvgPool2d((224, 224))
        self.hairstyle_cut_flag = 0
        self.color_cut_flag = 0

        if not opts.no_coarse_mapper: 
            self.course_mapping = SubHairMapper(opts, 4)
        if not opts.no_medium_mapper:
            self.medium_mapping = SubHairMapper(opts, 4)
        if not opts.no_fine_mapper:
            #self.fine_mapping = SubHairMapper(opts, 10)
            self.fine_mapping = SubHairMapper(opts, 6)

    def gen_image_embedding(self, img_tensor, clip_model, preprocess):
        #这里需要优化成获得服装的掩码吗？
        masked_generated = self.face_pool(img_tensor)#只是池化，不是生成掩码
        masked_generated_renormed = self.transform(masked_generated * 0.5 + 0.5)#将(masked_generated * 0.5 + 0.5)的值映射到-1到1上
        return clip_model.encode_image(masked_generated_renormed)

    def forward(self, x, hairstyle_text_inputs, color_text_inputs, hairstyle_tensor, color_tensor):
        if hairstyle_text_inputs.shape[1] != 1: # 有款式文本
            hairstyle_embedding = self.clip_model.encode_text(hairstyle_text_inputs).unsqueeze(1).repeat(1, 18, 1).detach()
        # elif hairstyle_tensor.shape[1] != 1: # 有款式参考图片
        #     hairstyle_embedding = self.gen_image_embedding(hairstyle_tensor, self.clip_model, self.preprocess).unsqueeze(1).repeat(1, 18, 1).detach()
        else: # 两样都没有
            hairstyle_embedding = torch.ones(x.shape[0], 18, 512).cuda()

        # if color_text_inputs.shape[1] != 1:#有颜色文本
        #     color_embedding = self.clip_model.encode_text(color_text_inputs).unsqueeze(1).repeat(1, 18, 1).detach()
        # elif color_tensor.shape[1] != 1:#有颜色参考图
        #     color_embedding = self.gen_image_embedding(color_tensor, self.clip_model, self.preprocess).unsqueeze(1).repeat(1, 18, 1).detach()
        # else:# 两样都没有
        #     color_embedding = torch.ones(x.shape[0], 18, 512).cuda()


        if (hairstyle_text_inputs.shape[1] == 1) and (hairstyle_tensor.shape[1] == 1):#不知道这句是不是在判断有没有文本输入
            self.hairstyle_cut_flag = 1
            # cut_flag=1时，只经过全连接层，不注入embedding
        else:
            self.hairstyle_cut_flag = 0
        # if(color_text_inputs.shape[1] == 1) and (color_tensor.shape[1] == 1):
        #     self.color_cut_flag = 1
        # else:
        #     self.color_cut_flag = 0

        x_coarse = x[:, :4, :]
        x_medium = x[:, 4:8, :]
        x_fine = x[:, 8:, :]

        if not self.opts.no_coarse_mapper:
            x_coarse = self.course_mapping(x_coarse, hairstyle_embedding[:, :4, :], cut_flag=self.hairstyle_cut_flag)
        else:
            x_coarse = torch.zeros_like(x_coarse)
        if not self.opts.no_medium_mapper:
            x_medium = self.medium_mapping(x_medium, hairstyle_embedding[:, 4:8, :], cut_flag=self.hairstyle_cut_flag)
        else:
            x_medium = torch.zeros_like(x_medium)
        if not self.opts.no_fine_mapper:
            x_fine = self.fine_mapping(x_fine, color_embedding[:, 8:, :], cut_flag=self.color_cut_flag)
        else:
            x_fine = torch.zeros_like(x_fine)

        out = torch.cat([x_coarse, x_medium, x_fine], dim=1)
        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class SubHairMapper(Module):
    def __init__(self, opts, layernum):
        super(SubHairMapper, self).__init__()
        self.opts = opts
        self.layernum = layernum
        self.pixelnorm = PixelNorm()
        self.modulation_module_list = nn.ModuleList([ModulationModule(self.layernum) for i in range(5)])

    def forward(self, x, embedding, cut_flag=0):
        x = self.pixelnorm(x)
        for modulation_module in self.modulation_module_list:
            x = modulation_module(x, embedding, cut_flag)      
        return x
    

class ModulationModule(Module):
    def __init__(self, layernum):
        super(ModulationModule, self).__init__()
        self.layernum = layernum
        self.fc = Linear(512, 512)
        self.norm = LayerNorm([self.layernum, 512], elementwise_affine=False)
        self.gamma_function = Sequential(Linear(512, 512), LayerNorm([512]), LeakyReLU(), Linear(512, 512))
        self.beta_function = Sequential(Linear(512, 512), LayerNorm([512]), LeakyReLU(), Linear(512, 512))
        self.leakyrelu = LeakyReLU()

    def forward(self, x, embedding, cut_flag):
        x = self.fc(x)
        x = self.norm(x) 	
        if cut_flag == 1:
            return x
        gamma = self.gamma_function(embedding.float())
        beta = self.beta_function(embedding.float())
        # out = x * (1 + gamma) + beta
        out = x * gamma + beta
        out = self.leakyrelu(out)
        return out