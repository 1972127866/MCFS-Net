import torch
from torch import nn
from utils.style_loss import StyleLoss

class TextureLoss(nn.Module):
    def __init__(self):
        super(TextureLoss, self).__init__()
        self.style_loss=StyleLoss()

    def crop_image(self,img,sample_size):
        cropped=torch.zeros(img.shape[0],sample_size,sample_size,3).cuda()
        i=103
        j=103
        cropped=img[:,i:i+sample_size,j:j+sample_size]

        return cropped

    def get_texture_loss(self,real,fake):

        fake_crop=self.crop_image(fake.permute(0,2,3,1),50) # B 50 50 3
        fake_crop=fake_crop.permute(0,3,1,2) # 1 3 50 50

        real_crop=self.crop_image(real.permute(0,2,3,1),50) # B 50 50 3
        real_crop=real_crop.permute(0,3,1,2) # 1 3 50 50
        
        percept_loss = 250*self.style_loss(fake_crop,real_crop).mean()

        return percept_loss