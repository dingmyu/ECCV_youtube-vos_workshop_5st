import torch
from torch import nn
import torch.nn.functional as F
# from torchvision import models
# import resnet as models
from torchE.nn import SyncBatchNorm2d
from pspnet import PSPNet as PSPNet_rgb
from pspnet_flow import PSPNet as PSPNet_flow


class PSP2S(nn.Module):
    def __init__(self, backbone='resnet', layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, use_softmax=True, use_aux=False, pretrained=True, syncbn=True, group_size=8, group=None):
        super(PSP2S, self).__init__()
        self.rgb_branch = PSPNet_rgb(backbone='resnet', layers=layers, classes=2, zoom_factor=zoom_factor, syncbn=True, group_size=group_size, group=group)
        self.flow_branch = PSPNet_flow(backbone='resnet', layers=layers, classes=2, zoom_factor=zoom_factor, syncbn=True, group_size=group_size, group=group)
        
    def forward(self, x, aug_mask, flow_patch):
        x1 = self.rgb_branch(x, aug_mask)
        x2 = self.flow_branch(flow_patch, aug_mask)
        res = x1 + x2
        # softmax
        res = F.log_softmax(res, dim=1)
        return res


if __name__ == '__main__':
#     import os
#     os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    x = torch.autograd.Variable(torch.FloatTensor(1, 3, 54 * 8 + 1, 54 * 8 + 1).zero_().cuda())                            
    f = torch.autograd.Variable(torch.FloatTensor(1, 2, 54 * 8 + 1, 54 * 8 + 1).zero_().cuda()) 
    p = torch.autograd.Variable(torch.FloatTensor(1, 1, 54 * 8 + 1, 54 * 8 + 1).zero_().cuda()) 
    model = PSP2S().cuda()
    print(model)
    output = model(x, p, f)
    print('PSPNet', output.size())
