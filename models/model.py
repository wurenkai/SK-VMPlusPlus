import torch
import torch.nn as nn
import torch.nn.functional as F
from models.res2net import res2net50_v1b_26w_4s
import torchvision
from mamba_ssm import Mamba

class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
                d_model=input_dim//4, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale= nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2,x_mamba3,x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out

class SKVMPlusPlus(nn.Module):
    def __init__(self):
        super(SKVMPlusPlus, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.x5_dem_1 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_dem_1 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_dem_1 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x2_dem_1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x1_dem_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x5_x4 = nn.Sequential(PVMLayer(input_dim=64, output_dim=64), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_x3 = nn.Sequential(PVMLayer(input_dim=64, output_dim=64), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_x2 = nn.Sequential(PVMLayer(input_dim=64, output_dim=64), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x2_x1 = nn.Sequential(PVMLayer(input_dim=64, output_dim=64), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.x5_x4_x3 = nn.Sequential(PVMLayer(input_dim=64, output_dim=64), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_x3_x2 = nn.Sequential(PVMLayer(input_dim=64, output_dim=64), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_x2_x1 = nn.Sequential(PVMLayer(input_dim=64, output_dim=64), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.x5_x4_x3_x2 = nn.Sequential(PVMLayer(input_dim=64, output_dim=64), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_x3_x2_x1 = nn.Sequential(PVMLayer(input_dim=64, output_dim=64), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))
        
        self.x5_x4_x3_x2_x1 = nn.Sequential(PVMLayer(input_dim=64, output_dim=64), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))

        self.level4 = nn.Sequential(PVMLayer(input_dim=64, output_dim=64), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.level3 = nn.Sequential(PVMLayer(input_dim=64, output_dim=64), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.level2 = nn.Sequential(PVMLayer(input_dim=64, output_dim=64), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.level1 = nn.Sequential(PVMLayer(input_dim=64, output_dim=64), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x5_dem_5 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.output4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output1 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))

    def forward(self, x):
        input = x

        # '''
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x1 = self.resnet.maxpool(x)      
        # ---- low-level features ----
        x2 = self.resnet.layer1(x1)      
        x3 = self.resnet.layer2(x2)    
        x4 = self.resnet.layer3(x3)    
        x5 = self.resnet.layer4(x4)    
        # '''


        x5_dem_1 = self.x5_dem_1(x5)
        x4_dem_1 = self.x4_dem_1(x4)
        x3_dem_1 = self.x3_dem_1(x3)
        x2_dem_1 = self.x2_dem_1(x2)
        x1_dem_1 = self.x1_dem_1(x1)

        x5_4 = self.x5_x4(abs(F.upsample(x5_dem_1,size=x4.size()[2:], mode='bilinear')+x4_dem_1))
        x4_3 = self.x4_x3(abs(F.upsample(x4_dem_1,size=x3.size()[2:], mode='bilinear')+x3_dem_1))
        x3_2 = self.x3_x2(abs(F.upsample(x3_dem_1,size=x2.size()[2:], mode='bilinear')+x2_dem_1))
        x2_1 = self.x2_x1(abs(F.upsample(x2_dem_1,size=x1.size()[2:], mode='bilinear')+x1_dem_1))


        x5_4_3 = self.x5_x4_x3(abs(F.upsample(x5_4, size=x4_3.size()[2:], mode='bilinear') + x4_3))
        x4_3_2 = self.x4_x3_x2(abs(F.upsample(x4_3, size=x3_2.size()[2:], mode='bilinear') + x3_2))
        x3_2_1 = self.x3_x2_x1(abs(F.upsample(x3_2, size=x2_1.size()[2:], mode='bilinear') + x2_1))


        x5_4_3_2 = self.x5_x4_x3_x2(abs(F.upsample(x5_4_3, size=x4_3_2.size()[2:], mode='bilinear') + x4_3_2))
        x4_3_2_1 = self.x4_x3_x2_x1(abs(F.upsample(x4_3_2, size=x3_2_1.size()[2:], mode='bilinear') + x3_2_1))

        x5_4_3_2_1 = self.x5_x4_x3_x2_x1(abs(F.upsample(x5_4_3_2, size=x4_3_2_1.size()[2:], mode='bilinear') + x4_3_2_1))

        level4 = self.level4(x4_dem_1 + x5_4)
        level3 = self.level3(x3_dem_1 + x4_3 + x5_4_3)
        level2 = self.level2(x2_dem_1 + x3_2 + x4_3_2 + x5_4_3_2)
        level1 = self.level1(x1_dem_1 + x2_1 + x3_2_1 + x4_3_2_1 + x5_4_3_2_1)

        x5_dem_5 = self.x5_dem_5(x5)
        output4 = self.output4(F.upsample(x5_dem_5,size=level4.size()[2:], mode='bilinear') + level4)
        output3 = self.output3(F.upsample(output4,size=level3.size()[2:], mode='bilinear') + level3)
        output2 = self.output2(F.upsample(output3,size=level2.size()[2:], mode='bilinear') + level2)
        output1 = self.output1(F.upsample(output2,size=level1.size()[2:], mode='bilinear') + level1)

        output = F.upsample(output1, size=input.size()[2:], mode='bilinear')
        if self.training:
            return output
        return output


class LossNet(torch.nn.Module):
    def __init__(self):
        super(LossNet, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))


    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        loss = 0.0
        x = input
        y = target

        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.mse_loss(x, y)
        return loss