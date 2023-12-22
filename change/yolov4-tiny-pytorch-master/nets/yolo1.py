import torch
import torch.nn as nn

from nets.CSPdarknet53_tiny import darknet53_tiny
from nets.attention import cbam_block, eca_block, se_block

# from CSPdarknet53_tiny import darknet53_tiny
# from attention import cbam_block, eca_block, se_block

attention_block = [se_block, cbam_block, eca_block]

#-------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   Conv2d + BatchNormalization + LeakyReLU
#-------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()

        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        x = self.downsample(x)
        return x


#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        BasicConv(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m
#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
# class SPPF(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.maxpool = nn.MaxPool2d(5, 1, padding=2)

#     def forward(self, x):
#         o1 = self.maxpool(x)
#         o2 = self.maxpool(o1)
#         o3 = self.maxpool(o2)
#         return torch.cat([x, o1, o2, o3], dim=1)

class SPPCSPC(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(13, 9, 5)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = BasicConv(c1, c_, 1, 1)
        self.cv2 = BasicConv(c1, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv3 = BasicConv(4 * c_, c_, 1, 1)
        self.cv4 = BasicConv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.cv3(torch.cat([m(x1) for m in self.m] + [x1], 1))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))
    
class SPPF(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SPPF, self).__init__()
        self.cv1      = BasicConv(512,256,1)
        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])
        self.cv2      = BasicConv(1024,512,1)

    def forward(self, x):
        x =self.cv1(x)
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)
        features =self.cv2(features)

        return features

# def make_three_conv(filters_list, in_filters):
#     m = nn.Sequential(
#         BasicConv(in_filters, filters_list[0], 1),
#         BasicConv(filters_list[0], filters_list[1], 3),
#         BasicConv(filters_list[1], filters_list[0], 1),
#     )
#     return m

class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi=0, pretrained=False):
        super(YoloBody, self).__init__()
        self.phi            = phi
        self.backbone       = darknet53_tiny(pretrained)
        self.spp            = SPPF()
        self.sppcspc        = SPPCSPC(512,256)

        self.downsample = Downsample(128,256)
        self.conv_for_P5    = BasicConv(512,256,1)
        self.yolo_headP5    = yolo_head([512, len(anchors_mask[0]) * (5 + num_classes)],256)

        self.upsample       = Upsample(256,128)
        self.yolo_headP4    = yolo_head([256, len(anchors_mask[1]) * (5 + num_classes)],512)

        if 1 <= self.phi and self.phi <= 3:
            self.feat1_att      = attention_block[self.phi - 1](128)
            self.feat2_att      = attention_block[self.phi - 1](256)
            self.feat3_att      = attention_block[self.phi - 1](512)
            self.upsample_att   = attention_block[self.phi - 1](128)
            # self.dosample_att   = attention_block[self.phi - 1](256)
    def forward(self, x):
        #---------------------------------------------------#
        #   生成CSPdarknet53_tiny的主干模型
        #   feat1的shape为104,104,64
        #   feat2的shape为52,52,128
        #   feat3的shape为13,13,512
        #---------------------------------------------------#
        feat1,feat2,feat3 = self.backbone(x)
        if 1 <= self.phi and self.phi <= 3:
            feat1 = self.feat1_att(feat1)
            feat2 = self.feat2_att(feat2)
            feat3 = self.feat3_att(feat3)


        # spp_feat1 = self.spp(feat1)
        # spp_feat2 = self.spp(feat2)
        # conv3_feat1 =self.conv3_f1(spp_feat1)
        # conv3_feat2 =self.conv3_f2(spp_feat2)

        #13,13,512 > 13,13,512
        # spp_feat3 = self.spp(feat3)
        #13,13,512>13,13,256
        
        spp_feat3 = self.sppcspc(feat3)

        # conv3_feat3 =self.conv3_f3(spp_feat3)

        # 13,13,512 -> 13,13,256
        # P5 = self.conv_for_P5(spp_feat3)
        # 13,13,256 -> 13,13,512 -> 13,13,255
        out0 = self.yolo_headP5(P5) 
        
        #52,52,128 > 26,26,256
        P3_Downsample = self.downsample(feat1)
        #13,13,256 > 26,26,128
        P5_Upsample = self.upsample(P5)

        if 1 <= self.phi and self.phi <= 3:
            P5_Upsample = self.upsample_att(P5_Upsample)
            # P3_Dosample = self.dosample_att(P3_Dosample)

        # 26,26,256 + 26,26,128 -> 26,26,384
        # P4 = torch.cat([P5_Upsample,P5_Upsample],axis=1)

        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([feat2,P3_Downsample],axis=1)

        # 26,26,512 -> 26,26,256 -> 26,26,255
        out1 = self.yolo_headP4(P4)
        
        return out0, out1

# input_tensor = torch.randn(1,128,52,52)
# model = Downsample(128,256)
# a =model(input_tensor)
# print(a.size())
# model = SPPCSPC(512,256)
# a = model(input_tensor)
# print(a.size())
# model = SPPF()
# output_tensor =model(input_tensor)
# print(output_tensor.size())
# print("原始输入张量统计信息:")
# print("平均值:", torch.mean(input_tensor))
# print("标准差:", torch.std(input_tensor))
# print("最小值:", torch.min(input_tensor))
# print("最大值:", torch.max(input_tensor))

# print("\nSPPF 输出张量统计信息:")
# print("平均值:", torch.mean(output_tensor))
# print("标准差:", torch.std(output_tensor))
# print("最小值:", torch.min(output_tensor))
# print("最大值:", torch.max(output_tensor))