import torch
import torch.nn as nn

from nets.CSPdarknet53_tiny import darknet53_tiny
from nets.attention import cbam_block, eca_block, se_block
 
# from CSPdarknet53_tiny import darknet53_tiny
# from attention import cbam_block, eca_block, se_block

attention_block = [se_block, cbam_block, eca_block]

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

class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi=0, pretrained=False):
        super(YoloBody, self).__init__()
        self.phi            = phi
        self.backbone       = darknet53_tiny(pretrained)

        self.sppcspc_p5        = SPPCSPC(512,256)
        self.sppcspc_p4        = SPPCSPC(256,128)
        
        
        self.upsample       = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv_for_P5    = BasicConv(512,256,1)
        self.conv_for_P4    = BasicConv(256,128,1)
        self.conv_for_P3    = BasicConv(128,64,1)
        
        self.conv1           = BasicConv(64,128,1)

        self.yolo_headP5    = yolo_head([512, len(anchors_mask[0]) * (5 + num_classes)],256)
        self.yolo_headP4    = yolo_head([256, len(anchors_mask[1]) * (5 + num_classes)],384)

        if 1 <= self.phi and self.phi <= 3:
            self.feat1_att      = attention_block[self.phi - 1](64)
            self.feat2_att      = attention_block[self.phi - 1](128)
            self.feat3_att      = attention_block[self.phi - 1](256)
            self.feat4_att      = attention_block[self.phi - 1](512)
    def forward(self, x):
        #---------------------------------------------------#
        #   生成CSPdarknet53_tiny的主干模型
        #   feat1的shape为104,104,64
        #   feat2的shape为52,52,128
        #   feat3的shape为26,26,256
        #   feat4的shape为13,13,512
        #---------- -----------------------------------------#
        
        feat1,feat2,feat3,feat4 = self.backbone(x)
        # 13,13,512 ->13,13,256
        # 26,26,256 ->26,26,128
        P5 = self.sppcspc_p5(feat4)
        P4 = self.sppcspc_p4(feat3)
        
        if 1 <= self.phi and self.phi <= 3:
            feat1 = self.feat1_att(feat1)
            feat2 = self.feat2_att(feat2)
            feat3 = self.feat3_att(P4)
            feat4 = self.feat4_att(P5)
        
        # 13,13,512 ->13,13,256
        # P5_conv = self.conv_for_P5(P5)
        # 13,13,256 -> 26,26,256
        P5_upsample = self.upsample(P5)
        
        if 1 <= self.phi and self.phi <= 3:
            P5_upsample = self.upsample_att(P5_upsample)
            
        # 26,26,256 cat 26,26,256 = 26,26,512
        P4 = torch.cat([feat3,P5_upsample],1)
        # 26,26,512 ->26,26,256
        P4 = self.conv_for_P5(P4)
        
        # 26,26,256 -> 26,26,128
        P4_conv = self.conv_for_P4(P4)
        # 26,26,128 -> 52,52,128
        P4_upsample = self.upsample(P4_conv)
        
        if 1 <= self.phi and self.phi <= 3:
            P5_upsample = self.upsample_att(P4_upsample)
            
        # 52,52,128 cat 52,52,128 = 52,52,256
        P3 = torch.cat([feat2,P4_upsample],1)
        # 52,52,256 -> 52,52,128
        P3 = self.conv_for_P4(P3)
        
        
        # 52,52,128 -> 52,52,64
        P3_conv = self.conv_for_P3(P3)
        # 52,52,64 -> 104,104,64
        P3_upsample = self.upsample(P3_conv)
        # 104,104,128
        # P3_upsample = self.conv1(P3_upsample)
        
        if 1 <= self.phi and self.phi <= 3:
            P3_upsample = self.upsample_att(P3_upsample)
            
        # 104,104,64 cat 104,104,64 = 104,104,128
        P2 = torch.cat([feat1,P3_upsample],1)
        # 104,104,128 -> 104,104,64
        P2 = self.conv_for_P3(P2)
        
        # 104,104,64 -> 52,52,64
        P2_downsample = nn.functional.interpolate(P2,scale_factor=0.5)
        # 52,52,64 -> 52,52,128
        P2_downsample = self.conv1(P2_downsample)
        # 52,52,128 cat 52,52,128 = 52,52,256
        P3 = torch.cat([P2_downsample,P3],1)
        P3 = torch.cat([P3,P4_upsample],1)
        #26,26,256 -> 13,13,256
        P4_downsample =nn.functional.interpolate(P4,scale_factor=0.5)
        #52,52,256 -> 26,26,256
        P3_downsample =nn.functional.interpolate(P3,scale_factor=0.5)
        # 26,26,256 cat ,26,26,128 = 26,26,256
        P5 = torch.cat([P4_downsample,P5],1)
        # P5 =torch.cat([P3_downsample,P5],1)
        P5 = self.conv_for_P5(P5)
        
        out0 = self.yolo_headP5(P5)
        out1 = self.yolo_headP4(P3)
        
        return out0, out1
