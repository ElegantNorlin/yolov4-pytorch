from collections import OrderedDict

import torch
import torch.nn as nn

from nets.CSPdarknet import darknet53
# yolo4.py文件中除了最后yolohead没有用conv2d函数定义的卷积，其余都是conv2d卷积，该卷积的特点是输入、输出特征尺寸(宽和高)相等
# 定义CBL卷积层
# filter_in输入特征通道数
# filter_out输出特征通道数
# kernel_size卷积核尺寸
# stride卷积步长
def conv2d(filter_in, filter_out, kernel_size, stride=1):
    # 动态的定义填充padding的数值，这样卷积出来的特征尺寸才会与原来相等
    # if语句中的情况是：如果卷积核尺寸为1，则填充为0，此时CBL卷积输出特征的尺寸与输入特征尺寸相等
    # pad的计算公式带入卷积公式得到输出特征尺寸与输入特征尺寸相等
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#---------------------------------------------------#
#   SPP（空间金字塔池化）结构，利用不同大小的池化核进行池化
#   池化后堆叠
#---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    # pool_sizes为池化核尺寸列表
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()
        # 定义三个最大池化网络
        # pool_size为最大池化核尺寸
        # pool_size//2为最大池化填充padding
        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        # [::-1]从后向前取所有元素
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        # 特征拼接
        features = torch.cat(features + [x], dim=1)

        return features

#---------------------------------------------------#
#   卷积 + 上采样
#   上采样函数中先是一层卷积，然后才是上采样层
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   三次卷积块
#   这三次卷积位于SPP模块前后
#---------------------------------------------------#
# in_filters为第一次卷积的输入特征通道数
def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   五次卷积块
#---------------------------------------------------#
def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#   
        #   生成CSPdarknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        self.backbone = darknet53(None)
        # 第一个三层卷积
        self.conv1 = make_three_conv([512,1024],1024)
        # SPP模块
        self.SPP = SpatialPyramidPooling()
        # 第二个三层卷积
        self.conv2 = make_three_conv([512,1024],2048)

        self.upsample1 = Upsample(512,256)
        self.conv_for_P4 = conv2d(512,256,1)
        self.make_five_conv1 = make_five_conv([256, 512],512)

        self.upsample2 = Upsample(256,128)
        self.conv_for_P3 = conv2d(256,128,1)
        self.make_five_conv2 = make_five_conv([128, 256],256)

        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        final_out_filter2 = num_anchors * (5 + num_classes)
        self.yolo_head3 = yolo_head([256, final_out_filter2],128)

        self.down_sample1 = conv2d(128,256,3,stride=2)
        self.make_five_conv3 = make_five_conv([256, 512],512)

        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        final_out_filter1 =  num_anchors * (5 + num_classes)
        self.yolo_head2 = yolo_head([512, final_out_filter1],256)

        self.down_sample2 = conv2d(256,512,3,stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024],1024)

        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        final_out_filter0 =  num_anchors * (5 + num_classes)
        self.yolo_head1 = yolo_head([1024, final_out_filter0],512)


    def forward(self, x):
        #  backbone
        # x2,x1,x0的尺寸从小到大排序
        x2, x1, x0 = self.backbone(x)

        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 
        P5 = self.conv1(x0)
        # 13,13,512 -> 13,13,2048
        P5 = self.SPP(P5)
        # 13,13,2048 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        P5 = self.conv2(P5)

        # 13,13,512 -> 13,13,256 -> 26,26,256
        P5_upsample = self.upsample1(P5)
        # 26,26,512 -> 26,26,256
        P4 = self.conv_for_P4(x1)
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([P4,P5_upsample],axis=1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        P4 = self.make_five_conv1(P4)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        P4_upsample = self.upsample2(P4)
        # 52,52,256 -> 52,52,128
        P3 = self.conv_for_P3(x2)
        # 52,52,128 + 52,52,128 -> 52,52,256
        P3 = torch.cat([P3,P4_upsample],axis=1)
        # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        P3 = self.make_five_conv2(P3)

        # 52,52,128 -> 26,26,256
        P3_downsample = self.down_sample1(P3)
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([P3_downsample,P4],axis=1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        P4 = self.make_five_conv3(P4)

        # 26,26,256 -> 13,13,512
        P4_downsample = self.down_sample2(P4)
        # 13,13,512 + 13,13,512 -> 13,13,1024
        P5 = torch.cat([P4_downsample,P5],axis=1)
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        P5 = self.make_five_conv4(P5)

        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,52,52)
        #---------------------------------------------------#
        out2 = self.yolo_head3(P3)
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,26,26)
        #---------------------------------------------------#
        out1 = self.yolo_head2(P4)
        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,13,13)
        #---------------------------------------------------#
        out0 = self.yolo_head1(P5)

        return out0, out1, out2

