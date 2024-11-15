import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from lib.Res2Net_v1b import res2net50_v1b_26w_4s
from torch.autograd import Variable
import math


def cus_sample(feat, **kwargs):

    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=False)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM_Block(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(CBAM_Block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)
    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)

        return x


class SEM(nn.Module):
    def __init__(self, hchannel, channel):
        super(SEM, self).__init__()
        self.conv1_1 = Conv1x1(hchannel, channel // 4)
        self.conv = BasicConv2d(hchannel//2, channel//4, 3, padding=1)
        self.conv1_3 = BasicConv2d(channel // 4, channel // 4, kernel_size=(1, 3), padding=(0, 1))
        self.conv3_1 = BasicConv2d(channel // 4, channel // 4, kernel_size=(3, 1), padding=(1, 0))
        self.conv3_3_1 = BasicConv2d(channel // 4, channel // 4, 3, padding=1)
        self.conv1_5 = BasicConv2d(channel // 4, channel // 4, kernel_size=(1, 5), padding=(0, 2))
        self.dconv5_1 = BasicConv2d(channel // 4, channel // 4, kernel_size=(5, 1), padding=(2, 0))
        self.conv3_3_2 = BasicConv2d(channel // 4, channel // 4, 3, dilation=2, padding=2)
        self.conv1_7 = BasicConv2d(channel // 4, channel // 4, kernel_size=(1, 7), padding=(0, 3))
        self.dconv7_1 = BasicConv2d(channel // 4, channel // 4, kernel_size=(7, 1), padding=(3, 0))
        self.conv3_3_3 = BasicConv2d(channel // 4, channel // 4, 3, dilation=3, padding=3)
        self.conv3_3_4 = BasicConv2d(channel // 4, channel // 4, 3, dilation=4, padding=4)
        self.conv1_2 = Conv1x1(channel, channel)
        self.conv3_3 = BasicConv2d(channel, channel, 3, padding=1)

    def forward(self, lf, hf):
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)
        hf = self.conv1_1(hf)  # 16
        xc = torch.chunk(lf, 4, dim=1)

        x_0 = torch.cat((xc[0], hf), 1)  # 32
        x_0 = self.conv(x_0)    # x0  32-16
        x_0 = self.conv1_3(x_0 + xc[1])
        x0_1 = self.conv3_1(x_0)  # x0+x1
        x0_1 = self.conv3_3_1(x0_1)

        x_1 = torch.cat((x0_1, hf), 1)
        x_1 = self.conv(x_1)  # x1
        x1_2 = self.conv1_5(x_1 + x0_1 + xc[2])
        x1_2 = self.dconv5_1(x1_2)
        x1_2 = self.conv3_3_2(x1_2)

        x_2 = torch.cat((x1_2, hf), 1)
        x_2 = self.conv(x_2)
        x2_3 = self.conv1_7(x_2 + x1_2 + xc[3])
        x2_3 = self.dconv7_1(x2_3)
        x2_3 = self.conv3_3_3(x2_3)

        x_3 = torch.cat((x2_3, hf), 1)
        x_3 = self.conv(x_3)
        x3_4 = self.conv1_3(x_3 + x2_3)
        x3_4 = self.conv3_1(x3_4)
        x3_4 = self.conv3_3_4(x3_4)

        x2 = x0_1 + x1_2
        x3 = x2 + x2_3
        x4 = x3 + x3_4
        xx = self.conv1_2(torch.cat((x0_1, x2, x3, x4), dim=1))
        x = self.conv3_3(xx)

        return x


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        ll = x1 + x2 + x3 + x4
        lh = -x1 + x2 - x3 + x4
        hl = -x1 - x2 + x3 + x4
        hh = x1 - x2 - x3 + x4

        return ll, lh, hl, hh


class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def diff_x(self, input, r):
        assert input.dim() == 4

        left = input[:, :, r:2 * r + 1]
        middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
        right = input[:, :, -1:] - input[:, :, -2 * r - 1:    -r - 1]

        output = torch.cat([left, middle, right], dim=2)

        return output

    def diff_y(self, input, r):
        assert input.dim() == 4

        left = input[:, :, :, r:2 * r + 1]
        middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
        right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1:    -r - 1]

        output = torch.cat([left, middle, right], dim=3)

        return output

    def forward(self, x):
        assert x.dim() == 4
        return self.diff_y(self.diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


class GF(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GF, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)
        self.epss = 1e-12

    def forward(self, lr_x, lr_y, hr_x, l_a):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        lr_x = lr_x.double()
        lr_y = lr_y.double()
        hr_x = hr_x.double()
        l_a = l_a.double()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2 * self.r + 1 and w_lrx > 2 * self.r + 1

        ## N
        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        # l_a = torch.abs(l_a)
        l_a = torch.abs(l_a) + self.epss

        t_all = torch.sum(l_a)
        l_t = l_a / t_all

        ## mean_attention
        mean_a = self.boxfilter(l_a) / N
        ## mean_a^2xy
        mean_a2xy = self.boxfilter(l_a * l_a * lr_x * lr_y) / N
        ## mean_tax
        mean_tax = self.boxfilter(l_t * l_a * lr_x) / N
        ## mean_ay
        mean_ay = self.boxfilter(l_a * lr_y) / N
        ## mean_a^2x^2
        mean_a2x2 = self.boxfilter(l_a * l_a * lr_x * lr_x) / N
        ## mean_ax
        mean_ax = self.boxfilter(l_a * lr_x) / N

        ## A
        temp = torch.abs(mean_a2x2 - N * mean_tax * mean_ax)
        A = (mean_a2xy - N * mean_tax * mean_ay) / (temp + self.eps)
        ## b
        b = (mean_ay - A * mean_ax) / (mean_a)

        # --------------------------------
        # Mean
        # --------------------------------
        A = self.boxfilter(A) / N
        b = self.boxfilter(b) / N

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return (mean_A * hr_x + mean_b).float()


class getAlpha(nn.Module):
    def __init__(self, in_channels):
        super(getAlpha, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class ESM(nn.Module):
    def __init__(self):
        super(ESM, self).__init__()
        self.conv_cat = BasicConv2d(64*2, 64, 3, padding=1)
        self.CA = ChannelAttention(64)
        self.SA = SpatialAttention()
        self.GA = getAlpha(64)
        self.DWT = DWT()
        self.GF = GF(r=2, eps=1e-2)
        #self.up = torch.nn.PixelShuffle(2)
        self.up = cus_sample
        self.conv = Conv1x1(64, 64)
        self.conv_ll = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.conv_lh = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.conv_hl = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.conv_hh = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.once_conv_1 = BasicConv2d(64 + 64, 64, kernel_size=1)
        self.once_conv_2 = BasicConv2d(64*2, 64, kernel_size=1)

        self.block = nn.Sequential(
            BasicConv2d(64, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 1, 3, padding=1))

    def forward(self, x):
        x1, x2, x3, x4 = self.DWT(x)

        x_ll = self.GF(x1, x1, x1, x1)
        x_lh = self.GF(x2, x2, x2, x2)
        x_hl = self.GF(x3, x3, x3, x3)
        x_hh = self.GF(x4, x4, x4, x4)
        x_ll_w = torch.nn.functional.softmax(x_ll, dim=1)
        x_lh_w = torch.nn.functional.softmax(x_lh, dim=1)
        x_hl_w = torch.nn.functional.softmax(x_hl, dim=1)
        x_hh_w = torch.nn.functional.softmax(x_hh, dim=1)

        x_ll = self.conv_ll(torch.matmul(x1, x_ll_w))
        x_lh = self.conv_lh(torch.matmul(x2, x_lh_w))
        x_hl = self.conv_hl(torch.matmul(x3, x_hl_w))
        x_hh = self.conv_hh(torch.matmul(x4, x_hh_w))

        x_c = torch.cat((x_lh, x_hl), dim=1)
        x_c = self.once_conv_1(x_c)

        x_c_w = self.GA(x_c)
        x_lh = x_lh * x_c_w
        x_hl = x_hl * (1 - x_c_w)

        x_ll = self.SA(x_ll) * x_ll

        x_l = x_ll + x_lh

        x_hh = self.conv(x_hh)
        x_hh_w = self.CA(x_hh)
        x_hh_1 = x_hh * x_hh_w
        x_h = x_hh_1 + x_hl

        f1 = self.once_conv_2(torch.cat((x_h, x_l), dim=1))
        out1 = self.conv(x_h + f1)
        out1 = self.up(out1, scale_factor=2)
        out = x + out1
        out = self.block(out)

        return out


# without BN version
class ASPP(nn.Module):
    def __init__(self, in_channel=64, out_channel=64):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel,out_channel, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(out_channel * 5, out_channel, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net


class AP_MP(nn.Module):
    def __init__(self, stride=8):
        super(AP_MP, self).__init__()
        self.sz=stride
        self.gapLayer=nn.AvgPool2d(kernel_size=self.sz, stride=8)
        self.gmpLayer=nn.MaxPool2d(kernel_size=self.sz, stride=8)

    def forward(self, x1, x2):
        apimg=self.gapLayer(x1)
        mpimg=self.gmpLayer(x2)
        byimg=torch.norm(abs(apimg-mpimg), p=2, dim=1, keepdim=True)
        return byimg


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.aspp = ASPP(64)
        self.conv1x1 = Conv1x1(channel*2+1, 64)
        self.conv = BasicConv2d(64, 64, 3, padding=1)
        self.glbamp = AP_MP()
        self.CBAM = CBAM_Block(channel)
        self.channel = channel

    def forward(self, x_i, x_e, x_ii):

        x1_i = self.aspp(x_i)
        x2_i = self.CBAM(x_i)

        xw_ii = self.CBAM(x_ii)

        x1 = x1_i * xw_ii + x_i
        x2 = self.conv(x2_i * x_i)

        fe = self.glbamp(x1, x2)
        fe = fe/math.sqrt(self.channel)

        if fe.size()[2:] != x_e.size()[2:]:
            fe = F.interpolate(fe, size=x_e.size()[2:], mode='bilinear', align_corners=False)
        edge = fe * x_e + x_e

        out1 = self.conv1x1(torch.cat([x1, edge, x2], dim=1))
        out = self.conv(out1 + x_i)

        return out


class Network(nn.Module):
    def __init__(self, imagenet_pretrained=True):
        super(Network, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained)
        # if self.training:
        # self.initialize_weights()

        self.rfb1_1 = RFB_modified(256, 64)
        self.rfb2_1 = RFB_modified(512, 64)
        self.rfb3_1 = RFB_modified(1024, 64)
        self.rfb4_1 = RFB_modified(2048, 64)
        self.conv = Conv1x1(2048, 64)
        self.block = nn.Sequential(
            BasicConv2d(64, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1))

        self.upsample = cus_sample
        self.edge = ESM()
        self.sem1 = SEM(64, 64)
        self.sem2 = SEM(64, 64)
        self.sem3 = SEM(64, 64)
        self.sem4 = SEM(64, 64)

        self.aspp = ASPP(64)

        self.fam1 = FAM(64)
        self.fam2 = FAM(64)
        self.fam3 = FAM(64)

        self.reduce2 = Conv1x1(64, 128)
        self.reduce3 = Conv1x1(64, 256)

        self.predictor1 = nn.Conv2d(64, 1, 3, padding=1)
        self.predictor2 = nn.Conv2d(128, 1, 3, padding=1)
        self.predictor3 = nn.Conv2d(256, 1, 3, padding=1)
        self.predictor4 = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x):
        #x1, x2, x3, x4 = self.resnet(x)
        x0 = self.resnet.conv1(x)
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)
        x0 = self.resnet.maxpool(x0)
        x1 = self.resnet.layer1(x0)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        x1_rfb = self.rfb1_1(x1)  # channel -> 64
        x2_rfb = self.rfb2_1(x2)  # channel -> 64
        x3_rfb = self.rfb3_1(x3)  # channel -> 64
        x4_rfb = self.rfb4_1(x4)  # channel -> 64

        x2_r = -1 * (torch.sigmoid(x2_rfb)) + 1
        x2_r = self.block(x2_r)

        x3_r = -1 * (torch.sigmoid(x3_rfb)) + 1
        x3_r = self.block(x3_r)

        x4_r = -1 * (torch.sigmoid(x4_rfb)) + 1
        x4_r = self.block(x4_r)

        x2a = self.sem2(x2_rfb, x2_r)
        x2a = self.upsample(x2a, scale_factor=2)
        x3a = self.sem3(x3_rfb, x3_r)
        x3a = self.upsample(x3a, scale_factor=4)
        x4a = self.sem4(x4_rfb, x4_r)
        x4a = self.upsample(x4a, scale_factor=8)
        x41 = self.conv(x4)
        x41 = self.upsample(x41, scale_factor=8)
        x4 = self.aspp(x41)

        edge = self.edge(x1_rfb)
        edge_att = torch.sigmoid(edge)

        x34 = self.fam1(x4a, edge_att, x4)
        x234 = self.fam2(x3a, edge_att, x34)
        x1234 = self.fam3(x2a, edge_att, x234)

        x34 = self.reduce3(x34)
        x234 = self.reduce2(x234)

        o3 = self.predictor3(x34)
        o3 = F.interpolate(o3, scale_factor=4, mode='bilinear', align_corners=False)
        o2 = self.predictor2(x234)
        o2 = F.interpolate(o2, scale_factor=4, mode='bilinear', align_corners=False)
        o1 = self.predictor1(x1234)
        o1 = F.interpolate(o1, scale_factor=4, mode='bilinear', align_corners=False)
        #oe = self.predictor4(edge)
        oe = F.interpolate(edge_att, scale_factor=4, mode='bilinear', align_corners=False)

        return o3, o2, o1, oe


if __name__ == '__main__':
    import numpy as np
    from time import time
    net = Network(imagenet_pretrained=False)
    net.eval()

    dump_x = torch.randn(1, 3, 352, 352)
    frame_rate = np.zeros((1000, 1))
    for i in range(1000):
        start = time()
        y = net(dump_x)
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    print(y.shape)