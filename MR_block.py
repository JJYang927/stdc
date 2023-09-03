import torch
import torch.nn as nn
import torch.nn.functional as F


class conv3x3(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, groups=1):
        super(conv3x3, self).__init__()
        self.groups = groups
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.register_parameter('gate', nn.Parameter(torch.randn(1, 1, 3, 3)))
        self.in_channels = in_planes
        self.out_channels = out_planes
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        gate = self.sigmoid(self.gate)
        sobel_kernel = gate.repeat(self.out_channels, self.in_channels // self.groups, 1, 1)
        masked_weight = self.conv.weight * sobel_kernel.view(self.out_channels, self.in_channels // self.groups, 3, 3)
        x = F.conv2d(x, masked_weight, self.conv.bias, self.conv.stride, self.conv.padding, groups=self.groups)
        out = self.bn(x)
        return out


class conv1x1(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, groups=1):
        super(conv1x1, self).__init__()
        self.groups = groups
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.register_parameter('gate', nn.Parameter(torch.randn(1, 1, 1, 1)))
        self.in_channels = in_planes
        self.out_channels = out_planes
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        gate = self.sigmoid(self.gate)
        sobel_kernel = gate.repeat(self.out_channels, self.in_channels // self.groups, 1, 1)
        masked_weight = self.conv.weight * sobel_kernel.view(self.out_channels, self.in_channels // self.groups, 1, 1)
        x = F.conv2d(x, masked_weight, self.conv.bias, self.conv.stride, self.conv.padding, groups=self.groups)
        out = self.bn(x)
        return out


class MRBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(MRBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MRBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(MRBottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out