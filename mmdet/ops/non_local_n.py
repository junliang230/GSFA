import torch
import torch.nn as nn
from mmcv.cnn import constant_init, normal_init
from torch.nn import Parameter
import math

class PixelAGG_N(nn.Module):
    """Non-local module for pixel-level fusion among frames.


    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio.
        use_scale (bool): Whether to scale pairwise_weight by 1/inter_channels.
        conv_cfg (dict): The config dict for convolution layers.
            (only applicable to conv_out)
        norm_cfg (dict): The config dict for normalization layers.
            (only applicable to conv_out)
        mode (str): Options are `embedded_gaussian` and `dot_product`.
    """

    def __init__(self,
                 in_channels,
                 reduction=2,
                 use_scale=True,
                 zeros_init=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 mode='embedded_gaussian'):
        super(PixelAGG_N, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian', 'dot_product']

        # g, theta, phi are actually `nn.Conv2d`. Here we use ConvModule for
        # potential usage.
        self.g = nn.Conv2d(
            self.in_channels,
            self.inter_channels,
            kernel_size=1)
        self.theta = nn.Conv2d(
            self.in_channels,
            self.inter_channels,
            kernel_size=1)
        self.phi = nn.Conv2d(
            self.in_channels,
            self.inter_channels,
            kernel_size=1)
        self.conv_out = nn.Conv2d(
            self.inter_channels,
            self.in_channels,
            kernel_size=1)
        # self.gamma = Parameter(torch.zeros(1))

        self.init_weights(zeros_init=zeros_init)

    def init_weights(self, std=0.01, zeros_init=True):
        for m in [self.g, self.theta, self.phi]:
            normal_init(m, std=std)
        if zeros_init:
            constant_init(self.conv_out, 0)
        else:
            normal_init(self.conv_out, std=std)

    def embedded_gaussian(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1]**0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, x):
        n, _, h, w = x.shape
        if n > 1000:
            num = 300
            output = x.clone()
            len_ = 1 #math.ceil(n/num)
            for i in range(len_):
                x1 = x[i*num:min((i+1)*num, n), :, :, :]
                # g_x: [1, C, N*HxW]
                g_x = self.g(x1).view(1, self.inter_channels, -1)
                # g_x = g_x.permute(0, 2, 1)

                # theta_x: [1, N*HxW, C]
                theta_x = self.theta(x1).view(1, self.inter_channels, -1)
                theta_x = theta_x.permute(0, 2, 1)

                # phi_x: [1, C, N*HxW]
                phi_x = self.phi(x1).view(1, self.inter_channels, -1)
                # phi_x = phi_x.permute(0, 2, 1)

                pairwise_func = getattr(self, self.mode)
                # pairwise_weight: [1, N*HxW, N*HxW]
                pairwise_weight = pairwise_func(theta_x, phi_x)

                # y: [1, C, N*HxW]
                y = torch.matmul(g_x, pairwise_weight)
                # y: [N, C, H, W]
                y = y.reshape(min((i+1)*num, n)-i*num, self.inter_channels, h, w)

                # output = x.clone()
                output[i*num:min((i+1)*num, n), :, :, :] = x[i*num:min((i+1)*num, n), :, :, :] + self.conv_out(y)
        else:
            # g_x: [1, C, N*HxW]
            g_x = self.g(x).view(1, self.inter_channels, -1)
            # g_x = g_x.permute(0, 2, 1)

            # theta_x: [1, N*HxW, C]
            theta_x = self.theta(x).view(1, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)

            # phi_x: [1, C, N*HxW]
            phi_x = self.phi(x).view(1, self.inter_channels, -1)
            # phi_x = phi_x.permute(0, 2, 1)

            pairwise_func = getattr(self, self.mode)
            # pairwise_weight: [1, N*HxW, N*HxW]
            pairwise_weight = pairwise_func(theta_x, phi_x)

            # y: [1, C, N*HxW]
            y = torch.matmul(g_x, pairwise_weight)
            # y: [N, C, H, W]
            y = y.reshape(n, self.inter_channels, h, w)

            output = x + self.conv_out(y)

        return output