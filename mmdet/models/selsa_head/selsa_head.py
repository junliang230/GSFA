import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from mmcv.cnn import constant_init, kaiming_init, normal_init
from mmdet.core import auto_fp16
from ..registry import SELSA_HEAD

@SELSA_HEAD.register_module
class SelsaHead(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 out_channels=1024,
                 nongt_dim=3,  # number of frames
                 feat_dim=1024,  # 1024
                 dim=[1024, 1024, 1024],
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 apply=True):
        super(SelsaHead, self).__init__()
        self.norm_eval = norm_eval
        self.norm_cfg = norm_cfg
        self.apply = apply
        self.fc1 = nn.Linear(in_channels*7*7, 1024)
        self.fc2 = nn.Linear(1024, out_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.mish = Mish()
        self.attention_1 = nn.Sequential(*[semantic_aggregation(nongt_dim, feat_dim, dim)])
        self.attention_2 = nn.Sequential(*[semantic_aggregation(nongt_dim, feat_dim, dim)])
        self.with_PixelAGG = False
        if self.with_PixelAGG:
            from mmdet.ops.non_local import PixelAGG
            # from mmdet.ops.non_local_n import PixelAGG_N
            self.pixle_agg = PixelAGG(in_channels=256, reduction=8, use_scale=False, zeros_init=True) #reduction=8
            # self.pixle_agg = PixelAGG_N(in_channels=256, reduction=8, use_scale=False, zeros_init=True) #reduction=8
        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01) #kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, 0, 0.01) #which inition method should be chosen?

    @auto_fp16()
    def forward(self, x):
        if self.with_PixelAGG:
            x = self.pixle_agg(x)
            # x = self.pixle_agg_c(x)
            # x = x1 + x2 - x
            # x = self.conv1(x)
            # x = self.relu(x)
        out = x.reshape(x.size(0), -1)
        if not self.apply:
            out = self.fc1(out)
            return out
        out = self.fc1(out)
        residual_1, _ = self.attention_1(out)
        out1 = out + residual_1
        out1 = self.relu(out1)

        out1 = self.fc2(out1)
        residual_2, _ = self.attention_2(out1)
        out1 = out1 + residual_2 + out
        # out = out.split(min(512,out.size(0)), 0)[0]
        out1 = self.relu(out1)

        # if self.training:
        #     return [out1, similar]

        return out1

    def train(self, mode=True):
        super(SelsaHead, self).train(mode)
        if self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only ???
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


class SE(nn.Module):
    def __init__(self, in_channels=256, channels=256, se_ratio=2):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, channels // se_ratio, kernel_size=1, padding=0),
            nn.BatchNorm2d(channels // se_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // se_ratio, channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False) #false
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        y = self.sigmoid(out)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x1 = self.conv1(x1)
        x1 = F.interpolate(x1, x.size()[2:])
        x2 = self.sigmoid(x1)
        x2 = x * x2.expand_as(x)

        return x2+x1


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel=256, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class semantic_aggregation(nn.Module):
    def __init__(self,
                 nongt_dim,  # number of frames
                 feat_dim,  # 1024
                 dim=[1024, 1024, 1024],
                 ):
        super(semantic_aggregation, self).__init__()

        self.nongt_dim = nongt_dim
        self.feat_dim = feat_dim
        self.dim = dim
        self.dim[0] = dim[1] = 1024
        self.fc1 = nn.Linear(dim[0], dim[0])
        self.fc2 = nn.Linear(dim[0], dim[1])
        # self.fc3 = nn.Linear(dim[0], dim[0])
        # self.fc4 = nn.Linear(dim[0], dim[0])
        # self.fc5 = nn.Linear(dim[0], dim[0])
        self.conv = nn.Conv2d(dim[0], dim[1], kernel_size=1)

    def forward(self, x):
        q_data = self.fc1(x)
        q_data_batch = q_data.view(-1, 1, self.dim[0]).permute(1, 0, 2)  # (1, 6300, 1024)
        k_data = self.fc2(x)
        k_data_batch = k_data.view(-1, 1, self.dim[1]).permute(1, 2, 0)  # (1, 1024, 6300)
        # q_data1 = self.fc3(x)
        # q_data_batch1 = q_data1.view(-1, 1, self.dim[0]).permute(1, 0, 2)  # (1, 6300, 1024)
        # q_data2 = self.fc4(x)
        # q_data_batch2 = q_data2.view(-1, 1, self.dim[0]).permute(1, 0, 2)  # (1, 6300, 1024)
        # q_data3 = self.fc5(x)
        # q_data_batch3 = q_data3.view(-1, 1, self.dim[0]).permute(1, 0, 2)  # (1, 6300, 1024)
        # k_data1 = q_data1.view(-1, 1, self.dim[0])
        # k_data_batch1 = k_data1.permute(1, 2, 0)  # (1, 1024, 6300)
        # aff1 = torch.bmm(q_data_batch1, k_data_batch)
        # aff1 = (1.0 / math.sqrt(float(self.dim[1]))) * aff1
        # aff2 = torch.bmm(q_data_batch2, k_data_batch)
        # aff2 = (1.0 / math.sqrt(float(self.dim[1]))) * aff2
        # aff3 = torch.bmm(q_data_batch3, k_data_batch)
        # aff3 = (1.0 / math.sqrt(float(self.dim[1]))) * aff3
        v_data = x #x.clone().detach().requires_grad_(True)  # (6300, 1024) ??? how to copy
        aff = torch.bmm(q_data_batch, k_data_batch)  # (1, 6300, 6300)
        aff_scale = (1.0 / math.sqrt(float(self.dim[1]))) * aff
        # aff_scale = aff_scale + aff1 #+ aff2 #+ aff3
        aff_scale = aff_scale.permute(1, 0, 2)  # (6300, 1, 6300)
        weighted_aff = aff_scale

        aff_softmax = F.softmax(weighted_aff, dim=2)  # (6300, 1, 6300)

        similarity = aff_softmax

        aff_softmax_reshape = aff_softmax.view(x.shape[0], x.shape[0])  # (6300, 6300)

        output_t = torch.mm(aff_softmax_reshape, v_data)  # (6300, 1024)
        output_t = output_t.view(-1, self.feat_dim, 1, 1)  # (6300, 1024, 1, 1)
        linear_out = self.conv(output_t)  # (6300, 1024, 1, 1)
        output = linear_out.view(linear_out.shape[0], linear_out.shape[1])  # (6300, 1024)

        return output, similarity



# model = SelsaHead()
# for m in model.modules():
#     print(m)
# img = torch.randn(512, 256, 7, 7)
# img.requires_grad = True
# output = model(img)
# print(output.size())