import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init, normal_init
from mmcv.runner import load_checkpoint

from mmdet.core import auto_fp16
from mmdet.utils import get_root_logger
from ..backbones import ResNeXt
from ..backbones.resnext import make_res_layer
from ..registry import SHARED_HEADS


@SHARED_HEADS.register_module
class ResxLayer(nn.Module):

    def __init__(self,
                 depth,
                 stage=3,
                 stride=2,
                 dilation=1,
                 style='pytorch',
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 with_cp=False,
                 dcn=None):
        super(ResxLayer, self).__init__()
        self.norm_eval = norm_eval
        self.norm_cfg = norm_cfg
        self.stage = stage
        self.fp16_enabled = False
        self.groups = 32
        self.base_width = 4
        self.style = style
        self.with_cp = with_cp
        self.norm_cfg = norm_cfg
        block, stage_blocks = ResNeXt.arch_settings[depth]
        stage_block = stage_blocks[stage]
        planes = 64 * 2**stage
        inplanes = 64 * 2**(stage - 1) * block.expansion

        num_blocks = stage_block
        res_layer = make_res_layer(
            block,
            inplanes,
            planes,
            num_blocks,
            stride=stride,
            dilation=dilation,
            groups=self.groups,
            base_width=self.base_width,
            style=self.style,
            with_cp=self.with_cp,
            conv_cfg=None,
            norm_cfg=self.norm_cfg,
            dcn=None,
            gcb=None)
        self.add_module('layer{}'.format(stage + 1), res_layer)
        self.conv_256 = nn.Conv2d(inplanes * 2, 256, kernel_size=1, groups=2)
        self.relu_256 = nn.ReLU(inplace=True)
        self.with_PixelAGG = False
        if self.with_PixelAGG:
            from mmdet.ops.non_local import PixelAGG
            # from mmdet.ops.non_local_c import PixelAGG_C
            self.pixle_agg = PixelAGG(in_channels=2048, reduction=8, use_scale=False, zeros_init=True)  # reduction=8
            # self.pixle_agg_c = PixelAGG_C(in_channels=256, reduction=8, use_scale=False, zeros_init=True) #reduction=8

    def init_weights(self, pretrained=None):
        normal_init(self.conv_256, 0, 0.01)
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    @auto_fp16()
    def forward(self, x):
        res_layer = getattr(self, 'layer{}'.format(self.stage + 1))
        out = res_layer(x)
        if self.with_PixelAGG:
            out = self.pixle_agg(out)
        out = self.conv_256(out)
        # out = self.bn_256(out)
        out = self.relu_256(out)
        return out

    def train(self, mode=True):
        super(ResxLayer, self).train(mode)
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
