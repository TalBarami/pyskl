import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch.nn.modules.utils import _triple

# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import BACKBONES
from .resnet3d import ResNet3d


@BACKBONES.register_module()
class ResNet3dSlowOnly(ResNet3d):
    """SlowOnly backbone based on ResNet3d.

    Args:
        conv1_kernel (tuple[int]): Kernel size of the first conv layer. Default: (1, 7, 7).
        inflate (tuple[int]): Inflate Dims of each block. Default: (0, 0, 1, 1).
        **kwargs (keyword arguments): Other keywords arguments for 'ResNet3d'.
    """

    def __init__(self, conv1_kernel=(1, 7, 7), inflate=(0, 0, 1, 1), **kwargs):
        super().__init__(conv1_kernel=conv1_kernel, inflate=inflate, **kwargs)


class RoleAwareStem(nn.Module):
    """
    Drop-in replacement for ResNet3d.conv1 (expects a ConvModule-like callable).

    Splits input channels into:
      child: [0:J)
      adult: [J:2J)
      optional extra: [2J:2J+extra)

    Applies separate ConvModules and fuses.
    """
    def __init__(self,
                 num_joints: int,
                 base_channels: int,
                 conv1_kernel,
                 conv1_stride,
                 conv_cfg,
                 norm_cfg,
                 act_cfg,
                 fusion: str = 'sum',
                 extra_channels: int = 0,
                 extra_mode: str = 'ignore'):
        super().__init__()
        assert fusion in ('sum', 'concat')
        assert extra_mode in ('ignore', 'to_child', 'to_adult')

        self.J = int(num_joints)
        self.extra = int(extra_channels)
        self.fusion = fusion
        self.extra_mode = extra_mode

        # Build child/adult input channel counts
        child_in = self.J
        adult_in = self.J
        if self.extra > 0:
            if self.extra_mode == 'to_child':
                child_in += self.extra
            elif self.extra_mode == 'to_adult':
                adult_in += self.extra

        stride = (conv1_stride[0], conv1_stride[1], conv1_stride[1])
        pad = tuple([(k - 1) // 2 for k in _triple(conv1_kernel)])

        self.child = ConvModule(
            child_in, base_channels,
            kernel_size=conv1_kernel,
            stride=stride,
            padding=pad,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.adult = ConvModule(
            adult_in, base_channels,
            kernel_size=conv1_kernel,
            stride=stride,
            padding=pad,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        if fusion == 'concat':
            self.fuse = ConvModule(
                2 * base_channels, base_channels,
                kernel_size=1, stride=1, padding=0,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        else:
            self.fuse = None

    def forward(self, x):
        J = self.J
        child = x[:, :J]
        adult = x[:, J:2*J]

        if self.extra > 0 and self.extra_mode != 'ignore':
            extra = x[:, 2*J:2*J + self.extra]
            if self.extra_mode == 'to_child':
                child = torch.cat([child, extra], dim=1)
            else:
                adult = torch.cat([adult, extra], dim=1)

        y_child = self.child(child)
        y_adult = self.adult(adult)

        if self.fusion == 'sum':
            return y_child + y_adult

        return self.fuse(torch.cat([y_child, y_adult], dim=1))

@BACKBONES.register_module()
class RoleAwareResNet3dSlowOnly(ResNet3dSlowOnly):
    def __init__(self,
                 num_joints: int = 17,
                 fusion: str = 'sum',
                 extra_channels: int = 0,
                 extra_mode: str = 'ignore',
                 **kwargs):
        self.num_joints = int(num_joints)
        self.fusion = fusion
        self.extra_channels = int(extra_channels)
        self.extra_mode = extra_mode

        in_channels = 2 * self.num_joints + self.extra_channels
        kwargs = dict(kwargs)
        kwargs['in_channels'] = in_channels
        super().__init__(**kwargs)

        self.conv1 = RoleAwareStem(
            num_joints=self.num_joints,
            base_channels=self.base_channels,
            conv1_kernel=self.conv1_kernel,
            conv1_stride=self.conv1_stride,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            fusion=fusion,
            extra_channels=self.extra_channels,
            extra_mode=self.extra_mode
        )

    def init_weights(self, pretrained=None):
        # 1) Let parent init (kaiming) + load checkpoint (strict=False)
        super().init_weights(pretrained=pretrained)

        # If no checkpoint provided, nothing else to do
        if not isinstance(self.pretrained, str):
            return

        # 2) Load checkpoint state_dict on CPU
        ckpt = torch.load(self.pretrained, map_location='cpu')
        sd = ckpt.get('state_dict', ckpt)

        # helper: find key with optional backbone prefix
        def get_key(*candidates):
            for k in candidates:
                if k in sd:
                    return k
            return None

        # 3) Pull original conv1 ConvModule params (conv+bn) from checkpoint
        k_conv_w = get_key('conv1.conv.weight', 'backbone.conv1.conv.weight')
        if k_conv_w is None:
            # nothing to map (unexpected for pose-heatmap K400)
            return

        w = sd[k_conv_w]  # [base_channels, 17, 1, 7, 7] typically

        # scale if we fuse by sum (recommended)
        scale = 0.5 if self.fusion == 'sum' else 1.0
        w = w * scale

        # 4) Copy into both stems
        with torch.no_grad():
            # conv weights
            self.conv1.child.conv.weight.copy_(w)
            self.conv1.adult.conv.weight.copy_(w)

            # BN params/buffers (if exist)
            # ConvModule in mmcv usually has .bn with weight/bias/running_mean/running_var
            for attr in ['weight', 'bias', 'running_mean', 'running_var']:
                k_bn = get_key(f'conv1.bn.{attr}', f'backbone.conv1.bn.{attr}')
                if k_bn is not None and hasattr(self.conv1.child.bn, attr):
                    getattr(self.conv1.child.bn, attr).copy_(sd[k_bn])
                    getattr(self.conv1.adult.bn, attr).copy_(sd[k_bn])

            # num_batches_tracked is optional
            k_nbt = get_key('conv1.bn.num_batches_tracked', 'backbone.conv1.bn.num_batches_tracked')
            if k_nbt is not None and hasattr(self.conv1.child.bn, 'num_batches_tracked'):
                self.conv1.child.bn.num_batches_tracked.copy_(sd[k_nbt])
                self.conv1.adult.bn.num_batches_tracked.copy_(sd[k_nbt])