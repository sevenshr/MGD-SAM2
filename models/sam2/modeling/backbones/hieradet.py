# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from iopath.common.file_io import g_pathmgr

from ....sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from ....sam2.modeling.sam2_utils import DropPath, MLP
from ....sam2.modeling.sam2_utils import LayerNorm2d
from einops import rearrange
import math

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs


def resize_as(x, y, interpolation='bilinear'):
    return F.interpolate(x, size=y.shape[-2:], mode=interpolation)

def image2patches(x):
    """b c q (hg h) (wg w) -> (hg wg b) c q h w"""
    x = rearrange(x, 'b c q (hg h) (wg w) -> b (hg wg) c q h w', hg=2, wg=2)
    x = rearrange(x, 'b u c q h w -> (b u) c q h w')
    return x

def patches2image(x):
    """(hg wg b) c h w -> b c (hg h) (wg w)"""
    x = rearrange(x, '(b u) c h w -> b u c h w', u=4)
    x = rearrange(x, 'b (hg wg) c h w -> b c (hg h) (wg w)', hg=2, wg=2)
    return x

def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,
        encoder_mode = None,
        in_chans=3,
        embed_dims=[96, 192, 384, 768],  #[144, 288, 576, 1152]
        img_size: int = 1024,
        patch_size: int = 4,

        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        weights_path=None,
        return_interm_layers=True,  # return feats from every stage
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
        )
        
        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        cur_stage = 1
        self.blocks = nn.ModuleList()
        self.embed_dim = embed_dims
        self.depth = stages
        self.scale_factor = encoder_mode['scale_factor']

        self.tuning_stage = "1234"

        self.adaptor = 'adaptor'
        self.mpadapter = MPAdapter(self.scale_factor, self.embed_dim,
                                                self.tuning_stage, self.depth,
                                                self.adaptor,
                                                )


        for i in range(depth):
            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

        if weights_path is not None:
            with g_pathmgr.open(weights_path, "rb") as f:
                chkpt = torch.load(f, map_location="cpu")
            logging.info("loading Hiera", self.load_state_dict(chkpt, strict=False))

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        inp = x
        x = self.patch_embed(x)
        # x: (B, H, W, C)
        
        self.block1 = []
        self.block2 = []
        self.block3 = []
        self.block4 = []
        outputs = []

        for i, blk in enumerate(self.blocks):
            if i < self.stage_ends[0]+2:
                self.block1.append(blk)  
            elif self.stage_ends[0]+1 < i < self.stage_ends[1]+2:
                self.block2.append(blk)  
            elif self.stage_ends[1]+1 < i < self.stage_ends[2]+2:
                self.block3.append(blk) 
            elif self.stage_ends[2]+1 < i:
                self.block4.append(blk)  

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])


        for i, blk in enumerate(self.block1):
            if '1' in self.tuning_stage:
                x = self.mpadapter.get_prompt(x, 1, i)
            x = blk(x)
            

            if i == self.depth[0]-1:
                feat1 = x.permute(0, 3, 1, 2)
                outputs.append(feat1)

        for i, blk in enumerate(self.block2):
            if '2' in self.tuning_stage:
                x = self.mpadapter.get_prompt(x, 2, i)
            x = blk(x)
            
            if i == self.depth[1]-2:

                feat2 = x.permute(0, 3, 1, 2)
                outputs.append(feat2)


        for i, blk in enumerate(self.block3):
            if '3' in self.tuning_stage:
                x = self.mpadapter.get_prompt(x, 3, i)
            x = blk(x)
            

            if i == self.depth[2]-2:
                feat3 = x.permute(0, 3, 1, 2)
                outputs.append(feat3)


        for i, blk in enumerate(self.block4):            
            if '4' in self.tuning_stage:
                x = self.mpadapter.get_prompt(x, 4, i)
            x = blk(x)
            

            if i == self.depth[3]-2:
                feat4 = x.permute(0, 3, 1, 2)
                outputs.append(feat4)


        return outputs

    def get_layer_id(self, layer_name):
        # https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        num_layers = self.get_num_layers()

        if layer_name.find("rel_pos") != -1:
            return num_layers + 1
        elif layer_name.find("pos_embed") != -1:
            return 0
        elif layer_name.find("patch_embed") != -1:
            return 0
        elif layer_name.find("blocks") != -1:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers + 1

    def get_num_layers(self) -> int:
        return len(self.blocks)


def to_2tuple(x):
    if isinstance(x, container_abcs.Iterable):
        return x
    return tuple(repeat(x, 2))

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor



class MPAdapter(nn.Module):
    def __init__(self, scale_factor, embed_dims, tuning_stage, depths, adaptor):
        """
        Args:
        """
        super(MPAdapter, self).__init__()
        self.scale_factor = scale_factor
        self.embed_dims = embed_dims

        self.tuning_stage = tuning_stage
        self.depths = depths

        self.adaptor = adaptor


        if self.adaptor == 'adaptor':
            if '1' in self.tuning_stage:
                for i in range(self.depths[0]+1):
                    lightweight_mlp = nn.Sequential(
                        nn.Linear(self.embed_dims[0], self.embed_dims[0] // self.scale_factor),
                            DWConv3D_adapter(self.embed_dims[0] // self.scale_factor),
                            nn.GELU(),
                            nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0])
                        )
                    setattr(self, 'lightweight_mlp1_{}'.format(str(i)), lightweight_mlp)
                self.shared_mlp1 = nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0])

            if '2' in self.tuning_stage:
                for i in range(self.depths[1]+1):
                    lightweight_mlp = nn.Sequential(
                        nn.Linear(self.embed_dims[1], self.embed_dims[1] // self.scale_factor),
                            DWConv3D_adapter(self.embed_dims[1] // self.scale_factor),
                            nn.GELU(),
                            nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1])
                        )
                    setattr(self, 'lightweight_mlp2_{}'.format(str(i)), lightweight_mlp)
                self.shared_mlp2 = nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1])

            if '3' in self.tuning_stage:
                for i in range(self.depths[2]+1):
                    lightweight_mlp = nn.Sequential(
                        nn.Linear(self.embed_dims[2], self.embed_dims[2] // self.scale_factor),
                            DWConv3D_adapter(self.embed_dims[2] // self.scale_factor),
                            nn.GELU(),
                            nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2])
                        )
                    setattr(self, 'lightweight_mlp3_{}'.format(str(i)), lightweight_mlp)
                self.shared_mlp3 = nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2])

            if '4' in self.tuning_stage:
                for i in range(self.depths[3]+1):
                    lightweight_mlp = nn.Sequential(
                        nn.Linear(self.embed_dims[3], self.embed_dims[3] // self.scale_factor),
                            DWConv3D_adapter(self.embed_dims[3] // self.scale_factor),
                            nn.GELU(),
                            nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3])
                        )
                    setattr(self, 'lightweight_mlp4_{}'.format(str(i)), lightweight_mlp)
                self.shared_mlp4 = nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3])


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()



    def get_prompt(self, x, block_num, depth_num):


        if self.adaptor == 'adaptor':
            lightweight_mlp = getattr(self, 'lightweight_mlp' + str(block_num) + '_' + str(depth_num))

            feat = lightweight_mlp(x)

        x = x + feat

        return x



class DWConv3D_adapter(nn.Module):
    def __init__(self, dim=768):
        super(DWConv3D_adapter, self).__init__()
        self.dwconv_loc = nn.Conv3d(
            dim, dim,
            kernel_size=(2,3,3),
            stride=(2, 1, 1),
            padding=(0,1,1),
            groups=dim,
        )

        self.dwconv_glb = nn.Conv3d(
            dim, dim,
            kernel_size=(2,3,3),
            stride=(2, 1, 1),
            padding=(0,1,1),
            groups=dim,
        )



    def forward(self, x):

        B, H, W, C = x.shape
        b = B//5
        res = x
        x = x.permute(0,3,1,2).contiguous()
        x_loc, x_glb = x.split([4 * b, 1 * b], dim=0)

        concated_locs = patches2image(x_loc)
        loc = resize_as(concated_locs, x_glb)

        loc2glb = torch.stack([x_glb,loc],dim=2)

        glb = resize_as(x_glb, concated_locs)
        glb2loc = torch.stack([concated_locs,glb],dim=2)

        new_glb = self.dwconv_glb(loc2glb) 
        new_concated_locs = self.dwconv_loc(glb2loc)
        new_loc = image2patches(new_concated_locs)

        x = torch.cat((new_loc, new_glb), 0)
        
        x = x.squeeze(2).permute(0,2,3,1).contiguous()

        x = (x + res )

        return x