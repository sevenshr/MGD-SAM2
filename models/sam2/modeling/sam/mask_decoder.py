# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Optional, Tuple, Type

from ....sam2.modeling.sam2_utils import LayerNorm2d, MLP

from einops import rearrange
from torch import nn, Tensor
import warnings
import math
from ....sam2.modeling.sam.transformer import Attention


def get_sdpa_settings():
    if torch.cuda.is_available():
        old_gpu = torch.cuda.get_device_properties(0).major < 7
        # only use Flash Attention on Ampere (8.0) or newer GPUs
        use_flash_attn = torch.cuda.get_device_properties(0).major >= 8
        if not use_flash_attn:
            warnings.warn(
                "Flash Attention is disabled as it requires a GPU with Ampere (8.0) CUDA capability.",
                category=UserWarning,
                stacklevel=2,
            )
        # keep math kernel for PyTorch versions before 2.2 (Flash Attention v2 is only
        # available on PyTorch 2.2+, while Flash Attention v1 cannot handle all cases)
        pytorch_version = tuple(int(v) for v in torch.__version__.split(".")[:2])
        if pytorch_version < (2, 2):
            warnings.warn(
                f"You are using PyTorch {torch.__version__} without Flash Attention v2 support. "
                "Consider upgrading to PyTorch 2.2+ for Flash Attention v2 (which could be faster).",
                category=UserWarning,
                stacklevel=2,
            )
        math_kernel_on = pytorch_version < (2, 2) or not use_flash_attn
    else:
        old_gpu = True
        use_flash_attn = False
        math_kernel_on = True

    return old_gpu, use_flash_attn, math_kernel_on

OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = get_sdpa_settings()

def image2patches(x):
    """b c (hg h) (wg w) -> (hg wg b) c h w"""
    x = rearrange(x, 'b c (hg h) (wg w) -> b (hg wg) c h w', hg=2, wg=2)
    x = rearrange(x, 'b u c h w -> (b u) c h w')
    return x

def patches2image(x):
    """(hg wg b) c h w -> b c (hg h) (wg w)"""
    x = rearrange(x, '(b u) c h w -> b u c h w', u=4)
    x = rearrange(x, 'b (hg wg) c h w -> b c (hg h) (wg w)', hg=2, wg=2)
    return x



class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 1,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = True,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = True,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        # When outputting a single mask, optionally we can dynamically fall back to the best
        # multimask output token if the single mask output token gives low stability scores.
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

        self.new_multifieldcrossatt = MCEM(transformer_dim,num_heads = 8, 
                                    attention_downsample_rate =2,mlp_dim = 512)

        self.new_fusion1 = HMIM(transformer_dim)
        self.new_fusion2 = HMIM(transformer_dim)


    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: batched SAM token for mask output
        """
        masks, iou_pred, mask_tokens_out, object_score_logits, upscaled_embedding = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
        else:
            # Take the mask output token. Here we *always* use the token for single mask output.
            # At test time, even if we track after 1-click (and using multimask_output=True),
            # we still take the single mask token here. The rationale is that we always track
            # after multiple clicks during training, so the past tokens seen during training
            # are always the single mask token (and we'll let it be the object-memory token).
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape

        # Prepare output
        return masks, iou_pred, sam_tokens_out, object_score_logits, upscaled_embedding

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        
        src = src + dense_prompt_embeddings
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features

            # multi-view interation in MCEM and HMIM
            b_c = src.shape[0]//5
            loc_src, glb_scr = src.split([4 * b_c, 1 * b_c], dim=0)
            src, x_glb_pre = self.new_multifieldcrossatt(loc_src, glb_scr)  
            feat_s1, x_glb_pre2 = self.new_fusion2(feat_s1, x_glb_pre, b_c)
            feat_s0, _ = self.new_fusion1(feat_s0, x_glb_pre2, b_c)


            feat_s0 = self.conv_s0(feat_s0)
            feat_s1 = self.conv_s1(feat_s1)

            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)
        
        return masks, iou_pred, mask_tokens_out, object_score_logits, upscaled_embedding

    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds, similar to https://github.com/fairinternal/onevision/pull/568.
        """
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out


class HMIM(nn.Module):
    def __init__(self,dim=256):
        super().__init__()
        

        self.locpre2glb = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim)
        self.fuse1 = nn.Conv2d(dim * 2, dim, kernel_size=1)

        self.glb2loc = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim)
        self.fuse2 = nn.Conv2d(dim * 2, dim, kernel_size=1)


        self.fuse3 = nn.Sequential(
            LayerNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim , dim, kernel_size=1)
        )

       
    def forward(self, x, x_locs_pre, b):
        """x:transformer, y:cnn"""

        x_loc, x_glb = x.split([4 * b, 1 * b], dim=0)
        concated_locs = patches2image(x_loc)

        high_x_locs_pre = self.locpre2glb(x_locs_pre)
        x_glb_enh =  self.fuse1(torch.cat([x_glb, high_x_locs_pre],dim=1)) + x_glb
        
        low_x_glb_enh_up = F.interpolate(x_glb_enh, scale_factor=2, mode='bilinear')
        x_glb_sup = self.glb2loc(low_x_glb_enh_up)
        x_locs_enh = self.fuse2( torch.cat([x_glb_sup, concated_locs],dim=1)) + concated_locs
        x_loc_enh = image2patches(x_locs_enh)

        x = torch.cat((x_loc_enh, x_glb_enh), 0)
        x = self.fuse3(x)
        x_locs_later_use, _ = x.split([4 * b, 1 * b], dim=0)
        x_locs_later_use = patches2image(x_locs_later_use)

        return x,x_locs_later_use

    

class MCEM(nn.Module):

    def __init__(
        self,
        d_model, 
        num_heads,
        attention_downsample_rate: int = 2,
        mlp_dim: int = 2048,
        dropout=0,
        activation: Type[nn.Module] = nn.ReLU,
    ):
        super(MCEM, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.mlp1 = MLP(
            d_model, mlp_dim, d_model, num_layers=2
        )
        self.mlp2 = MLP(
            d_model, mlp_dim, d_model, num_layers=2
        )
        self.attention = nn.ModuleList([
            Attention(d_model, num_heads, downsample_rate=attention_downsample_rate, dropout=dropout),
            Attention(d_model, num_heads, downsample_rate=attention_downsample_rate, dropout=dropout),
            Attention(d_model, num_heads, downsample_rate=attention_downsample_rate, dropout=dropout),
            Attention(d_model, num_heads, downsample_rate=attention_downsample_rate, dropout=dropout),
            Attention(d_model, num_heads, downsample_rate=attention_downsample_rate, dropout=dropout)
            
        ])

        self.pe = None
        self.positional_encoding = PositionEmbeddingSine2(num_pos_feats=d_model // 2, normalize=True)

    def forward(self,l,g):
        """
        l: 4b,c,h,w
        g: 1b,c,h,w
        """
        b, c, h, w = g.size() 
        concated_locs = patches2image(l)

        #get position encoding
        if self.pe is None:
            self.pe = self.positional_encoding(b, h * 2, w * 2)
        pe = F.interpolate(self.pe, size=(h* 2, w* 2), mode="bicubic")
        pos_emb = F.interpolate(pe, size=g.shape[-2:],  mode="bicubic")
        self.pos_emb_g = rearrange(pos_emb, 'b c h w -> b (h w) c')
        self.pos_emb_concat_l = rearrange(pe, 'b c h w ->b (h w) c')
        self.pos_emb_split_l = rearrange(pe, " b c (ng h) (nw w) -> (ng nw) b (h w) c", ng=2, nw=2)
        self.pos_emb_split_g = rearrange(pos_emb, " b c (ng h) (nw w) -> (ng nw) b (h w) c", ng=2, nw=2)

        #local to global attention
        g_flatten = rearrange(g, 'b c h w -> b (h w) c')
        concated_locs_flatten = rearrange(concated_locs, 'b c h w ->b (h w) c')
        g_flatten = g_flatten + self.attention[0](g_flatten + self.pos_emb_g, concated_locs_flatten + self.pos_emb_concat_l, concated_locs_flatten)
        g_flatten = self.norm1(g_flatten)
        g_flatten = g_flatten + self.mlp1(g_flatten)
        g_flatten = self.norm2(g_flatten)
        

        #global to local attention
        l_split = rearrange(concated_locs, "b c (ng h) (nw w) -> (ng nw) b (h w) c", ng=2, nw=2)
        g_split = rearrange(g_flatten, 'b (h w) c -> b h w c', h=h, w=w)
        g_split = rearrange(g_split, " b (ng h) (nw w) c -> (ng nw) b (h w) c", ng=2, nw=2)
        outputs_re = []
        for i in range(4):

            outputs_re.append(self.attention[i + 1](l_split[i]+self.pos_emb_split_l[i], g_split[i]+self.pos_emb_split_g[i], g_split[i]))
        outputs_re = torch.cat(outputs_re, 0)  
        outputs_re = rearrange(outputs_re, 'b (h w)  c  -> b c h w', h=h, w=w)
        outputs_re = rearrange(outputs_re, '(hg wg b) c h w -> b c (hg h) (wg w)', hg=2, wg=2)
        outputs_re = rearrange(outputs_re, 'b c h w  -> b (h w) c', h=h*2, w=w*2)

        l_flatten = concated_locs_flatten + outputs_re

        l_flatten = self.norm3(l_flatten)
        l_flatten = l_flatten + self.mlp2(l_flatten)
        l_flatten = self.norm4(l_flatten)  
        new_concated_locs = rearrange(l_flatten, " b (h w) c -> b c h w", h=h*2, w=w*2)
        new_l = image2patches(new_concated_locs)
        new_g = rearrange(g_flatten, " b (h w) c -> b c h w", h=h, w=w)
        output = torch.cat((new_l, new_g), 0)  

        return output, new_concated_locs 


class PositionEmbeddingSine2:
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.dim_t = torch.arange(0, self.num_pos_feats, dtype=torch.float32, device='cuda') 

    def __call__(self, b, h, w):
        mask = torch.zeros([1, h, w], dtype=torch.bool, device='cuda')
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(dim=1, dtype=torch.float32)  
        x_embed = not_mask.cumsum(dim=2, dtype=torch.float32)  
        if self.normalize:
            eps = 1e-6
            y_embed = ((y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale).cuda()
            x_embed = ((x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale).cuda()

        dim_t = self.temperature ** (2 * (self.dim_t // 2) / self.num_pos_feats) 

        pos_x = x_embed[:, :, :, None] / dim_t 
        pos_y = y_embed[:, :, :, None] / dim_t  
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(
            3)  
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(
            3)  
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2) 