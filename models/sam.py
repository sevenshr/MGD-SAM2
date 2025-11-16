import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from .sam2.modeling.sam import MaskDecoder, TwoWayTransformer
logger = logging.getLogger(__name__)
from .iou_loss import IOU
from typing import Any, Optional, Tuple
from .sam2.modeling.backbones.image_encoder import ImageEncoder
from .sam2.modeling.backbones.hieradet import Hiera
from .sam2.modeling.backbones.image_encoder import FpnNeck
from torch.nn.init import trunc_normal_

from einops import rearrange

def resize_as(x, y, interpolation='bilinear'):
    return F.interpolate(x, size=y.shape[-2:], mode=interpolation)

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

def rescale_to(x, scale_factor: float = 2, interpolation='nearest'):
    return F.interpolate(x, scale_factor=scale_factor, mode=interpolation)

def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        # print(layer)
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)

def to_2tuple(x):
    return tuple(repeat(x, 2))

class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss

def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W


@register('sam')
class SAM(nn.Module):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None,
        sam_mask_decoder_extra_args=None,
        use_high_res_features_in_sam=True,
        directly_add_no_mem_embed=True,
        iou_prediction_use_sigmoid=False,
        pred_obj_scores: bool = True,
        pred_obj_scores_mlp: bool = False,
        use_obj_ptrs_in_encoder=False,
        fixed_no_obj_ptr: bool = False,
        use_multimask_token_for_obj_ptr: bool = True,
                 ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_size = encoder_mode['patch_size']
        self.embed_dim = encoder_mode['embed_dim']
        self.inp_size = inp_size
        self.image_encoder = ImageEncoder(
            trunk = Hiera(
                img_size = self.inp_size,
                encoder_mode = encoder_mode,
                embed_dims=encoder_mode['backbone_channel_list'][::-1],
                embed_dim=encoder_mode['backbone_channel_list'][-1],
                num_heads=encoder_mode['num_heads_hiera'],
                stages=encoder_mode['stages'],
                window_pos_embed_bkg_spatial_size= encoder_mode['window_pos_embed_bkg_spatial_size'],
                window_spec=encoder_mode['window_spec'],
                global_att_blocks=encoder_mode['global_att_blocks'],
            ),
            neck = FpnNeck(
                d_model = encoder_mode['d_model'],
                backbone_channel_list= encoder_mode['backbone_channel_list'],
                fpn_top_down_levels= encoder_mode['fpn_top_down_levels'],

            ),
            img_size = self.inp_size,
        )
        self._bb_feat_sizes = [
            (int(self.inp_size/4),int(self.inp_size/4)),
            (int(self.inp_size/8),int(self.inp_size/8)),
            (int(self.inp_size/16),int(self.inp_size/16)),
        ]
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid
        self.pred_obj_scores = pred_obj_scores
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=1,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        

        self.detail_enhance = DetailRefinement(img_dim = 32, feature_dim = 32, norm = nn.BatchNorm2d, act = nn.ReLU)

        self.loss_mode = loss
        if self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()

        elif self.loss_mode == 'bbce':
            self.criterionBCE = BBCEWithLogitLoss()

        elif self.loss_mode == 'iou':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.criterionIOU = IOU()

        elif self.loss_mode == 'biou':
            self.criterionBCE = BBCEWithLogitLoss()
            self.criterionIOU = IOU()

        self.cosine_loss = torch.nn.CosineEmbeddingLoss()

        self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        
        
        self.image_embedding_size = self.inp_size // encoder_mode['patch_size']
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])


    def set_input(self, input, gt_mask):
        input = input.to(self.device)
        gt_mask = gt_mask.to(self.device)
        
        glb_img = rescale_to(input, scale_factor=0.5, interpolation='bilinear')
        loc_img = image2patches(input)
        img = torch.cat((loc_img, glb_img), dim=0)

        glb_mask = rescale_to(gt_mask, scale_factor=0.5, interpolation='nearest')
        loc_mask = image2patches(gt_mask)
        mask = torch.cat((loc_mask, glb_mask), dim=0)

        self.input = img
        self.gt_mask = mask
        self.org_img = input
        self.org_mask = gt_mask



    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)


    def forward(self):
        bs = self.input.shape[0]

        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        self.features = self.image_encoder(self.input)
        
        self.features = self.features.copy()
        assert len(self.features["backbone_fpn"]) == len(self.features["vision_pos_enc"])
        assert len(self.features["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = self.features["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = self.features["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        _, vision_feats, _, _ = self.features, vision_feats, vision_pos_embeds, feat_sizes

        if self.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, 1, 256).to('cuda'))

        feats = [
            feat.permute(1, 2, 0).view(bs, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}


        high_res_features = self._features["high_res_feats"]


        low_res_masks, iou_predictions,sam_output_tokens,object_score_logits, upscaled_embedding = self.sam_mask_decoder(
            image_embeddings=self._features["image_embed"],
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
        )

        # Upscale the masks to the original image resolution
        masks_aux = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)


        masks = self.detail_enhance(self.org_img, upscaled_embedding, low_res_masks)

        self.pred_mask = masks
        self.pred_mask_aux = masks_aux

    def infer(self, input):

        img = input
        glb_img = rescale_to(input, scale_factor=0.5, interpolation='bilinear')
        loc_img = image2patches(input)
        input = torch.cat((loc_img, glb_img), dim=0)

        bs = input.shape[0]

        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        self.features = self.image_encoder(input)
        

        self.features = self.features.copy()
        assert len(self.features["backbone_fpn"]) == len(self.features["vision_pos_enc"])
        assert len(self.features["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = self.features["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = self.features["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        _, vision_feats, _, _ = self.features, vision_feats, vision_pos_embeds, feat_sizes

        if self.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, 1, 256).to('cuda'))
        feats = [
            feat.permute(1, 2, 0).view(bs, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        high_res_features = self._features["high_res_feats"]
        # Predict masks
        low_res_masks, iou_predictions,sam_output_tokens,object_score_logits, upscaled_embedding = self.sam_mask_decoder(
            image_embeddings=self._features["image_embed"],
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
        )


        masks = self.detail_enhance(img, upscaled_embedding, low_res_masks)


        return masks

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        
        masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        
        return masks

    def postprocess_masks2(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        """
        Perform PostProcessing on output masks.
        """
        from .sam2.utils.misc import get_connected_components

        masks = masks.float()
        # if self.max_hole_area > 0:
        if self.max_hole_area > 0:
            # Holes are those connected components in background with area <= self.fill_hole_area
            # (background regions are those with mask scores <= self.mask_threshold)
            mask_flat = masks.flatten(0, 1).unsqueeze(1)  # flatten as 1-channel image
            labels, areas = get_connected_components(mask_flat <= self.mask_threshold)
            is_hole = (labels > 0) & (areas <= self.max_hole_area)
            is_hole = is_hole.reshape_as(masks)
            # We fill holes with a small positive mask score (10.0) to change them to foreground.
            masks = torch.where(is_hole, self.mask_threshold + 10.0, masks)

        if self.max_sprinkle_area > 0:
            labels, areas = get_connected_components(mask_flat > self.mask_threshold)
            is_hole = (labels > 0) & (areas <= self.max_sprinkle_area)
            is_hole = is_hole.reshape_as(masks)
            # We fill holes with negative mask score (-10.0) to change them to background.
            masks = torch.where(is_hole, self.mask_threshold - 10.0, masks)

        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks


    def backward_G(self):

        """Calculate GAN and L1 loss for the generator"""
        bs = self.pred_mask.shape[0] * 4

        self.loss_G = self.criterionBCE(self.pred_mask, self.org_mask)
        self.loss_G_aux =  0.3 * self.criterionBCE(self.pred_mask_aux, self.gt_mask)
        
        if self.loss_mode == 'iou' or self.loss_mode == 'biou':
            self.loss_G += _iou_loss(self.pred_mask, self.org_mask)
            self.loss_G_aux +=  0.3 * _iou_loss(self.pred_mask_aux, self.gt_mask)
        self.loss_G += self.loss_G_aux

        self.loss_G.backward()


    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer.step()  # udpate G's weights

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


class ME(nn.Module):
    def __init__(self, in_features, norm, act):
        super().__init__()

        self.linear = nn.Conv2d(in_features, in_features // 2, kernel_size=1, )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features // 2, in_features // 2, kernel_size=3, padding=3 // 2),
            norm(in_features // 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_features // 2, in_features // 2, kernel_size=5, padding=5 // 2),
            norm(in_features // 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_features // 2, in_features // 2, kernel_size=7, padding=7 // 2),
            norm(in_features // 2))

        self.projector = nn.Sequential(act(),
            nn.Conv2d(in_features // 2 * 3, in_features, kernel_size=1, ),
            norm(in_features),)

        self.na = act()

    def forward(self, x):
        identity = x
        x = self.linear(x)
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = torch.cat([conv1_x,conv2_x,conv3_x],dim=1)


        x = self.projector(x)

        return self.na(x+identity)
    


class DetailRefinement(nn.Module):
    def __init__(self, img_dim, feature_dim, norm, act):
        super().__init__()
        bias = False

        self.img_in_conv = nn.Sequential(
            nn.Conv2d(3, img_dim, 3, padding = 1, bias = bias),
            norm(img_dim),
            act()
        )


        self.img_er = ME(img_dim, norm, act)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim, 32, 3, padding = 1, bias = bias),

            norm(32),
            act(),
            nn.Conv2d(32, 16, 3, padding = 1, bias = bias),
            norm(16),
            act(),
        )


        self.fusion_feature = nn.Conv3d(
            feature_dim, feature_dim,
            kernel_size=(2,3,3),
            stride=(2, 1, 1),
            padding=(0,1,1),
            groups=feature_dim)
        
            
        self.fusion_feature_bn = nn.Sequential(
            norm(feature_dim),
            act(),
        )

        self.out_conv = nn.Conv2d(16, 1, 1)
        
        self.feature_upsample1 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),

        )

        self.feature_upsample2 = nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
            nn.Conv2d(feature_dim, feature_dim, 3, padding = 1, bias = bias),
            norm(feature_dim),
            act(),
        )

        self.feature_upsample3 = nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
            nn.Conv2d(feature_dim, feature_dim, 3, padding = 1, bias = bias),
            norm(feature_dim),
            act(),
        )

    
    def forward(self, img, feature, aux_mask):
        #img 1024×1024，feature 128×128
        bs = feature.shape[0] // 5

        loc_feature, glb_feature = feature.split([4 * bs, 1 * bs], dim=0)

        loc_feature_cat = patches2image(loc_feature)
        glb_feature = self.feature_upsample1(glb_feature)

        feature = torch.stack([loc_feature_cat, glb_feature], dim = 2)

        feature = self.fusion_feature(feature).squeeze(2)

        feature = self.fusion_feature_bn(feature)           # 256×256
         

        img_feature = self.img_in_conv(img)
        img_feature = self.img_er(img_feature) 

        feature = self.feature_upsample2(feature + resize_as(img_feature, feature))
        feature = self.feature_upsample3(feature + resize_as(img_feature, feature))

        out_feature = self.fusion_conv(feature)
        out = self.out_conv(out_feature)

        return out
    
