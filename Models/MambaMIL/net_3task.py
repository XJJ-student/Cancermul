import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath, PatchEmbed
from timm.models.vision_transformer import _load_weights

import math
from collections import namedtuple
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
import random
import faiss

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

# --------------------------------------------------------
# EVA-02: A Visual Representation for Neon Genesis
# Github source: https://github.com/baaivision/EVA/EVA02
# Copyright (c) 2023 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Yuxin Fang
#
# Based on https://github.com/lucidrains/rotary-embedding-torch
# --------------------------------------------------------'


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )    
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    bimamba_type="none",
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)



class MambaMIL(nn.Module):
    def __init__(self, 
                 num_classes, 
                 num_classes_subtype, 
                 num_classes_stage, 
                 num_pathway_in_function_RNA, 
                 num_pathway_in_function_DNA,
                 input_size, 
                 img_size=224, 
                 patch_size=16, 
                 depth=24, 
                 embed_dim=512, 
                 channels=3, 
                 ssm_cfg=None, 
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5, 
                 rms_norm: bool = False, 
                 initializer_cfg=None,
                 residual_in_fp32=False,
                 pt_hw_seq_len=14,
                 final_pool_type='none',
                 if_abs_pos_embed=False):
        super(MambaMIL, self).__init__()
        
        self.num_classes = num_classes
        self.num_classes_subtype = num_classes_subtype
        self.num_classes_stage = num_classes_stage

        self.num_pathway_in_function_RNA = num_pathway_in_function_RNA  
        self.pathway_linear_layers_RNA =  [nn.ModuleList([]).cuda() for i in self.num_pathway_in_function_RNA]
        function_count = 0
        for num_pathway in num_pathway_in_function_RNA:
            for i in range(len(num_pathway)):
                self.pathway_linear_layers_RNA[function_count].append(nn.Sequential(
                    nn.LayerNorm(num_pathway[i]),
                    nn.Linear(num_pathway[i], 512),
                    nn.LayerNorm(512)
                ).cuda())
            function_count = function_count + 1
          
        self.num_pathway_in_function_DNA = num_pathway_in_function_DNA  
        self.pathway_linear_layers_DNA =  [nn.ModuleList([]).cuda() for i in self.num_pathway_in_function_DNA]
        function_count = 0
        for num_pathway in num_pathway_in_function_DNA:
            for i in range(len(num_pathway)):
                self.pathway_linear_layers_DNA[function_count].append(nn.Sequential(
                    nn.LayerNorm(num_pathway[i]),
                    nn.Linear(num_pathway[i], 512),
                    nn.LayerNorm(512)
                ).cuda())
            function_count = function_count + 1
       
        self._fc1 = nn.Sequential(nn.Linear(input_size, 512), nn.ReLU())
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.num_classes)
        self._fc3 = nn.Linear(512, self.num_classes_subtype)
        self._fc4 = nn.Linear(512, self.num_classes_stage)

        self.residual_in_fp32 = residual_in_fp32
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed

        self.d_model = self.num_features = self.embed_dim = embed_dim
        

        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=True,
                    layer_idx=i,
                    bimamba_type='v2',
                    drop_path=inter_dpr[i],
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon)

        # original init
        self.apply(segm_init_weights)

        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        
        self.cross_mamba = Mamba(512,bimamba_type="v3")
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        
        self.fusion_w = nn.Parameter(torch.ones(2))

    def forward(self, bag_wsi_feats, bag_rna_feats,bag_dna_feats,  inference_params=None):
        #####################################################################
        # wsi
        region_wsi = []
        for wsi_group in bag_wsi_feats:
            # 每个group的patch
            residual = None
            wsi_group = wsi_group.unsqueeze(0)
            patch_hidden_states = self._fc1(wsi_group)

            for layer in self.layers:
                patch_hidden_states, residual = layer(
                    patch_hidden_states, residual, inference_params=inference_params
                )

            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            patch_hidden_states = fused_add_norm_fn(
                self.drop_path(patch_hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
            patch_hidden_states = patch_hidden_states.mean(dim=1)
            region_wsi.append(patch_hidden_states)

        region_wsi = torch.cat(region_wsi, dim=-2)
        region_wsi = region_wsi.unsqueeze(0)
        region_wsi_fusion = region_wsi# [1, ]

        residual = None
        region_hidden_states = region_wsi
        
        if region_hidden_states.shape[1]>1:
            for layer in self.layers:
                region_hidden_states, residual = layer(
                    region_hidden_states, residual, inference_params=inference_params
                )

            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            region_hidden_states = fused_add_norm_fn(
                self.drop_path(region_hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        wsi_feature = self.norm(region_hidden_states)
        
        #####################################################################
        # RNA
        function_rna = []
        rna_group_count = 0
        for rna_group in bag_rna_feats:
          
            segments = torch.split(rna_group, self.num_pathway_in_function_RNA[rna_group_count], dim=-1)
            segnum = 0
            transformed_tensors = []
            
            
            for layers in self.pathway_linear_layers_RNA[rna_group_count]:

                x = segments[segnum]
                
                per_segment = layers(x)
                transformed_tensors.append(per_segment)
                segnum = segnum + 1
            rna_group_count += 1

            rna_group_pathway = torch.cat(transformed_tensors, dim=-2)

            if rna_group_pathway.shape[1]>1:
                residual = None
                pathway_hidden_states = rna_group_pathway

                for layer in self.layers:
                    pathway_hidden_states, residual = layer(
                        pathway_hidden_states, residual, inference_params=inference_params
                    )

                # Set prenorm=False here since we don't need the residual
                fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
                pathway_hidden_states = fused_add_norm_fn(
                    self.drop_path(pathway_hidden_states),
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
                pathway_hidden_states = pathway_hidden_states.mean(dim=1)
                function_rna.append(pathway_hidden_states)
            
            else:

                function_rna.append(rna_group_pathway.squeeze(0))

        # DNA
        function_dna = []
        dna_group_count = 0
        for dna_group in bag_dna_feats:
            segments = torch.split(dna_group, self.num_pathway_in_function_DNA[dna_group_count], dim=-1)
            segnum = 0
            transformed_tensors = []

            for layers in self.pathway_linear_layers_DNA[dna_group_count]:
                x = segments[segnum]
                per_segment = layers(x)
                transformed_tensors.append(per_segment)
                segnum = segnum + 1
            dna_group_count += 1

            dna_group_pathway = torch.cat(transformed_tensors, dim=-2)

            if dna_group_pathway.shape[1]>1:
                residual = None
                pathway_hidden_states = dna_group_pathway

                for layer in self.layers:
                    pathway_hidden_states, residual = layer(
                        pathway_hidden_states, residual, inference_params=inference_params
                    )

                # Set prenorm=False here since we don't need the residual
                fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
                pathway_hidden_states = fused_add_norm_fn(
                    self.drop_path(pathway_hidden_states),
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
                pathway_hidden_states = pathway_hidden_states.mean(dim=1)
                function_dna.append(pathway_hidden_states)
            
            else:

                function_dna.append(dna_group_pathway.squeeze(0))

        # DNA与RNA融合
        
        function_rna = torch.cat(function_rna, dim=-2)
        function_dna = torch.cat(function_dna, dim=-2)
        
        function_rna = torch.add(function_rna, function_dna)
        function_rna = function_rna.unsqueeze(0)
        function_rna_fusion = function_rna
        
        
        residual = None
        function_hidden_states = function_rna
        
 
        for layer in self.layers:
            function_hidden_states, residual = layer(
                function_hidden_states, residual, inference_params=inference_params
            )

        # Set prenorm=False here since we don't need the residual
        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
        function_hidden_states = fused_add_norm_fn(
            self.drop_path(function_hidden_states),
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=self.residual_in_fp32,
        )
        
        rna_feature = self.norm(function_hidden_states)

        
        # 数据融合
        region_wsi_fusion = self.norm1(region_wsi_fusion)
        function_rna_fusion = self.norm2(function_rna_fusion)
        

        if region_wsi_fusion.shape[1]>1 and function_rna_fusion.shape[1]>1:
            low_level_fusion = self.cross_mamba(self.norm1(region_wsi_fusion),extra_emb=self.norm2(function_rna_fusion))
        else:
            low_level_fusion = torch.cat([region_wsi_fusion, function_rna_fusion], dim=-2)
            
        if wsi_feature.shape[1]>1 and rna_feature.shape[1]>1:
            high_level_fusion = self.cross_mamba(self.norm1(wsi_feature),extra_emb=self.norm2(rna_feature))
        else:
            high_level_fusion = torch.cat([wsi_feature, rna_feature], dim=-2)

        # 归一化权重
        w1 = torch.exp(self.fusion_w[0]) / torch.sum(torch.exp(self.fusion_w))
        w2 = torch.exp(self.fusion_w[1]) / torch.sum(torch.exp(self.fusion_w))

        # 自适应融合
        final_feat = low_level_fusion * w1 + high_level_fusion * w2
        final_feat = final_feat.mean(dim=-2)
        
        
        logits = self._fc2(final_feat) # [1, 512]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        logits_subtype = self._fc3(final_feat) # [1, 512]
        logits_stage = self._fc4(final_feat) # [1, 512]
        
        
        results_dict = {'logits': hazards, 'S':S, 'subtype':logits_subtype,'stage':logits_stage}
        
        return results_dict